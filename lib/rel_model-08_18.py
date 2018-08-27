
"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn import PairwiseDistance as pdist
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.lstm.decoder_rnn import DecoderRNN
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math
import ipdb
import os


def extract_grad(grad):
    import ipdb
    ipdb.set_trace()


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1  # 6 = batch size
    rois_per_image = scores.new(num_im)  # scores: [384]; rois_per_image: [6]
    lengths = []
    # we need to sort rois within the same image
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)  # [64, 64, 64, 64, 64, 64]
    # Goes from a TxB packed sequence to a BxT or vice versa
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed

MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=2, nl_edge=2, dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        # self.classes: word list, len is 151; 
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)  # [151.200]; self.classes is gt words list 1~150
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)  # return container [dim_input, embed_dim] array
        self.obj_embed.weight.data = embed_vecs.clone()  # equals embed_vecs, [151, 200]

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        if self.nl_obj > 0:
            self.obj_ctx_rnn = AlternatingHighwayLSTM(
                input_size=self.obj_dim+self.embed_dim+128,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                recurrent_dropout_probability=dropout_rate)

            decoder_inputs_dim = self.hidden_dim
            if self.pass_in_obj_feats_to_decoder:
                decoder_inputs_dim += self.obj_dim + self.embed_dim

            self.decoder_rnn = DecoderRNN(self.classes, embed_dim=self.embed_dim,
                                          inputs_dim=decoder_inputs_dim,
                                          hidden_dim=self.hidden_dim,
                                          recurrent_dropout_probability=dropout_rate)
        else:
            self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + 128, self.num_classes)

        if self.nl_edge > 0:
            input_dim = self.embed_dim
            if self.nl_obj > 0:
                input_dim += self.hidden_dim
            if self.pass_in_obj_feats_to_edge:
                input_dim += self.obj_dim
            self.edge_ctx_rnn = AlternatingHighwayLSTM(input_size=input_dim,
                                                       hidden_size=self.hidden_dim,
                                                       num_layers=self.nl_edge,
                                                       recurrent_dropout_probability=dropout_rate)

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:,2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:,0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def edge_ctx(self, obj_feats, obj_dists, im_inds, obj_preds, box_priors=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings [#boxes, 200]
        obj_embed2 = self.obj_embed2(obj_preds)
        # obj_embed3 = F.softmax(obj_dists, dim=1) @ self.obj_embed3.weight
        # inp_feasts: [#boxes, if 'passinto' == true: obj_ctx + obj_fmaps (512+4096); else obj_ctx (512) dim]
        inp_feats = torch.cat((obj_embed2, obj_feats), 1)

        # Sort by the confidence of the maximum detection.
        confidence = F.softmax(obj_dists, dim=1).data.view(-1)[
            obj_preds.data + arange(obj_preds.data) * self.num_classes]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, confidence, box_priors)

        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]

        # now we're good! unperm
        # edg_ctx: [#boxes, 512]
        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def obj_ctx(self, obj_feats, obj_dists, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None):
        """
        Object context and object classification.
        :param obj_feats: obj_pre_rep, [num_obj, 4096+200+128]
        :param obj_dists: result.rm_obj_dists.detach(), [num_obj, 151]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: od_obj_labels, [num_obj] the GT labels of the image 
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        #ipdb.set_trace()
        # Encode in order
        # Sort by the confidence of the maximum detection; 
        # [384,151]->[384,151]->[384,150]->[384,2], (scores, true_index - 1)-> scores; index is 150-d but truth is 151-d
        confidence = F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[0]
        # sort rois(boxes) within the !same! img according to some order; 
        # a = a[perm][inv_perm]
        # perm: [384]; inv_perm: [384]; ls_transposed: len=64 (6,...6,6)
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, confidence, box_priors)
        # Pass object features, sorted within the same img by score
        obj_inp_rep = obj_feats[perm].contiguous()  # make cache/memory contiguous
        # [6, 64, 151], (batch_size, num_timesteps, input_size)
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        # encoder_rep: [#boxes, 512]
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        
        #ipdb.set_trace()

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(torch.cat((obj_inp_rep, encoder_rep), 1) if self.pass_in_obj_feats_to_decoder else encoder_rep,
                                         ls_transposed)
            # when training sgdet: obj_preds = F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[1] .+ 1
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp, #obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,  # not None when sgdet
                )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data, self.num_classes))


        #obj_preds = Variable(F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[1] + 1)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        :param obj_priors: from faster rcnn output boxes
        :param obj_fmaps: 4096-dim roi feature maps
        :param obj_logits: result.rm_obj_dists.detach()
        :param im_inds:
        :param obj_labels: od_obj_labels, gt
        :param boxes:
        :return: obj_dists2: [#boxes, 151], new score for boxes
                 obj_preds: [#boxes], prediction/class value
                 edge_ctx: [#boxes, 512], new features for boxes

        """

        # Object State:
        # obj_embed: [#boxes, 200], and self.obj_embed.weight are both Variable
        # obj_logits: result.rm_obj_dists.detach(), [#boxes, 151], detector scores before softmax
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        # center_size returns boxes as (center_x, center_y, width, height)
        # pos_embed: [#boxes, 128], Variable, from boxes after Sequential processing
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))
        # obj_pre_rep: [#boxes, 4424], Variable
        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)
        
        if self.nl_obj > 0:
            # obj_dists2: [#boxes, 151], new score for box
            # obj_preds: [#boxes], prediction/class value
            # obj_ctx: [#boxes, 512], new features vector for box
            obj_dists2, obj_preds, obj_ctx = self.obj_ctx(
                obj_pre_rep,  #obj_fmaps,  # original: obj_pre_rep,
                obj_logits,
                im_inds,
                obj_labels,
                box_priors,
                boxes_per_cls,
            )
        else:
            # UNSURE WHAT TO DO HERE
            if self.mode == 'predcls':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
            else:
                obj_dists2 = self.decoder_lin(obj_pre_rep)

            if self.mode == 'sgdet' and not self.training:
                # NMS here for baseline

                probs = F.softmax(obj_dists2, 1)
                nms_mask = obj_dists2.data.clone()
                nms_mask.zero_()
                for c_i in range(1, obj_dists2.size(1)):
                    scores_ci = probs.data[:, c_i]
                    boxes_ci = boxes_per_cls.data[:, c_i]

                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

                obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,1:].max(1)[1] + 1
            else:
                obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1
            obj_ctx = obj_pre_rep
        
        # Edge State:
        edge_ctx = None

        if self.nl_edge > 0:
            # edge_ctx: [#boxes, 512]
            edge_ctx = self.edge_ctx(
                torch.cat((obj_fmaps, obj_ctx), 1) if self.pass_in_obj_feats_to_edge else obj_ctx,
                obj_dists=obj_dists2.detach(),  # Was previously obj_logits.
                im_inds=im_inds,
                obj_preds=obj_preds,
                box_priors=box_priors,
            )

        return obj_dists2, obj_preds, edge_ctx


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision=limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, dropout_rate=rec_dropout,
                                         order=order,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        ###################################
        self.post_lstm = nn.Linear(self.hidden_dim, self.pooling_dim * 2)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_lstm.bias.data.zero_()

        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim*2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()
        
        # not too large; because in the same img, rel class is mostly 0; if too large, most neg rel is repeated
        self.neg_num = 1


        """
        self.embdim = 100 
        self.obj1_fc= nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes * self.embdim, bias=True),
            nn.BatchNorm1d(self.num_classes * self.embdim),
            nn.ReLU(inplace=True),
        )
        self.obj2_fc= nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes * self.embdim, bias=True),
            nn.BatchNorm1d(self.num_classes * self.embdim),
            nn.ReLU(inplace=True),
        )
        self.rel_seq = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_rels * self.embdim, bias=True),
            nn.BatchNorm1d(self.num_rels * self.embdim),
            nn.ReLU(inplace=True),
        )
        #self.new_roi_fmap_obj = load_vgg(pretrained=False).classifier
        """

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))  # vgg.classifier
    def new_obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.new_roi_fmap_obj(feature_pool.view(rois.size(0), -1))  # vgg.classifier

    def get_neg_examples(self, rel_labels):
        """
        Given relationship combination (positive examples), return the negative examples.
        :param rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
        :return: neg_rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
        """
        neg_rel_labels = []

        num_im = rel_labels.data[:,0].max()+1
        im_inds = rel_labels.data.cpu().numpy()[:,0]
        rel_type = rel_labels.data.cpu().numpy()[:,3]
        box_pairs = rel_labels.data.cpu().numpy()[:,:3]

        for im_ind in range(num_im):

            pred_ind = np.where(im_inds == im_ind)[0]
         
            rel_type_i = rel_type[pred_ind]

            rel_labels_i = box_pairs[pred_ind][:,None,:]
            row_num = rel_labels_i.shape[0]
            rel_labels_i = torch.LongTensor(rel_labels_i).expand_as(torch.Tensor(row_num, self.neg_num, 3))
            neg_pairs_i = rel_labels_i.contiguous().view(-1, 3).cpu().numpy()

            neg_rel_type_i = np.zeros(self.neg_num)

            for k in range(rel_type_i.shape[0]):

                neg_rel_type_k = np.delete(rel_type_i, np.where(rel_type_i == rel_type_i[k])[0]) # delete same rel class
                #assert neg_rel_type_k.shape[0] != 0
                if neg_rel_type_k.shape[0] != 0: 
                    neg_rel_type_k = np.random.choice(neg_rel_type_k, size=self.neg_num, replace=True)
                    neg_rel_type_i = np.concatenate((neg_rel_type_i,neg_rel_type_k),axis=0)
                else:
                    orig_cls = np.arange(self.num_rels)
                    cls_pool = np.delete(orig_cls, np.where( orig_cls == rel_type_i[k] )[0])
                    neg_rel_type_k = np.random.choice(cls_pool, size=self.neg_num, replace=False)
                    neg_rel_type_i = np.concatenate((neg_rel_type_i,neg_rel_type_k),axis=0) 

            neg_rel_type_i = np.delete(neg_rel_type_i, np.arange(self.neg_num))  # delete the first few rows
            assert neg_pairs_i.shape[0] == neg_rel_type_i.shape[0]
            neg_rel_labels.append(np.column_stack((neg_pairs_i,neg_rel_type_i)))

        neg_rel_labels = torch.LongTensor(np.concatenate(np.array(neg_rel_labels), 0))
        neg_rel_labels = neg_rel_labels.cuda(rel_labels.get_device(), async=True)

        return neg_rel_labels

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """

        # Detector
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")
        im_inds = result.im_inds - image_offset
        # boxes: [#boxes, 4], without box deltas; where narrow error comes from, should .detach()
        boxes = result.rm_box_priors.detach()   

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet' # sgcls's result.rel_labels is gt and not None
            # rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)
            rel_labels_neg = self.get_neg_examples(result.rel_labels)
            rel_inds_neg = rel_labels_neg[:,:3]

        #torch.cat((result.rel_labels[:,0].contiguous().view(236,1),result.rm_obj_labels[result.rel_labels[:,1]].view(236,1),result.rm_obj_labels[result.rel_labels[:,2]].view(236,1),result.rel_labels[:,3].contiguous().view(236,1)),-1)
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)  #[275,3], [im_inds, box1_inds, box2_inds]

        # rois: [#boxes, 5]
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        # result.rm_obj_fmap: [384, 4096]
        #result.rm_obj_fmap = self.obj_feature_map(result.fmap.detach(), rois) # detach: prevent backforward flowing
        result.rm_obj_fmap = self.obj_feature_map(result.fmap.detach(), rois.detach()) # detach: prevent backforward flowing

        # BiLSTM
        result.rm_obj_dists, result.rm_obj_preds, edge_ctx = self.context(
            result.rm_obj_fmap,   # has been detached above
            # rm_obj_dists: [#boxes, 151]; Prevent gradients from flowing back into score_fc from elsewhere
            result.rm_obj_dists.detach(),  # .detach:Returns a new Variable, detached from the current graph
            im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes.data, result.boxes_all.detach() if self.mode == 'sgdet' else result.boxes_all)
        

        # Post Processing
        # nl_egde <= 0
        if edge_ctx is None:
            edge_rep = self.post_emb(result.rm_obj_preds)
        # nl_edge > 0
        else: 
            edge_rep = self.post_lstm(edge_ctx)  # [384, 4096*2]
     
        # Split into subject and object representations
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)  #[384,2,4096]
        subj_rep = edge_rep[:, 0]  # [384,4096]
        obj_rep = edge_rep[:, 1]  # [384,4096]
        prod_rep = subj_rep[rel_inds[:, 1]] * obj_rep[rel_inds[:, 2]]  # prod_rep, rel_inds: [275,4096], [275,3]
    

        if self.use_vision: # True when sgdet
            # union rois: fmap.detach--RoIAlignFunction--roifmap--vr [275,4096]
            vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])

            if self.limit_vision:  # False when sgdet
                # exact value TBD
                prod_rep = torch.cat((prod_rep[:,:2048] * vr[:,:2048], prod_rep[:,2048:]), 1) 
            else:
                prod_rep = prod_rep * vr  # [275,4096]
                if self.training:
                    vr_neg = self.visual_rep(result.fmap.detach(), rois, rel_inds_neg[:, 1:])
                    prod_rep_neg = subj_rep[rel_inds_neg[:, 1]].detach() * obj_rep[rel_inds_neg[:, 2]].detach() * vr_neg 
                    rel_dists_neg = self.rel_compress(prod_rep_neg)
                    

        if self.use_tanh:  # False when sgdet
            prod_rep = F.tanh(prod_rep)

        result.rel_dists = self.rel_compress(prod_rep)  # [275,51]

        if self.use_bias:  # True when sgdet
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.rm_obj_preds[rel_inds[:, 1]],
                result.rm_obj_preds[rel_inds[:, 2]],
            ), 1))


        if self.training:
            judge = result.rel_labels.data[:,3] != 0
            if judge.sum() != 0:  # gt_rel exit in rel_inds
                select_rel_inds = torch.arange(rel_inds.size(0)).view(-1,1).long().cuda()[result.rel_labels.data[:,3] != 0]
                com_rel_inds = rel_inds[select_rel_inds]
                twod_inds = arange(result.rm_obj_preds.data) * self.num_classes + result.rm_obj_preds.data
                result.obj_scores = F.softmax(result.rm_obj_dists.detach(), dim=1).view(-1)[twod_inds]   # only 1/4 of 384 obj_dists will be updated; because only 1/4 objs's labels are not 0

                # positive overall score
                obj_scores0 = result.obj_scores[com_rel_inds[:,1]]
                obj_scores1 = result.obj_scores[com_rel_inds[:,2]]
                rel_rep = F.softmax(result.rel_dists[select_rel_inds], dim=1)    # result.rel_dists has grad
                _, pred_classes_argmax = rel_rep.data[:,:].max(1)  # all classes
                max_rel_score = rel_rep.gather(1, Variable(pred_classes_argmax.view(-1,1))).squeeze()  # SqueezeBackward, GatherBackward
                score_list = torch.cat((com_rel_inds[:,0].float().contiguous().view(-1,1), obj_scores0.data.view(-1,1), obj_scores1.data.view(-1,1), max_rel_score.data.view(-1,1)), 1)
                prob_score = max_rel_score * obj_scores0.detach() * obj_scores1.detach()
                #pos_prob[:,1][result.rel_labels.data[:,3] == 0] = 0  # treat most rel_labels as neg because their rel cls is 0 "unknown"  
                
                # negative overall score
                obj_scores0_neg = result.obj_scores[rel_inds_neg[:,1]]
                obj_scores1_neg = result.obj_scores[rel_inds_neg[:,2]]
                rel_rep_neg = F.softmax(rel_dists_neg, dim=1)   # rel_dists_neg has grad
                _, pred_classes_argmax_neg = rel_rep_neg.data[:,:].max(1)  # all classes
                max_rel_score_neg = rel_rep_neg.gather(1, Variable(pred_classes_argmax_neg.view(-1,1))).squeeze() # SqueezeBackward, GatherBackward
                score_list_neg = torch.cat((rel_inds_neg[:,0].float().contiguous().view(-1,1), obj_scores0_neg.data.view(-1,1), obj_scores1_neg.data.view(-1,1), max_rel_score_neg.data.view(-1,1)), 1)
                prob_score_neg = max_rel_score_neg * obj_scores0_neg.detach() * obj_scores1_neg.detach()

                # use all rel_inds, already irrelavant with im_inds, which is only use to extract region from img and produce rel_inds
                # 384 boxes---(rel_inds)(rel_inds_neg)--->prob_score,prob_score_neg 
                all_rel_inds = torch.cat((result.rel_labels.data[select_rel_inds], rel_labels_neg), 0)  # [#pos_inds+#neg_inds, 4]
                flag = torch.cat((torch.ones(prob_score.size(0),1).cuda(),torch.zeros(prob_score_neg.size(0),1).cuda()),0)
                score_list_all = torch.cat((score_list,score_list_neg), 0) 
                all_prob = torch.cat((prob_score,prob_score_neg), 0)  # Variable, [#pos_inds+#neg_inds, 1]

                _, sort_prob_inds = torch.sort(all_prob.data, dim=0, descending=True)

                sorted_rel_inds = all_rel_inds[sort_prob_inds]
                sorted_flag = flag[sort_prob_inds].squeeze()  # can be used to check distribution of pos and neg
                sorted_score_list_all = score_list_all[sort_prob_inds]
                sorted_all_prob = all_prob[sort_prob_inds]  # Variable
                
                # positive triplet and score list
                pos_sorted_inds = sorted_rel_inds.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 1).view(-1,4)
                pos_trips = torch.cat((pos_sorted_inds[:,0].contiguous().view(-1,1), result.rm_obj_labels.data.view(-1,1)[pos_sorted_inds[:,1]], result.rm_obj_labels.data.view(-1,1)[pos_sorted_inds[:,2]], pos_sorted_inds[:,3].contiguous().view(-1,1)), 1)
                pos_score_list = sorted_score_list_all.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 1).view(-1,4)
                pos_exp = sorted_all_prob[sorted_flag == 1]  # Variable 

                # negative triplet and score list
                neg_sorted_inds = sorted_rel_inds.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 0).view(-1,4)
                neg_trips = torch.cat((neg_sorted_inds[:,0].contiguous().view(-1,1), result.rm_obj_labels.data.view(-1,1)[neg_sorted_inds[:,1]], result.rm_obj_labels.data.view(-1,1)[neg_sorted_inds[:,2]], neg_sorted_inds[:,3].contiguous().view(-1,1)), 1)
                neg_score_list = sorted_score_list_all.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 0).view(-1,4)
                neg_exp = sorted_all_prob[sorted_flag == 0]  # Variable
                
                
                int_part = neg_exp.size(0) // pos_exp.size(0)
                decimal_part = neg_exp.size(0) % pos_exp.size(0)
                int_inds = torch.arange(pos_exp.size(0))[:,None].expand_as(torch.Tensor(pos_exp.size(0), int_part)).contiguous().view(-1)
                int_part_inds = (int(pos_exp.size(0) -1) - int_inds).long().cuda() # use minimum pos to correspond maximum negative
                if decimal_part == 0:
                    expand_inds = int_part_inds
                else:
                    expand_inds = torch.cat((torch.arange(pos_exp.size(0))[(pos_exp.size(0) - decimal_part):].long().cuda(), int_part_inds), 0)  
                
                result.pos = pos_exp[expand_inds]
                result.neg = neg_exp
                result.anchor = Variable(torch.zeros(result.pos.size(0)).cuda())
                # some variables .register_hook(extract_grad)

                return result

            else:  # no gt_rel in rel_inds
                print("no gt_rel in rel_inds!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                twod_inds = arange(result.rm_obj_preds.data) * self.num_classes + result.rm_obj_preds.data
                result.obj_scores = F.softmax(result.rm_obj_dists.detach(), dim=1).view(-1)[twod_inds]

                # positive overall score
                obj_scores0 = result.obj_scores[rel_inds[:,1]]
                obj_scores1 = result.obj_scores[rel_inds[:,2]]
                rel_rep = F.softmax(result.rel_dists, dim=1)    # [275, 51]
                _, pred_classes_argmax = rel_rep.data[:,:].max(1)  # all classes
                max_rel_score = rel_rep.gather(1, Variable(pred_classes_argmax.view(-1,1))).squeeze() # SqueezeBackward, GatherBackward
                prob_score = max_rel_score * obj_scores0.detach() * obj_scores1.detach()
                #pos_prob[:,1][result.rel_labels.data[:,3] == 0] = 0  # treat most rel_labels as neg because their rel cls is 0 "unknown"  
                
                # negative overall score
                obj_scores0_neg = result.obj_scores[rel_inds_neg[:,1]]
                obj_scores1_neg = result.obj_scores[rel_inds_neg[:,2]]
                rel_rep_neg = F.softmax(rel_dists_neg, dim=1)   
                _, pred_classes_argmax_neg = rel_rep_neg.data[:,:].max(1)  # all classes
                max_rel_score_neg = rel_rep_neg.gather(1, Variable(pred_classes_argmax_neg.view(-1,1))).squeeze() # SqueezeBackward, GatherBackward
                prob_score_neg = max_rel_score_neg * obj_scores0_neg.detach() * obj_scores1_neg.detach()

                # use all rel_inds, already irrelavant with im_inds, which is only use to extract region from img and produce rel_inds
                # 384 boxes---(rel_inds)(rel_inds_neg)--->prob_score,prob_score_neg 
                all_rel_inds = torch.cat((result.rel_labels.data, rel_labels_neg), 0)  # [#pos_inds+#neg_inds, 4]
                flag = torch.cat((torch.ones(prob_score.size(0),1).cuda(),torch.zeros(prob_score_neg.size(0),1).cuda()),0)
                all_prob = torch.cat((prob_score,prob_score_neg), 0)  # Variable, [#pos_inds+#neg_inds, 1]

                _, sort_prob_inds = torch.sort(all_prob.data, dim=0, descending=True)

                sorted_rel_inds = all_rel_inds[sort_prob_inds]
                sorted_flag = flag[sort_prob_inds].squeeze()  # can be used to check distribution of pos and neg
                sorted_all_prob = all_prob[sort_prob_inds]  # Variable

                pos_sorted_inds = sorted_rel_inds.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 1).view(-1,4)
                neg_sorted_inds = sorted_rel_inds.masked_select(sorted_flag.view(-1,1).expand(-1,4).cuda() == 0).view(-1,4)
                pos_exp = sorted_all_prob[sorted_flag == 1]  # Variable  
                neg_exp = sorted_all_prob[sorted_flag == 0]  # Variable

                int_part = neg_exp.size(0) // pos_exp.size(0)
                decimal_part = neg_exp.size(0) % pos_exp.size(0)
                int_inds = torch.arange(pos_exp.data.size(0))[:,None].expand_as(torch.Tensor(pos_exp.data.size(0), int_part)).contiguous().view(-1)
                int_part_inds = (int(pos_exp.data.size(0) -1) - int_inds).long().cuda() # use minimum pos to correspond maximum negative
                if decimal_part == 0:
                    expand_inds = int_part_inds
                else:
                    expand_inds = torch.cat((torch.arange(pos_exp.size(0))[(pos_exp.size(0) - decimal_part):].long().cuda(), int_part_inds), 0)  
                
                result.pos = pos_exp[expand_inds]
                result.neg = neg_exp
                result.anchor = Variable(torch.zeros(result.pos.size(0)).cuda())

                return result
        ###################### Testing ###########################

        # extract corrsponding scores according to the box's preds
        twod_inds = arange(result.rm_obj_preds.data) * self.num_classes + result.rm_obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]   # [384]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)    # [275, 51]
        
        # sort product of obj1 * obj2 * rel
        return filter_dets(bboxes, result.obj_scores,
                           result.rm_obj_preds, rel_inds[:, 1:],
                           rel_rep)


    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs

"""# write data into txt
# write data into txt
dirname = '/home/yiwuzhong/motifs/'
pathname = os.path.join(dirname)

with open(os.path.join(pathname, 'triplet.txt'), 'a') as g:
    p = pos_trips.cpu().numpy()
    pp = pos_exp.data.cpu().numpy()
    ppp = pos_score_list.cpu().numpy()
    pppp = pos_sorted_inds.cpu().numpy()
    pn = neg_trips.cpu().numpy()
    ppn = neg_exp.data.cpu().numpy()
    pppn = neg_score_list.cpu().numpy()
    ppppn = neg_sorted_inds.cpu().numpy()
    g.write('\n \n Positive triplets:')
    for i in range(pos_trips.size(0)):
        g.write("\n pos_box_inds: ")
        g.write(np.array2string(pppp[i, 1:],  formatter={'float_kind':lambda x: "%.4f" % x}))
        g.write("--")
        obj1_str = self.classes[p[i, 1]]
        obj2_str = self.classes[p[i, 2]]
        rel_str = self.rel_classes[p[i, 3]]
        g.write(' {} - {} - {}'.format(obj1_str, obj2_str, rel_str))
    for i in range(pos_trips.size(0)):
        g.write("\n pos_score_list:")
        g.write(np.array2string(ppp[i, 1:],  formatter={'float_kind':lambda x: "%.4f" % x}))
        g.write("--")
        g.write(np.array2string(pp[i],  formatter={'float_kind':lambda x: "%.4f" % x}))
    g.write("\n \nPositive examples (" + str(pos_trips.size(0)) +  ") distribution among all examples (" + str(sorted_flag.size(0)) + ")")
    g.write("\n")
    g.write(np.array2string(torch.arange(sorted_flag.size(0)).cuda()[sorted_flag == 1].cpu().numpy(),  formatter={'float_kind':lambda x: "%.0f" % x}))
    g.write('\n \n Negative triplets:')
    for i in range(20):
        g.write("\n neg_box_inds: ")
        g.write(np.array2string(ppppn[i, 1:],  formatter={'float_kind':lambda x: "%.4f" % x}))
        obj1_str = self.classes[pn[i, 1]]
        obj2_str = self.classes[pn[i, 2]]
        rel_str = self.rel_classes[pn[i, 3]]
        g.write(' {} - {} - {}'.format(obj1_str, obj2_str, rel_str))
    for i in range(20):
        g.write("\n neg_score_list:")
        g.write(np.array2string(pppn[i, 1:],  formatter={'float_kind':lambda x: "%.4f" % x}))
        g.write("--")
        g.write(np.array2string(ppn[i],  formatter={'float_kind':lambda x: "%.4f" % x}))

"""