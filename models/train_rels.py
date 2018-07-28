"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os

from config import ModelConfig, BOX_SCALE, IM_SCALE, FG_FRACTION, RPN_FG_FRACTION
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
from init_logging import init_logging
import logging
import ipdb



conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

init_logging("/home/yiwuzhong/motifs/logging/" + conf.mode +".log")

train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

# ipdb.set_trace()

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )
"""
# Freeze the detector
# .named_parameters(): returns (string, Parameter), Tuple containing the name and parameter itself
for n, param in detector.detector.named_parameters():
    #print(n)
    if n.startswith('score'):
        param.requires_grad = False
    elif n.startswith('bbox'):
        param.requires_grad = False
    if n.startswith('roi'):
        param.requires_grad = False
    elif n.startswith('features'):
        param.requires_grad = False
    else:
        continue
"""

print(print_para(detector), flush=True)

# optimizer
def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps stabilize the models.
    # p.requires_grad == True if it's not Faster RCNN param; == False if it's Faster RCNN param
    # original: add all 'roi_fmap' in Relmodel into fc_params; add all not 'roi_fmap' in Relmodel into non_fc_params; params in faster rcnn, continue
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

#checkpoints downloaded
ckpt = torch.load(conf.ckpt)

# sgdet training
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']
    #print("vgrel ckpt:", ckpt['state_dict'].keys())

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])

# sgcls training
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])
    #print("vgdet ckpt:", ckpt['state_dict'].keys())

    detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

detector.cuda()


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            logging.info("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            logging.info(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    #ipdb.set_trace()
    result = detector[b]
    losses = {}
    if conf.mode == 'sgdet':
        ############################  Detector Loss  #################################

        # final classification
        labels = result.od_obj_labels  # [4000+]
        scores = result.od_obj_dists  # [4000+, 151]
        od_class_loss = F.cross_entropy(scores, labels)

        # final box location
        bbox_targets = result.od_box_targets  # [4000+, 4], gt box
        box_deltas = result.od_box_deltas  # [4000+, 151, 4], delta
        roi_boxes = result.od_box_priors  # [4000, 4], prior box

        # detector loss
        valid_inds = (labels.data != 0).nonzero().squeeze(1)
        fg_cnt = valid_inds.size(0)
        bg_cnt = labels.size(0) - fg_cnt

        # No gather_nd in pytorch so instead convert first 2 dims of tensor to 1d
        box_reg_mult = 2 * (1. / FG_FRACTION) * fg_cnt / (fg_cnt + bg_cnt + 1e-4)
        twod_inds = valid_inds * box_deltas.size(1) + labels[valid_inds].data

        od_box_loss = bbox_loss(roi_boxes[valid_inds], box_deltas.view(-1, 4)[twod_inds],
                             bbox_targets[valid_inds]) * box_reg_mult
        
        # RPN
        rpn_scores = result.rpn_scores  # [1536, 2], yes/no
        rpn_box_deltas = result.rpn_box_deltas  # [1536, 4]

        train_anchor_labels = b.train_anchor_labels[:, -1]
        train_anchors = b.train_anchors[:, :4]
        train_anchor_targets = b.train_anchors[:, 4:]

        train_valid_inds = (train_anchor_labels.data == 1).nonzero().squeeze(1)
        rpn_class_loss = F.cross_entropy(rpn_scores, train_anchor_labels)

        rpn_box_mult = 2 * (1. / RPN_FG_FRACTION) * train_valid_inds.size(0) / (train_anchor_labels.size(0) + 1e-4)
        rpn_box_loss = bbox_loss(train_anchors[train_valid_inds],
                                 rpn_box_deltas[train_valid_inds],
                                 train_anchor_targets[train_valid_inds]) * rpn_box_mult

        losses['rpn_class_loss'] = rpn_class_loss
        losses['rpn_box_loss'] = rpn_box_loss
        losses['od_class_loss'] = od_class_loss
        losses['od_box_loss'] = od_box_loss

        ############################  Detector Loss  #################################

    # cross_entropy(input, target): 
    # input, (#obj, 151), vector of #classes dim, which will be converted into probability (scores) by log_softmax
    # target, (#obj), corresponding obj labels belong to [1,150], which will be converted into one-hot vector
    # rm_obj_dists.shape:[164, 151]
    # rm_obj_labels.shape:[164]
    # result.rel_labels.shape:[1810, 4], [img_ind, box0_ind, box1_ind, rel_type]
    # result.rel_dists.shape:[1810, 51]

    losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None], # p.grad is None when param don't backward propagate
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


logging.info("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    #print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    logging.info("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)))

    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    mAp = val_epoch()
    scheduler.step(mAp)
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
        #print("exiting training early", flush=True)
        logging.info("exiting training early")
        break
