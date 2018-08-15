# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time
from torch.nn import PairwiseDistance as pdist
from lib.pytorch_misc import arange
from torch.autograd import Variable
import ipdb


def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores, obj1, obj2, rel_emb):
    """
    Filters detectionsa according to the score product: obj1 * obj2 * rel
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0) # 64
    assert obj_scores.size(0) == num_box  # 64

    assert obj_classes.size() == obj_scores.size()  # 64
    num_rel = rel_inds.size(0)  # 275
    assert rel_inds.size(1) == 2
    #assert pred_scores.size(0) == num_rel

    #obj_scores0 = obj_scores.data[rel_inds[:,0]]
    #obj_scores1 = obj_scores.data[rel_inds[:,1]]

    #pred_scores_max, pred_classes_argmax = pred_scores.data[:,1:].max(1)
    #pred_classes_argmax = pred_classes_argmax + 1
    # get maximum score among 150/50 classes, for single obj and rel seperately, then product and sort

    num_trip = rel_inds.size(0)
    num_classes = obj1.size(1)
    num_rels = rel_emb.size(1)
    embdim = rel_emb.size(2)

    two_inds1 = arange(obj_classes.data[rel_inds[:,0]]) * num_classes + obj_classes.data[rel_inds[:,0]]
    two_inds2 = arange(obj_classes.data[rel_inds[:,1]]) * num_classes + obj_classes.data[rel_inds[:,1]]
    obj1emb = obj1.view(-1, embdim)[two_inds1]  # (275,151,10) -> (275, 10)
    obj2emb = obj2.view(-1, embdim)[two_inds2]  # (275,151,10) -> (275, 10)

    d = obj1emb - obj2emb   # (275, 10)
    d = d[:, None, :].expand_as(rel_emb)   # (275, 51, 10), copy 51 times
    d = d + rel_emb  # (275, 51, 10)
    d = d.view(-1, embdim)  # (275*51, 10)
    d = torch.squeeze(torch.sqrt(d.pow(2).sum(1)))  # (275*51, 1) -> (275*51)

    rel_surgery = []

    for i in range(num_trip):
        start = num_rels * i
        end = num_rels * (i + 1)
        min_d, min_ind = d[start+1:end].min(0)  # ignore "unknown" ind = 0
        rel_surgery.append( [min_ind.data + 1, min_d.data] )  # restore the relationship class value

    rel_surgery = np.array(rel_surgery)
    rel_pred = torch.from_numpy(rel_surgery[:,0])  # double tensor, float element
    distance = torch.from_numpy(rel_surgery[:,1])  # double tensor, float element
   
    #rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    #rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)
    # rel_scores_vs: sorted distance, double tensor; rel_scores_idx: long tensor
    rel_scores_vs, rel_scores_idx = torch.sort(distance.contiguous().view(-1), dim=0, descending=False)

    # boxes_out: rois incorporated deltas
    # objs_np: rm_obj_preds from decoder rnn
    # obj_scores_np: rm_obj_dists from decoder rnn
    # rels: rel_inds from boxes overlapped; after surgery, sorted by overall_score / distance
    # pred_scores_sorted: extracted by max() among 50 cls, then rel scores sorted by overall scores
    rels = rel_inds[rel_scores_idx.cuda()].cpu().numpy()  # rel_inds is "cuda" long tensor
    sorted_rel_pred = rel_pred[rel_scores_idx].cpu().numpy().astype(int)  # if it's float, when column stack it with rels, the entity will be float
    #pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    pred_scores_sorted = None
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()

    return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted, sorted_rel_pred

# def _get_similar_boxes(boxes, obj_classes_topk, nms_thresh=0.3):
#     """
#     Assuming bg is NOT A LABEL.
#     :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
#     :param obj_classes: [num_box, topk] class labels
#     :return: num_box, topk, num_box, topk array containing similarities.
#     """
#     topk = obj_classes_topk.size(1)
#     num_box = boxes.size(0)
#
#     box_flat = boxes.view(-1, 4) if boxes.dim() == 3 else boxes[:, None].expand(
#         num_box, topk, 4).contiguous().view(-1, 4)
#     jax = bbox_overlaps(box_flat, box_flat).data > nms_thresh
#     # Filter out things that are not gonna compete.
#     classes_eq = obj_classes_topk.data.view(-1)[:, None] == obj_classes_topk.data.view(-1)[None, :]
#     jax &= classes_eq
#     boxes_are_similar = jax.view(num_box, topk, num_box, topk)
#     return boxes_are_similar.cpu().numpy().astype(np.bool)
