
import numpy as np
import numpy.random as npr
from config import BG_THRESH_HI, BG_THRESH_LO, FG_FRACTION, ROIS_PER_IMG
from lib.fpn.box_utils import bbox_overlaps
from lib.pytorch_misc import to_variable
import torch

#############################################################
# The following is only for object detection (faster rcnn training)
@to_variable
def proposal_assignments_det(rpn_rois, gt_boxes, gt_classes, image_offset, fg_thresh=0.5):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
    """

    fg_rois_per_image = int(np.round(ROIS_PER_IMG * FG_FRACTION))
    # to some extent, gt_classes[88,0] - image_offset = rois[12000, 0]
    gt_img_inds = gt_classes[:, 0] - image_offset

    all_boxes = torch.cat([rpn_rois[:, 1:], gt_boxes], 0)  # [88+12000, 4]

    ims_per_box = torch.cat([rpn_rois[:, 0].long(), gt_img_inds], 0)  # [88+12000, 1]
    # sort according to the image index from 0 to 5 (6 images)
    im_sorted, idx = torch.sort(ims_per_box, 0)
    all_boxes = all_boxes[idx]

    # Assume that the GT boxes are already sorted in terms of image id
    num_images = int(im_sorted[-1]) + 1

    labels = []
    rois = []
    bbox_targets = []
    for im_ind in range(num_images): # traversal each single image
        # find the index(row, column) of nonzero
        g_inds = (gt_img_inds == im_ind).nonzero()

        if g_inds.dim() == 0:
            continue
        g_inds = g_inds.squeeze(1)
        g_start = g_inds[0]  # the start index of certain image in gt_img_inds
        g_end = g_inds[-1] + 1    # the end index of certain image in gt_img_inds

        t_inds = (im_sorted == im_ind).nonzero().squeeze(1)
        t_start = t_inds[0]  # the start index of certain image in im_sorted
        t_end = t_inds[-1] + 1    # the end index of certain image in im_sorted

        # Max overlaps: for each predicted box, get the max ROI
        # Get the indices into the GT boxes too (must offset by the box start)
        # compare rois+gtbox and gtbox; ious [t_inds.shape[0], g_inds.shape[0]]
        ious = bbox_overlaps(all_boxes[t_start:t_end], gt_boxes[g_start:g_end]) 
        max_overlaps, gt_assignment = ious.max(1)  # gt_assignment is a relative index
        max_overlaps = max_overlaps.cpu().numpy()
        # print("Best overlap is {}".format(max_overlaps.max()))
        # print("\ngt assignment is {} while g_start is {} \n ---".format(gt_assignment, g_start))
        gt_assignment += g_start  # the absolute index in gt_classes[:,0]; shape ex: [2011] means 2011 rois+gt boxes of certain image
        
        # keep_inds_np: foreground index + background index; [256,] ex: 39+217
        keep_inds_np, num_fg = _sel_inds(max_overlaps, fg_thresh, fg_rois_per_image,
                                         ROIS_PER_IMG)

        if keep_inds_np.size == 0:
            continue
        # covert numpy array to LongTensor
        keep_inds = torch.LongTensor(keep_inds_np).cuda(rpn_rois.get_device())

        labels_ = gt_classes[:, 1][gt_assignment[keep_inds]]  # [256] * 6 = 1536
        bbox_target_ = gt_boxes[gt_assignment[keep_inds]]  #[256] * 6 = 1536

        # Clamp labels_ for the background RoIs to 0
        if num_fg < labels_.size(0):
            labels_[num_fg:] = 0
        # rois: [256, 5]
        rois_ = torch.cat((
            im_sorted[t_start:t_end, None][keep_inds].float(),
            all_boxes[t_start:t_end][keep_inds],
        ), 1)

        labels.append(labels_)
        rois.append(rois_)
        bbox_targets.append(bbox_target_)

    rois = torch.cat(rois, 0)
    labels = torch.cat(labels, 0)
    bbox_targets = torch.cat(bbox_targets, 0)
    # rois, labels, bbox_targets are all Tensor
    return rois, labels, bbox_targets


def _sel_inds(max_overlaps, fg_thresh=0.5, fg_rois_per_image=128, rois_per_image=256):

    # Foreground:
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= fg_thresh)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.shape[0])
    # randomly select #min(128, #fg_inds) foreground regions 
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    #print("fg is ", fg_rois_per_this_image)

    #Backgrounds:
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >= BG_THRESH_LO))[0]

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    #print("bg is ", bg_rois_per_this_image, "\n")
    return np.append(fg_inds, bg_inds), fg_rois_per_this_image

