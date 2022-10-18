"""
collection of different metrices for segmentation models
"""
import torch


def iou(true_masks: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
    """Computes the iou for binary segmentations masks.

    :param true_masks: (b,1,h,w)
    :type true_masks: torch.Tensor
    :param pred_masks: (b,1,h,w)
    :type pred_masks: torch.Tensor
    :return: Returns a tensor with dimension (b,fg,h,w) dimension. The second channel
    stands for foreground/background. If fg=1, it stands for forground, if fg=2 it stands for background
    :rtype: torch.Tensor
    """

    sum_masks = true_masks + pred_masks

    area_of_overlap_background = torch.sum(sum_masks == 0, (-2, -1))
    area_of_union_background = torch.sum(sum_masks < 2, (-2, -1))

    area_of_overlap_forground = torch.sum(sum_masks == 2, (-2, -1))
    area_of_union_forground = torch.sum(sum_masks > 0, (-2, -1))

    iou_background = area_of_overlap_background / area_of_union_background
    iou_forground = area_of_overlap_forground / area_of_union_forground

    iou = torch.concat((iou_forground, iou_background), dim=0)

    iou.cpu()

    return iou


def test_iou():
    t = torch.Tensor([[[[0, 1], [0, 1]]]])
    p = torch.Tensor([[[[0, 1], [0, 1]]]])

    ious = iou(t, p)

    assert torch.equal(ious, torch.Tensor([[1.0], [1.0]]))


def pixel_accuracy(true_masks: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
    """Computes the pixel accuracy for binary segmentations masks .

    :param true_masks: _description_
    :type true_masks: torch.Tensor
    :param pred_masks: _description_
    :type pred_masks: torch.Tensor
    :return: _description_
    :rtype: torch.Tensor
    """
    total_pixel = true_masks.shape[-2] * true_masks.shape[-1]
    equal_pixels = torch.sum(true_masks == pred_masks)
    accuracy = equal_pixels / total_pixel

    return accuracy
