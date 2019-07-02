from mxnet import gluon
from gluoncv.loss import _as_list
from mxnet import nd
import ipdb

class EastLoss(gluon.nn.Block):

  def __init__(self, cls_weight= 1.0 , iou_weight=100, angle_weight=2000, **kwargs):
    super(EastLoss, self).__init__(**kwargs)
    self.cls_weight = cls_weight
    self.iou_weight = iou_weight
    self.angle_weight = angle_weight

  def forward(self, score_gt, score_pred, geo_gt, geo_pred, training_masks, *args, **kwargs):
    # classification loss
    eps = 1e-5
    F = nd
    score_gt, score_pred, geo_gt, geo_pred, training_masks = [_as_list(x) for x in (score_gt, score_pred, geo_gt, geo_pred, training_masks)]

    cls_losses = []
    geo_losses = []
    sum_losses = []
    for sg, sp, gt, gp, tm in zip(*[score_gt, score_pred, geo_gt, geo_pred, training_masks ]):
      intersection = F.sum(sg * sp * tm)
      union = F.sum(tm * sg) + F.sum(tm * sp) + eps
      dice_loss = 1. - (2 * intersection / union)
      classification_loss = self.cls_weight * dice_loss
      cls_losses.append(classification_loss)

    # AABB loss
      top_gt, right_gt, bottom_gt, left_gt, angle_gt = F.split(gt, axis=1, num_outputs=5, squeeze_axis=1)
      top_pred, right_pred, bottom_pred, left_pred, angle_pred = F.split(gp, axis=1, num_outputs=5, squeeze_axis=1)


      area_gt = (top_gt + bottom_gt) * (left_gt + right_gt)
      area_pred = (top_pred + bottom_pred) * (left_pred + right_pred)
      w_union = F.minimum(left_gt, left_pred) + F.minimum(right_gt, right_pred)
      h_union = F.minimum(top_gt, top_pred) + F.minimum(bottom_gt, bottom_pred)

      area_inte = w_union * h_union
      area_union = area_gt + area_pred - area_inte
      L_AABB = -1.0 * F.log((area_inte + 1.0) / (area_union + 1.0))
      L_theta = 1.0 - F.cos(angle_gt - angle_pred)
      L_g =  self.iou_weight * L_AABB + self.angle_weight * L_theta
      geo_losses.append(F.mean(L_g * sg * tm))
      sum_losses.append(geo_losses[-1] + cls_losses[-1])
    return sum_losses, cls_losses, geo_losses
