import torch
import torch.nn.functional as F
from utils.utils import get_class_weights

# depth loss
def custom_loss_function(output, target):
    # di = output - target
    di = target - output
    output_height = 224  # For NYU , change to 224 if sunrgdb
    output_width = 224   # For NYU , change to 224 if sunrgdb
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

def cross_entropy_2d():
    def wrap(seg_preds, seg_targets, depth_pred, depth_target, class_inputs=None, class_targets=None,
             lambda_1=0.5, lambda_2=None, weight=None, pixel_average=True):

        # If the dataset is SUN RGB-D use class normalization weights in order to introduce balance to calculated loss as the number
        # of classes in SUN RGB-D dataset are not uniformly distributed.
        n, c, h, w = seg_preds.size()
        if c == 38:
            weight = get_class_weights('sun')

        # Calculate segmentation loss
        seg_inputs = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_inputs = seg_inputs[seg_targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        # Exclude the 0-valued pixels from the loss calculation as 0 values represent the pixels that are not annotated.
        seg_targets_mask = seg_targets > 0
        # Subtract 1 from all classes, in the ground truth tensor, in order to match the network predictions.
        # Remember, in network predictions, label 0 corresponds to label 1 in ground truth.
        seg_targets = seg_targets[seg_targets_mask] - 1

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.cross_entropy(seg_inputs, seg_targets, weight=weight, size_average=False)

        # Average the calculated loss value over each labeled pixel in the ground-truth tensor
        if pixel_average:
            seg_loss /= seg_targets_mask.float().data.sum()

        #print(lambda_1)
        #d_loss = depth_loss(depth_pred, depth_target)
        d_loss = custom_loss_function(depth_pred, depth_target)
        loss = (lambda_1 * seg_loss) + ((1.0 - lambda_1) * d_loss)

        # If scene classification function is utilized, calculate class loss, multiply with coefficient, lambda_2, sum with total loss
        if lambda_2 is not None:
            # Calculate classification loss
            class_targets -= 1
            class_loss = F.cross_entropy(class_inputs, class_targets)
            # Combine losses
            loss += lambda_2 * class_loss

            seg_loss = seg_loss.item()
            class_loss = class_loss.item()
            return loss, seg_loss, class_loss, d_loss
        return loss, seg_loss.item(), d_loss
    return wrap
