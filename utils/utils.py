import torch
import numpy as np

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def get_class_weights(dset_name='sun'):
    with open('/home/vinod/Workspace/MultitaskNet/datasets/sunrgbd/class_weights', 'r') as f:
        lines = f.read().splitlines()
    dset_weights, read = [], False

    for line in lines:
        if line.lower().find(dset_name) is not -1 and not read:
            read = True
            continue
        if line == '-':
            read = False
        if read:
            weights = line.split(', ')
            for weight in weights:
                dset_weights.append(float(weight))
    #print(dset_weights)
    return torch.cuda.FloatTensor(dset_weights)


def print_time_info(start_time, end_time):
    print('[INFO] Start and end time of the last session: %s - %s'
          % (start_time.strftime('%d.%m.%Y %H:%M:%S'), end_time.strftime('%d.%m.%Y %H:%M:%S')))
    print('[INFO] Total time previous session took:', (end_time - start_time), '\n')


def calculate_confusion_matrix(pred, label, num_seg_class, mask=None):
    if mask is None:
        mask = np.ones_like(label) == 1
    k = (label >= 0) & (pred < num_seg_class) & (mask.astype(np.bool))
    return np.bincount(num_seg_class * label[k].astype(int) + pred[k], minlength=num_seg_class**2).reshape(num_seg_class, num_seg_class)


def get_scores(conf_mat):
    if conf_mat.sum() == 0:
        return 0, 0, 0
    with np.errstate(divide='ignore', invalid='ignore'):
        global_acc = np.diag(conf_mat).sum() / np.float(conf_mat.sum())
        mean_acc = np.diag(conf_mat) / conf_mat.sum(1).astype(np.float)
        iou = np.diag(conf_mat) / (conf_mat.sum(1) + conf_mat.sum(0) - np.diag(conf_mat)).astype(np.float)
    return global_acc, np.nanmean(mean_acc), np.nanmean(iou)

def compute_errors(gt, pred):
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]
    
    pred[pred < MIN_DEPTH] = MIN_DEPTH
    pred[pred > MAX_DEPTH] = MAX_DEPTH
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10
