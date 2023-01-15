# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:39:00 2022

@author: rajgudhe
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
from sklearn.metrics import auc, roc_curve

def accuracy(output, target):
    pred = output.max(1)[1]
    return 100.0 * target.eq(pred).float().mean()


def save_checkpoint(checkpoint_dir, state, epoch):
    file_path = os.path.join(checkpoint_dir, 'epoch_{}.pth.tar'.format(epoch))
    torch.save(state, file_path)
    return file_path

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def evaluate(cfg, test_loader, model, model_checkpoint_path, device):
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    targets = [[] for _ in range(cfg.arch.num_classes)]
    probs = [[] for _ in range(cfg.arch.num_classes)]

    with torch.no_grad():
        model.eval()
        for input, target in tqdm(test_loader):
            input, target = input.to(device), target.to(device)
            prob = model(input)

            for i in range(cfg.arch.num_classes):
                targets[i].extend(target.cpu().numpy() == i)
                probs[i].extend(prob[:, i].cpu().numpy())

    return targets, probs

def partial_auc(fpr, tpr, thresh):
    loc = len([t for t in tpr if t >= thresh])
    pAUC = np.trapz(tpr[-1*loc:], fpr[-1*loc:]) * (1-thresh)
    return pAUC

def plot_multi_roc_curve(ys_true, ys_score, legend, title):
    n_classes = len(ys_true)

    # First aggregate all false positive rates
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pauc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ys_true[i], ys_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pauc[i] = partial_auc(fpr[i], tpr[i], thresh=0.8)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 8))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect('equal')
    lw = 2

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(['green', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} ( AUC = {1:0.2f})'
                       ''.format(legend[i], roc_auc[i], pauc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.plot([0, 1], [0.8, 0.8], color='navy', linestyle=':',
             lw=lw, label='Partial AUC threshold = 0.8 TPR')

    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title)
    #plt.savefig(os.path.join(figsave_path, str(title) + '.png'))
    plt.legend(loc="lower right", fontsize=12)
    plt.show()
