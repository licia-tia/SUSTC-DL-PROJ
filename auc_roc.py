import os

import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import torch.nn as nn

from models import EfficientNetB0
from process_data import SIIM_ISIC, get_data_transforms

import matplotlib.pyplot as plt
#device = 'cuda:2'

if __name__ == '__main__':

    net = EfficientNetB0()

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/acc80.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']

    print("best acc: ", best_acc)

    transform_train, transform_test = get_data_transforms()
    testset = SIIM_ISIC(train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    positive_scores_total = []
    targets_total = []

    with torch.no_grad():
        for batch_idx, (inputs, meta,  targets) in enumerate(testloader):
            #inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores = nn.functional.softmax(outputs, dim=1)

            positive_scores = scores[:, 1]
            targets_total.append(targets.to('cpu'))
            positive_scores_total.append(positive_scores.to('cpu'))


    targets_total = torch.cat(targets_total)
    positive_scores_total = torch.cat(positive_scores_total)
    # auc_score = roc_auc_score(targets_total, positive_scores_total)
    # print("auc: ", auc_score)


    y, x, _ = roc_curve(targets_total, positive_scores_total)
    auc_score2 = auc(y, x)
    print("auc: ", auc_score2)
    plt.figure()
    lw = 2
    plt.plot(y, x, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_score2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()