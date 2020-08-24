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
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ensemble_model = {}

    # effnet
    net = EfficientNetB0()
    checkpoint = torch.load('./checkpoint/effnet-b0.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    ensemble_model['eff1'] = net
    print('Load effnet-b0 with acc =', checkpoint['acc'])

    # effnet-2
    net = EfficientNetB0()
    checkpoint = torch.load('./checkpoint/effnet-b0-2.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    ensemble_model['eff2'] = net
    print('Load effnet-b0-2 with acc =', checkpoint['acc'])

    # cnn
    # net = Net(3, meta_dim=8)
    # checkpoint = torch.load('./checkpoint/cnn.pth', map_location='cpu')
    # net.load_state_dict(checkpoint['net'])
    # ensemble_model['cnn'] = net
    # print('Load cnn with acc =', checkpoint['acc'])

    # densenet
    # net = DenseNet201()
    # checkpoint = torch.load('./checkpoint/denseNet.pth')
    # net.load_state_dict(checkpoint['net'])
    # ensemble_model['dense'] = net
    # print('Load denseNet with acc =', checkpoint['acc'])

    transform_train, transform_test = get_data_transforms(size=224)
    testset = SIIM_ISIC(type='validate', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )

    for net in ensemble_model:
        name = net
        net = ensemble_model[net]
        net.eval()
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
                 lw=lw, label='ROC curve (AUC score = %0.2f)' % auc_score2)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for Melanoma Classification')
        plt.legend(loc="lower right")
        # plt.show()
        if not os.path.exists('./roc_auc'):
            os.mkdir('./roc_auc')
        plt.savefig('./roc_auc/' + net)

    # ensemble model
    positive_scores_total = []
    targets_total = []

    with torch.no_grad():
        for batch_idx, (inputs, meta, targets) in enumerate(testloader):
            # inputs, targets = inputs.to(device), targets.to(device)
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
             lw=lw, label='ROC curve (AUC score = %0.2f)' % auc_score2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for Melanoma Classification')
    plt.legend(loc="lower right")
    # plt.show()
    if not os.path.exists('./roc_auc'):
        os.mkdir('./roc_auc')
    plt.savefig('./roc_auc/' + net)