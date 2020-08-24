import argparse
import os
import torch
from torch import nn
from models import *
from process_data import get_data_transforms, SIIM_ISIC
from utils import progress_bar

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
# meta convertion
sex = {'female': 1, 'male': -1, 'unknown': 0}
anatom = {'palms/soles': [1, 0, 0, 0, 0, 0], 'lower extremity': [0, 1, 0, 0, 0, 0], 'upper extremity': [0, 0, 1, 0, 0, 0],
          'torso': [0, 0, 0, 4, 0, 0], 'oral/genital': [0, 0, 0, 0, 1, 0], 'head/neck': [0, 0, 0, 0, 0, 1],  'unknown': [0, 0, 0, 0, 0, 0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS324 Final')
    parser.add_argument('--data_root', default='/home/group3/DataSet', type=str, help='data root path')
    parser.add_argument('--csv', default='validation_set.csv', type=str, help='csv file name')
    parser.add_argument('--img_folder', default='Validation_set', type=str, help='image folder name')
    args = parser.parse_args()
    print('Data root:', args.data_root)
    print('CSV file:', args.csv)
    print('Image folder:', args.img_folder)

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
    testset = SIIM_ISIC(type='test', data_root=args.data_root, csv_file=args.csv,
                        img_folder=args.img_folder, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    device = 'cpu'
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, meta, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
            outputs = None
            for net in ensemble_model:
                if net == 'cnn':
                    net = ensemble_model[net]
                    net.eval()
                    for s in range(len(meta['sex'])):
                        meta['sex'][s] = sex[meta['sex'][s]]
                    meta['sex'] = torch.tensor(meta['sex']).unsqueeze(0)
                    meta['sex'] = torch.transpose(meta['sex'], 0, 1)
                    meta['age_approx'] = torch.tensor(meta['age_approx']).unsqueeze(0)
                    meta['age_approx'] = torch.transpose(meta['age_approx'], 0, 1)
                    for a in range(len(meta['anatom_site_general_challenge'])):
                        meta['anatom_site_general_challenge'][a] = anatom[
                            meta['anatom_site_general_challenge'][a]]
                    meta['anatom_site_general_challenge'] = torch.tensor(meta['anatom_site_general_challenge'])
                    # pdb.set_trace()
                    meta = torch.cat((meta['sex'], meta['age_approx'], meta['anatom_site_general_challenge']),
                                     1)
                    meta = meta.to(device)
                    if outputs is None:
                        outputs = net(inputs, meta)
                    else:
                        outputs += net(inputs, meta)
                else:
                    net = ensemble_model[net]
                    net.eval()
                    sm = nn.Softmax(dim=1)
                    if outputs is None:
                        outputs = sm(net(inputs))
                    else:
                        outputs += sm(net(inputs))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if predicted.eq(targets).sum().item() == 0:
                print('One misclassified sample: ' + str(batch_idx))

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
