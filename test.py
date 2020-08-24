import argparse
import os
import torch
from models import *
from process_data import get_data_transforms, SIIM_ISIC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS324 Final')
    parser.add_argument('--data_root', default='/home/group3/DataSet', type=str, help='data root path')
    parser.add_argument('--csv', default='validation_set.csv', type=str, help='csv file name')
    parser.add_argument('--img_folder', default='Validation_set', type=str, help='image folder name')
    args = parser.parse_args()
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ensemble_model = []

    # effnet
    net = EfficientNetB0()
    checkpoint = torch.load('./checkpoint/effnet-b0.pth')
    net.load_state_dict(checkpoint['net'])
    ensemble_model.append(net)
    print('Load effnet-b0 with acc =', checkpoint['acc'])

    # densenet
    net = DenseNet()
    checkpoint = torch.load('./checkpoint/effnet-b0.pth')
    net.load_state_dict(checkpoint['net'])
    ensemble_model.append(net)
    print('Load effnet-b0 with acc =', checkpoint['acc'])


    files = os.listdir('./checkpoint')
    for file in files:
        net = torch.nn.Module()
        checkpoint = torch.load('./checkpoint/' + file)
        net.load_state_dict(checkpoint['net'])
        ensemble_model.append(net)
        print(file)
        print('Acc: ' + checkpoint['acc'])
        print()

    transform_train, transform_test = get_data_transforms(size=224)
    testset = SIIM_ISIC(type='validate', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )



