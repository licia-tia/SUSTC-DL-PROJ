import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS324 Final')
    parser.add_argument('--test', default='/home/group3/DataSet/validation_set.csv', type=str, help='testset path')
    args = parser.parse_args()
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    ensemble_model = []
    files = os.listdir('./checkpoint')
    for file in files:
        net = torch.nn.Module()
        net.load_state_dict(checkpoint['net'])
        ensemble_model.append(net)
        print(file)
        print('Acc: ' + checkpoint['acc'])
        print()

