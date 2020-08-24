import argparse
import os
import torch
from models import *
from process_data import get_data_transforms, SIIM_ISIC
from utils import progress_bar

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()


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
    ensemble_model = []

    # effnet
    net = EfficientNetB0()
    net = net.to(device)
    checkpoint = torch.load('./checkpoint/effnet-b0.pth')
    net.load_state_dict(checkpoint['net'])
    ensemble_model.append(net)
    print('Load effnet-b0 with acc =', checkpoint['acc'])

    # densenet
    # net = DenseNet201()
    # net = net.to(device)
    # checkpoint = torch.load('./checkpoint/denseNet.pth')
    # net.load_state_dict(checkpoint['net'])
    # ensemble_model.append(net)
    # print('Load denseNet with acc =', checkpoint['acc'])

    transform_train, transform_test = get_data_transforms(size=224)
    testset = SIIM_ISIC(type='test', data_root=args.data_root, csv_file=args.csv,
                        img_folder=args.img_folder, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )
    for net in ensemble_model:
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, meta, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
