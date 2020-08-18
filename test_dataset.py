import torch
from process_data import get_data_transforms, SIIM_ISIC
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda:2'


def calculate_mean_std():
    trainset = SIIM_ISIC(transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    validset = SIIM_ISIC(train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    validloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for inputs, meta, targets in loader:
        data = inputs
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        if nb_samples % 10 == 0:
            print('step: ', nb_samples)
    for inputs, meta, targets in validloader:
        data = inputs
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        if nb_samples % 10 == 0:
            print('step: ', nb_samples)

    mean /= nb_samples
    std /= nb_samples
    print('mean: ', mean, 'std: ', std)


if __name__ == '__main__':
    train_transform, valid_transform = get_data_transforms()
    trainset = SIIM_ISIC(transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    validset = SIIM_ISIC(train=False, transform=train_transform)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=4, shuffle=True, num_workers=2)

    # how transform example
    image = Image.open('/home/group3/DataSet/Training_set/img_1.jpg')
    ori_img = transforms.ToTensor()(image)
    image = train_transform(image)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.imshow(ori_img.permute(1, 2, 0))
    plt.show()

    # calculate mean_std
    # calculate_mean_std()

    for batch_idx, (inputs, meta, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        sex = meta['sex']
        age = meta['age_approx']
        location = meta['anatom_site_general_challenge']
        # training process
        pass
