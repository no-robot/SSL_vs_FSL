import os
import torch
import numpy as np
import random
from omniglot import Omniglot
from mini_imagenet import MiniImageNet
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from data_utils import Identity, Homography, RandomPerspective


def get_data(dataset, dset_dir, image_size, model, phase, unsupervised, spc, batch_size, workers):
    dataset = dataset.lower()
    if dataset == 'omniglot_original':
        dset_dir = os.path.join(dset_dir, 'omniglot')
        transformations = get_transform(dataset, image_size, model)
        if phase == 'train':
            train_data = Omniglot(
                root=dset_dir,
                phase='background',
                unsupervised=unsupervised,
                spc=spc,
                pre_transform=transformations['pre_transform'],
                transform=transformations['transform'],
                post_transform=transformations['post_transform']
            )
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=int(workers))
        else:
            test_data = Omniglot(root=dset_dir, phase='evaluation', spc=20, pre_transform=transformations['test_transform'])
            data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=int(workers))

    elif dataset == 'omniglot_aug':
        dset_dir = os.path.join(dset_dir, 'omniglot')
        transformations = get_transform(dataset, image_size, model)
        if phase == 'train':
            train_data = Omniglot(
                root=dset_dir,
                phase=phase,
                unsupervised=unsupervised,
                spc=spc,
                pre_transform=transformations['pre_transform'],
                transform=transformations['transform'],
                post_transform=transformations['post_transform']
            )
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=int(workers))
        else:
            test_data = Omniglot(root=dset_dir, phase=phase, spc=20, pre_transform=transformations['test_transform'])
            data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=int(workers))

    elif dataset == 'miniimagenet':
        dset_dir = os.path.join(dset_dir, 'miniImagenet')
        transformations = get_transform(dataset, image_size, model)
        if phase == 'train':
            train_data = MiniImageNet(
                root=dset_dir,
                phase=phase,
                unsupervised=unsupervised,
                spc=spc,
                pre_transform=transformations['pre_transform'],
                transform=transformations['transform'],
                post_transform=transformations['post_transform']
            )
            data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=int(workers))
        else:
            test_data = MiniImageNet(root=dset_dir, phase=phase, spc=600, pre_transform=transformations['test_transform'])
            data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=int(workers))

    else:
        raise NotImplementedError

    return data_loader


def get_transform(dataset, image_size, model):
    if dataset in ['omniglot_original', 'omniglot_aug']:
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        if model in ['AE', 'VAE', 'RotNet']:
            pre_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
            transform = None
            post_transform = None
        elif model == 'AET':
            pre_transform = None
            param_normalize = transforms.Normalize(
                [0., 0., 16., 0., 0., 16., 0., 0.],
                [1., 1., 20., 1., 1., 20., 0.015, 0.015]
            )
            transform = Homography(
                shift=4,
                scale=(0.8, 1.2),
                fillcolor=255,
                resample=Image.BILINEAR,
                normalize=param_normalize
            )
            post_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        elif model in ['ISIF', 'ISIF_M']:
            pre_transform = None
            transform = Identity()
            post_transform = transforms.Compose([
                RandomPerspective(distortion_scale=0.5, interpolation=Image.BILINEAR, fillcolor=255),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        elif model in ['NPID', 'Baseline', 'ProtoNet']:
            pre_transform = transforms.Compose([
                RandomPerspective(distortion_scale=0.5, interpolation=Image.BILINEAR, fillcolor=255),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
            transform = None
            post_transform = None
        else:
            raise NotImplementedError

    elif dataset == 'miniimagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if model in ['AE', 'VAE']:
            test_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize
            ])

        if model in ['AE', 'VAE']:
            pre_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            transform = None
            post_transform = None
        elif model == 'RotNet':
            pre_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform = None
            post_transform = None
        elif model == 'AET':
            pre_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])
            param_normalize = transforms.Normalize(
                [0., 0., 16., 0., 0., 16., 0., 0.],
                [1., 1., 20., 1., 1., 20., 0.045, 0.045]
            )
            transform = Homography(
                shift=12,
                scale=(0.8, 1.2),
                fillcolor=(128, 128, 128),
                resample=Image.BILINEAR,
                normalize=param_normalize
            )
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        elif model in ['ISIF', 'ISIF_M']:
            pre_transform = None
            transform = Identity()
            post_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif model in ['NPID', 'Baseline', 'ProtoNet']:
            pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform = None
            post_transform = None
        else:
            raise NotImplementedError

    transformations = {
        'pre_transform': pre_transform,
        'transform': transform,
        'post_transform': post_transform,
        'test_transform': test_transform
    }
    return transformations


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dset_dir = './data'
    flag = 2
    if flag == 1:
        dataset = 'omniglot_original'
        image_size = 28
        color_map = 'gray'
    elif flag == 2:
        dataset = 'omniglot_aug'
        image_size = 28
        color_map = 'gray'
    elif flag == 3:
        dataset = 'miniImagenet'
        image_size = 224
        color_map = None

    model = 'AET'
    phase = 'train'
    unsupervised = True
    spc = 10
    batch_size = 2
    workers = 0

    data_loader = get_data(
        dataset=dataset,
        dset_dir=dset_dir,
        image_size=image_size,
        model=model,
        phase=phase,
        unsupervised=unsupervised,
        spc=spc,
        batch_size=batch_size,
        workers=workers)

    for i, data in enumerate(data_loader, 0):
        img = data['img']  # augmented images
        # p = data['p']
        # m, s = p.mean(dim=0), p.std(dim=0)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        if not train or not unsupervised:
            img = img[0]
        plt.imshow(img[0].permute(1, 2, 0).squeeze(), cmap=color_map)
        if 'img2' in data:
            img2 = data['img2']  # transformed images
            if not train or not unsupervised:
                img2 = img2[0]
            fig.add_subplot(1, 2, 2)
            plt.imshow(img2[0].permute(1, 2, 0).squeeze(), cmap=color_map)
        plt.show()
