import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import list_dir, list_files
from PIL import Image
from os.path import join
from data_utils import Subset
import json


class Omniglot(Dataset):
    """Omniglot: https://github.com/brendenlake/omniglot"""
    def __init__(self, root, phase='train', unsupervised=True, spc=20,
                 pre_transform=None, transform=None, post_transform=None):
        assert phase in ['background', 'evaluation', 'train', 'val', 'trainval', 'test']
        # 'background' and 'evaluation' correspond to the original data split
        # 'train', 'val', 'trainval' and 'test' correspond to the split according to Santoro et al./Vinyals et al.
        if phase == 'background':
            target_folder = join(root, 'images_background')
        elif phase == 'evaluation':
            target_folder = join(root, 'images_evaluation')
        elif phase in ['train', 'trainval']:
            target_folder = join(root, 'base.json')
        elif phase == 'val':
            target_folder = join(root, 'val.json')
        else:
            target_folder = join(root, 'novel.json')

        if phase in ['background', 'evaluation']:
            alphabets = list_dir(target_folder)
            self.characters = sum([[join(target_folder, a, c) for c in list_dir(join(target_folder, a))]
                                   for a in alphabets], [])
            images_and_labels = [[join(character, image) for image in list_files(character, '.png')]
                                 for character in self.characters]
        else:
            with open(target_folder, 'r') as f:
                self.meta = json.load(f)
            images_and_labels = [self.meta['image_names'][i:i + 20]
                                 for i in range(0, len(self.meta['image_names']), 20)]

            if phase == 'trainval':
                target_folder = join(root, 'val.json')
                with open(target_folder, 'r') as f:
                    self.meta = json.load(f)
                images_and_labels.extend([self.meta['image_names'][i:i + 20]
                                          for i in range(0, len(self.meta['image_names']), 20)])

        self.unsupervised = True if unsupervised and phase in ['background', 'train', 'trainval'] else False
        self.spc = spc  # samples per class (number of support plus query images) for few-shot classification

        if self.unsupervised:
            self.path_label_list = [[path, i] for i, path_list in enumerate(images_and_labels)
                                    for path in path_list]
        else:
            self.subset_loaders = [Subset(path_list, spc) for path_list in images_and_labels]

        self.pre_transform = pre_transform
        self.transform = transform
        self.post_transform = post_transform

    def __len__(self):
        if self.unsupervised:
            return len(self.path_label_list)
        else:
            return len(self.subset_loaders)

    def __getitem__(self, index):
        if self.unsupervised:
            file_path, labels = self.path_label_list[index]
            img = Image.open(file_path, mode='r').convert('L')
            data = {'idx': index}

            # Comment out the following lines to sample true positives instead of merely augmented images
            # start_idx = (index // 20) * 20
            # end_idx = start_idx + 20
            # pos_idx = [i for i in range(start_idx, end_idx) if i != index]
            # pos_idx = pos_idx[np.random.randint(20 - 1)]
            # file_path, _ = self.path_label_list[pos_idx]
            # img2 = Image.open(file_path, mode='r').convert('L')
            # img2 = self.post_transform(img2)
            # data.update(img2=img2)
            # data.update(labels=labels)

            if self.pre_transform is not None:
                img = self.pre_transform(img)

            if self.transform is not None:
                img2, p = self.transform(img)
                img2 = self.post_transform(img2)
                data.update(img2=img2, p=p)

            if self.post_transform is not None:
                img = self.post_transform(img)

            data.update(img=img)
            return data
        else:
            path_list = next(self.subset_loaders[index])
            img = [Image.open(file_path, mode='r').convert('L') for file_path in path_list]
            img = torch.stack([self.pre_transform(x) for x in img])
            labels = torch.LongTensor([index]).repeat(self.spc)
            data = {'img': img, 'labels': labels}
            return data
