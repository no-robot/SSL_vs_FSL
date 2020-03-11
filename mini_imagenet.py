import torch
from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from os.path import join
import re
from data_utils import Subset


class MiniImageNet(Dataset):
    def __init__(self, root, phase='train', unsupervised=True, spc=600,
                 pre_transform=None, transform=None, post_transform=None):
        assert phase in ['train', 'val', 'test']
        target_folder = join(root, 'train')
        label_path_dict = {}
        with open(join(root, phase + ".csv"), "r") as lines:
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                index, _, label = re.split(',|\.', line)
                index = int(index[-5:]) - 1
                label = label.replace('\n', '')
                if label not in label_path_dict:
                    label_path_dict[label] = []
                    path_list = listdir(join(target_folder, label))
                    id_list = [int(re.split('_|\.', path)[1]) for path in path_list]
                    sorted_path_list = [path_list[fid[0]] for fid in sorted(enumerate(id_list), key=lambda x: x[1])]
                path = join(target_folder, label, sorted_path_list[index])
                label_path_dict[label].append(path)

        self.unsupervised = True if unsupervised and phase == 'train' else False
        self.spc = spc  # samples per class (number of support plus query images) for few-shot classification
        if self.unsupervised:
            self.path_label_list = [[path, i] for i, (_, path_list) in enumerate(label_path_dict.items())
                                    for path in path_list]
        else:
            self.subset_loaders = [Subset(path_list, spc) for _, path_list in label_path_dict.items()]

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
            img = Image.open(file_path, mode='r').convert('RGB')
            data = {'idx': index}

            # Comment out the following lines to sample true positives instead of merely augmented images
            # start_idx = (index // 600) * 600
            # end_idx = start_idx + 600
            # pos_idx = [i for i in range(start_idx, end_idx) if i != index]
            # pos_idx = pos_idx[np.random.randint(600 - 1)]
            # file_path, _ = self.path_label_list[pos_idx]
            # img2 = Image.open(file_path, mode='r').convert('RGB')
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
            img = [Image.open(file_path, mode='r').convert('RGB') for file_path in path_list]
            img = torch.stack([self.pre_transform(x) for x in img])
            labels = torch.LongTensor([index]).repeat(self.spc)
            data = {'img': img, 'labels': labels}
            return data
