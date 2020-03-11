import os
import torch.optim as optim
import random
from model import *
from dataset import get_data


class Trainer(object):
    def __init__(self, args):
        # Device
        if args.cuda and torch.cuda.is_available():
            if args.gpu > -1:
                self.device = 'cuda:' + str(args.gpu)
            else:
                self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.eval = args.eval

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset.lower()
        if self.dataset == 'omniglot_original':
            self.in_channels = 1
            self.n_classes = 964
            self.n_data = 19280
        elif self.dataset == 'omniglot_aug':
            self.in_channels = 1
            self.n_classes = 4112
            self.n_data = 82240
        elif self.dataset == 'miniimagenet':
            self.in_channels = 3
            self.n_classes = 64
            self.n_data = 38400
        else:
            raise NotImplementedError
        self.image_size = args.image_size

        # Model
        self.backbone = args.backbone
        assert self.backbone in ['Conv4', 'ResNet18']
        self.z_dim = args.z_dim

        self.encoder_layer = args.encoder_layer
        self.n_way = args.n_way
        self.n_support = args.n_support
        self.n_query = args.n_query
        self.spc = self.n_support + self.n_query
        self.model = args.model
        self.base_model, self.unsupervised = self.init_model(args.model)
        self.base_model = self.base_model.to(self.device)
        self.downstream_model = args.downstream_model

        # Optimizer
        self.optim = optim.Adam(self.base_model.parameters(), lr=args.lr)

        self.max_iter = args.max_iter
        if self.model == 'ProtoNet':
            self.batch_size = self.n_way
        else:
            self.batch_size = args.batch_size
        self.workers = args.num_workers
        self.global_iter = 0

        # Statistics
        self.train_loss = []
        self.test_acc = []

        # Save and load
        self.save_step = args.save_step
        self.save_new_step = args.save_new_step
        self.ckpt_dir = args.ckpt_dir
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_base_model(self.ckpt_name)

    def init_model(self, model):
        if model == 'AE':
            base_model = AE(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'VAE':
            base_model = VAE(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'RotNet':
            base_model = RotNet(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'AET':
            base_model = AET(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'NPID':
            base_model = NPID(self.backbone, self.in_channels, self.image_size, self.z_dim, self.n_data)
            unsupervised = True
        elif model == 'ISIF':
            base_model = ISIF(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'ISIF_M':
            base_model = ISIF_M(self.backbone, self.in_channels, self.image_size, self.z_dim)
            unsupervised = True
        elif model == 'ProtoNet':
            base_model = ProtoNet(self.backbone, self.in_channels, self.image_size, self.z_dim, self.n_support)
            unsupervised = False
        else:
            raise NotImplementedError
        return base_model, unsupervised

    def train(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.base_model.train()
        train_loader = get_data(
            dataset=self.dataset,
            dset_dir=self.dset_dir,
            image_size=self.image_size,
            model=self.model,
            phase='train',
            unsupervised=self.unsupervised,
            spc=self.spc,
            batch_size=self.batch_size,
            workers=self.workers)

        stop = False
        while not stop:
            for data in train_loader:
                kwargs = {key: Variable(value.to(self.device)) for (key, value) in data.items()}
                output = self.base_model(**kwargs)
                kwargs['output'] = output
                loss = self.base_model.update(self.optim, **kwargs)

                self.global_iter += 1
                if self.global_iter % self.save_step == 0:
                    self.save_base_model('last')
                if self.global_iter in self.save_new_step:
                    self.save_base_model(str(self.global_iter))

                self.train_loss.append(loss.item())
                print('Iter: {}\ttrain loss: {:.6f}'.format(self.global_iter, loss.item()))

                if self.global_iter >= self.max_iter:
                    stop = True
                    break

    def test(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.base_model.eval()

        test_loader = get_data(
            dataset=self.dataset,
            dset_dir=self.dset_dir,
            image_size=self.image_size,
            model=self.model,
            phase=self.eval,
            unsupervised=self.unsupervised,
            spc=None,
            batch_size=1,
            workers=0
        )
        label_feat_dict = {}
        with torch.no_grad():
            for data in test_loader:
                for i in range(data['labels'].size(0)):
                    images = data['img'][i].to(self.device)
                    labels = data['labels'][i].tolist()
                    z = self.base_model.encode(images, last_layer=self.encoder_layer).cpu()
                    for j, label in enumerate(labels):
                        if label not in label_feat_dict:
                            label_feat_dict[label] = []
                        label_feat_dict[label].append(z[j])

        iter_num = 1000
        for _ in range(iter_num):
            class_list = label_feat_dict.keys()
            class_sample = random.sample(class_list, self.n_way)
            features = []
            for cl in class_sample:
                img_feat = label_feat_dict[cl]
                sample_ids = np.random.permutation(len(img_feat))[:self.n_support + self.n_query]
                # sample_ids = list(range(len(img_feat)))[:self.n_support + self.n_query]
                features.append(torch.stack([img_feat[i] for i in sample_ids]))
            features = torch.stack(features)
            features = Variable(features.to(self.device))  # dim = n_way x (n_support + n_query) x C x W x H

            scores = self.base_model.train_classifier(features, self.n_way, self.n_support, self.n_query)

            pred = scores.data.cpu().numpy().argmax(axis=1)
            target = np.repeat(range(self.n_way), self.n_query)
            acc = np.mean(pred == target) * 100
            self.test_acc.append(acc)
            print('{0:4.2f}, {1:4.2f}'.format(acc, np.mean(self.test_acc)))

        acc_all = np.array(self.test_acc)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        conf_95 = 1.96 * acc_std / np.sqrt(iter_num)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, conf_95))

        if self.eval == 'val':
            file_name = 'log_val.txt'
        elif self.eval == 'test':
            file_name = 'log_test.txt'
        log_file = os.path.join(self.ckpt_dir, file_name)
        with open(log_file, 'a') as f:
            f.write(self.downstream_model + '\n')
            f.write('{0:4.2f}%, {1:4.2f}%\n'.format(acc_mean, conf_95))

    def run_test(self):
        self.downstream_model = 'l' + str(self.encoder_layer) + '_w' + str(self.n_way) + \
                                '_s' + str(self.n_support)
        self.test_acc = []
        print('Train and test downstream model: ', self.downstream_model)
        self.test()

    def test_all(self):
        # Evaluate all layers
        print('Evaluate all layers')
        self.n_query = 15
        if self.dataset.lower() in ['omniglot_original', 'omniglot_aug']:
            dataset = 'omniglot'
            self.n_way = 20
        elif self.dataset.lower() == 'miniimagenet':
            dataset = 'miniimagenet'
            self.n_way = 5
        else:
            raise NotImplementedError
        self.n_support = 5

        final_test_acc = []
        for layer in range(1, self.base_model.encoder.num_layers + 1):
            self.encoder_layer = layer
            self.run_test()
            final_test_acc.append(np.mean(np.array(self.test_acc)))

        # Evaluate final layer
        print('Evaluate final layer')
        self.encoder_layer = self.base_model.encoder.num_layers
        if dataset == 'omniglot':
            self.n_way = 20
            self.n_support = 1
            self.run_test()

            self.n_way = 5
            self.n_support = 5
            self.run_test()

        self.n_way = 5
        self.n_support = 1
        self.run_test()

        # Evaluate best layer
        print('Evaluate best layer')
        self.encoder_layer = np.argmax(np.array(final_test_acc)) + 1
        if dataset == 'omniglot':
            self.n_way = 20
            self.n_support = 1
            self.run_test()

            self.n_way = 5
            self.n_support = 5
            self.run_test()

        self.n_way = 5
        self.n_support = 1
        self.run_test()

        # Overfitting
        print('Overfitting')
        if dataset == 'omniglot':
            self.n_way = 20
        elif dataset == 'miniimagenet':
            self.n_way = 5
        else:
            raise NotImplementedError
        self.n_support = 5

        checkpoints = [str(int(step)) for step in self.save_new_step if step <= self.max_iter]
        for c in checkpoints:
            self.load_base_model(c)
            self.downstream_model = 'l' + str(self.encoder_layer) + '_w' + str(self.n_way) +\
                                    '_s' + str(self.n_support) + '_' + c
            self.test_acc = []
            print('Train and test downstream model: ', self.downstream_model)
            self.test()

    def save_base_model(self, filename, silent=False):
        stats = {'train_loss': self.train_loss}
        model_states = {'net': self.base_model.state_dict()}
        optim_states = {'optim': self.optim.state_dict()}
        states = {'iteration': self.global_iter,
                  'stats': stats,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("Saved checkpoint '{}'".format(file_path))

    def load_base_model(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.global_iter = checkpoint['iteration']
            self.train_loss = checkpoint['stats']['train_loss']
            self.base_model.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("Loaded checkpoint '{}'".format(file_path))
        else:
            print("No checkpoint found at '{}'".format(file_path))
