import argparse
import random
import numpy as np
import torch
from trainer import Trainer


def str2bool(x):
    if x.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif x.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--train', type=str2bool, default=True,
                        help='train representation model')
    parser.add_argument('--eval', type=str, default='test',
                        help='evaluate on downstream task using validation ("val") or test ("test") data')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='enable cuda')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='used GPU if cuda is available and set to True and gpu > -1')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    parser.add_argument('--dset_dir', type=str, default='data',
                        help='dataset directory')
    parser.add_argument('--dataset', type=str, default='omniglot_aug',
                        help='dataset name: either "omniglot_original", "omniglot_aug" or "miniImagenet"')
    parser.add_argument('--image_size', type=int, default=28,
                        help='image size')

    parser.add_argument('--model', type=str, default='ISIF_M',
                        help='model for representation learning')
    parser.add_argument('--backbone', type=str, default='Conv4',
                        help='architecture backbone: either "Conv4" or "ResNet18"')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent dimension')

    parser.add_argument('--max_iter', type=float, default=2e5,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (for ProtoNets batch_size is overwritten by n_way)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='data loader num_workers')

    parser.add_argument('--downstream_model', type=str, default='l8_w20_s5',
                        help='downstream model name')
    parser.add_argument('--encoder_layer', type=int, default=8,
                        help='downstream model (classifier) takes features from that layer as input,'
                             'max-pooling is counted as individual layer')
    parser.add_argument('--n_way', type=int, default=20,
                        help='number of classes in few-shot classification')
    parser.add_argument('--n_support', type=int, default=5,
                        help='number of support examples per class (aka n_shot)')
    parser.add_argument('--n_query', type=int, default=15,
                        help='number of query examples per class')

    parser.add_argument('--save_step', type=int, default=100,
                        help='number of iterations before saving (replacing) checkpoint "last"')
    parser.add_argument('--save_new_step', type=list, default=[1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5],
                        help='number of iterations before creating new checkpoint.')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/omniglot_28_isif_m_c4_z64_bn_b64',
                        help='checkpoint directory')
    parser.add_argument('--ckpt_name', type=str, default='last',
                        help='load previous checkpoint CKPT_NAME of base model')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args)

    if args.train:
        print('Train base model: ', args.model)
        trainer.train()

    if args.eval in ['val', 'test']:
        # trainer.run_test()  # train and test downstream model on individual layer (specify n_way, n_support, n_query)
        trainer.test_all()  # run a bunch of tests
