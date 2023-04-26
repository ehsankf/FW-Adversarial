import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import sampler

import sys
sys.path.insert(0, '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10')
from models import *

import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/AT-AWP_cifar10_linf_wrn34-10.pth --data CIFAR10 --preprocess 'meanstd'
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/AT-AWP_cifar100_linf_wrn34-10.pth --data CIFAR100 --preprocess 'meanstd'

LBGAT

python eval.py --arch WideResNet34 --checkpoint ../../LBGAT/pretrained_models/cifar10_lbgat0_wideresnet34-10.pt --data CIFAR10 --preprocess '01'

"""


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    state_dict = {k.replace('module.attacker.model.', '').replace('module.model.','').\
                    replace('module.','').replace('model.',''):v for k,v in state_dict.items()}

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if ('sub_block' in k) or ('normalizer' in k) or ('attacker' in k):
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='WideResNet34',
                        choices=['WideResNet28', 'WideResNet34', 'PreActResNet18', 'ResNet50'])
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--preprocess', type=str, default='01',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--sample_rate', type=float, default=1.)
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)

    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', default=False, action='store_true')
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='./log.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--attack', type=str, default='auto_attack', choices=['auto_attack', 'FW_attack'],)

    args = parser.parse_args()
    num_classes = int(args.data[5:])

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')

    # model = ResNet18()
    if args.arch == 'WideResNet34':
        net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'WideResNet28':
        net = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'PreActResNet18':
        net = PreActResNet18(num_classes=num_classes)
    if args.arch == 'ResNet50':
        net = ResNet50()
    else:
        raise ValueError('Please use choose correct architectures.')

    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    net.load_state_dict(ckpt)

    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    model.to(device)
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform_chain, download=True)
    sampler_ind = torch.utils.data.SubsetRandomSampler(sampler(item, args.sample_rate)) if not args.sample_rate == 1. else None
    test_loader = data.DataLoader(item, batch_size=64, shuffle=False, sampler=sampler_ind, num_workers=0)

    from utils import sampler, eval_test, compute_norm

    nat_acc = eval_test(model, test_loader, device)

    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.attack == 'auto_attack':
        # load attack    
        from autoattack import AutoAttack
        adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path)
    
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
        from utils import sampler, eval_test, compute_norm
        nat_acc = eval_test(model, test_loader, device)
    
        # cheap version
        # example of custom version
        if args.version == 'custom':
            adversary.attacks_to_run = ['apgd-ce', 'fab']
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2

        # run attack and save images
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                         bs=args.batch_size)

    if args.attack == 'FW_attack':
        # load FW attack
        from eval_FWs_rand import FW_attack
        adv_complete = FW_attack(model, test_loader, args.epsilon, device)
        adv_complete = adv_complete.numpy()
    
    torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
            args.save_dir, args.attack, args.version, adv_complete.shape[0], args.epsilon))

