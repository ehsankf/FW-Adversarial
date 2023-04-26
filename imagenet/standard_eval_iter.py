import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

from utils import sampler, eval_test, compute_norm

import sys
sys.path.insert(0, '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')

import pdb
"""
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/AT-AWP_cifar10_linf_wrn34-10.pth --data CIFAR10 --preprocess 'meanstd'
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/AT-AWP_cifar100_linf_wrn34-10.pth --data CIFAR100 --preprocess 'meanstd'

LBGAT

python eval.py --arch WideResNet34 --checkpoint ../../LBGAT/pretrained_models/cifar10_lbgat0_wideresnet34-10.pt --data CIFAR10 --preprocess '01'


iters:

CUDA_VISIBLE_DEVICES=1 python eval_iter.py 
--arch WideResNet34 --data CIFAR100 --sample_rate 0.01 --train_model AWP

UDA_VISIBLE_DEVICES=1 python eval_iter.py --
arch WideResNet34 --data CIFAR100  --train_model AWP

"""

checkpoint_dict = {'CIFAR10': {'AWP': '../pretrained_models/AT-AWP_cifar10_linf_wrn34-10.pth', 
                               'LBGAT': '../../LBGAT/pretrained_models/cifar10_lbgat0_wideresnet34-10.pt',
                               'Standard': '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/Madry-PGD-L2/ResNet50/cifar_l2_0_5.pt'}, 
                   'CIFAR100': {'AWP': '../pretrained_models/AT-AWP_cifar100_linf_wrn34-10.pth', 
                       'LBGAT': '../../LBGAT/pretrained_models/cifar100_lbgat0_wideresnet34-10.pt'}, 'IMAGENET': {'Standard': 'imagenet_l2_3_0.pt'}}
preprocess_dict = {'Standard':'meanstd', 'AWP':'meanstd', 'LBGAT': '01'}

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'model' in state_dict.keys():
        state_dict = state_dict['model']

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    state_dict = {k.replace('module.attacker.model.', '').replace('module.model.','').\
                    replace('module.','').replace('model.',''):v for k,v in state_dict.items()}

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if ('sub_block' in k) or ('normalizer' in k) or ('attacker' in k):
            continue
        if any([k in s for s in ["optimizer", "schedule", "epoch"]]):
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
    parser.add_argument('--checkpoint', type=str, default='test')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'IMAGENET'],
                        help='Which dataset the eval is on')
    parser.add_argument('--train_model', type=str, default='AWP', choices=['AWP', 'LBGAT', 'Standard'],
                        help='specify the training model')
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

    args.checkpoint = checkpoint_dict[args.data][args.train_model]
    args.preprocess = preprocess_dict[args.train_model]
    if 'CIFAR' in args.data:
        num_classes = int(args.data[5:])
    else:
        num_classes = 1000

    if args.data == 'CIFAR10':
        args.data_dir = '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/data'

    if args.data == 'IMAGENET':
        args.data_dir = './ILSVRC2012_img_val'

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10' and args.train_model == 'Standard':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        elif args.data == 'IMAGENET':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
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
    elif args.arch == 'ResNet50' and args.data == 'IMAGENET':
        net = models.__dict__[args.arch.lower()]()
    elif args.arch == 'ResNet50':
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

    if args.data == "IMAGENET":
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),]
        transform_chain = transforms.Compose(transform_list)
        item = datasets.ImageFolder(root=args.data_dir, transform=transform_chain)
    else:
        item = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform_chain, download=True)

    sampler_ind = torch.utils.data.SequentialSampler(sampler(item, args.sample_rate)) if not args.sample_rate == 1. else None
    test_loader = data.DataLoader(item, batch_size=64, shuffle=False, sampler=sampler_ind, num_workers=6, pin_memory=True)

    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)


    
    from functools import partial
    print = partial(print, flush=True)

    epsilons_dict = {'auto_attack': [2./255., 4./255., 8./255.], 'FW_attack': [1., 3., 5.]}    
    for attack in ['auto_attack']:
     for epsilon in epsilons_dict[attack]:
        if attack == 'auto_attack':
            # load attack    
            from autoattack import AutoAttack
            adversary = AutoAttack(model, norm=args.norm, eps=epsilon, log_path=args.log_path)
    
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
    
        elif attack == 'FW_attack':
            # load FW attack
            from eval_FWs_rand import FW_attack
            adv_complete = FW_attack(model, test_loader, epsilon, device)
            adv_complete = adv_complete.detach()
        adv_data_loader = data.DataLoader(TensorDataset(adv_complete, y_test), batch_size=64, shuffle=False, num_workers=0)

        linf, l2, l0, lnuc = compute_norm(x_test, adv_complete)
        nat_acc = eval_test(model, test_loader, device)
        adv_acc = eval_test(model, adv_data_loader, device)
         
        torch.save({'x_test': x_test.detach().cpu().numpy(), 'y_test': y_test.detach().cpu().numpy(), \
                 'adv_complete': adv_complete.detach().cpu().numpy(), 'nat_acc': nat_acc, 'adv_acc': adv_acc,
                 'linf': linf, 'l2': l2, 'l0': l0, 'lnuc': lnuc}, 
                 '{}/{}_{}_{}_{}_1_{}_eps_{:.5f}.pth'.format(args.save_dir, \
                 args.data, args.train_model, attack, args.version, adv_complete.shape[0], epsilon))

