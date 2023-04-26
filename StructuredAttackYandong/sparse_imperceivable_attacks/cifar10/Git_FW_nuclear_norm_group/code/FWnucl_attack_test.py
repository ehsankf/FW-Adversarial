import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import models
from models import *
from attacks import FW_vanilla_batch

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--n_examples', type=int, default=1000)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon',  type=float, default=1.0,
                    help='trace norm perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=1.,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for FWnucl')
parser.add_argument('--model-path',
                    default='./cifar10.pth',
                    help='model for white-box attack evaluation')
parser.add_argument('--attack', default='FW', 
                    help='model for attack')

#Frank-Wolf parameters
parser.add_argument('--p', default=2, type=float, 
                    help='p-wasserstein distance or for the lp ball.')
parser.add_argument('--subsampling', type=int, default=0,
                    help='subsampling per channel or not for Vanilla FW method')
parser.add_argument('--type_ball', default='nuclear',
                    help='choice of ball for vanilla FW')
parser.add_argument('--mask', default='True',
                    help='doing group lasso on line or on a grid for Vanilla FW.')
parser.add_argument('--size_groups', default=4, type=int,
                    help='when mask is true, it give the size of the masks to apply')
parser.add_argument('--L', default=10, type=float,
                    help='upper bound on the liptschitz constant of adversarial loss')
parser.add_argument('--adaptive_ls', type=bool, default=False,
                    help='if true then do the gapFW/L.. step size.')
parser.add_argument('--random_start', type=int, default=0, help='random restart')
parser.add_argument('--loss_type', default='Cross', type=str, 
                    choices=['Cross', 'Log'], required=True)

#wide_resnet

parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')


args = parser.parse_args()

'''
Usage:

python FWnucl_attack_test.py --test-batch-size 10 --n_examples 10 --loss_type Cross --attack FWnucl --model ResNet50 --model_path path/to/model/checkpoint
'''

def get_loss_and_preds(model, x, y):
        logits = model(normalize(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        _, preds = torch.max(logits, 1)
        return loss, preds


def run_eval(model, attack, epsilon, num_steps, step_size, p, type_ball,
             mask, subsampling, size_groups):
        eval_loss_clean, eval_acc_clean, eval_preds_clean = [], [], []
        total, correct, correct_adv = 0, 0, 0
        test_loss = 0
        eval_x_adv_lnuc = 0
        succ_tot = 0


        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            
            data, target = data.to(device), target.to(device)
            # pgd attack
            x, y = Variable(data, requires_grad=True), Variable(target)
            #x = unnormalize(x).to(device)
            y = y.to(device)

            loss_clean, preds_clean = get_loss_and_preds(model, x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((torch.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if attack == 'FWnucl_group':
               FW_van = FW_vanilla_batch(model, type_ball=type_ball,
                                 p=p,
                                 mask=mask,
                                 radius=epsilon,
                                 eps=1e-10,
                                 T_max=num_steps,
                                 step_size=3.0,
                                 channel_subsampling=subsampling,
                                 size_groups=size_groups,
                                 group_subsampling=subsampling,
                                 normalizer = normalize,
                                 rand=True,
                                 group_norm='group', 
                                 nbr_par=8,
                                 loss_type=args.loss_type,
                                 device=device)
               x_adv = FW_van.attack(x, y) 
               x_adv.to(device)
            if attack == 'FWnucl':
               FW_van = FW_vanilla_batch(model, type_ball=type_ball,
                                 p=p,
                                 mask=mask,
                                 radius=epsilon,
                                 eps=1e-10,
                                 T_max=num_steps,
                                 step_size=3.0,
                                 channel_subsampling=subsampling,
                                 normalizer = normalize,
                                 rand=True,
                                 nbr_par=8,
                                 loss_type=args.loss_type,
                                 device=device)
               x_adv = FW_van.attack(x, y)
               x_adv.to(device)

            x_adv = normalize(x_adv) 
            x = normalize(x)
   
            outputs_adv = model(x_adv)
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs_adv, y)

            test_loss += loss.item()
            _, predicted_adv = outputs_adv.max(1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            correct_adv += predicted_adv.eq(target).sum().item()
            

            succ_ind = ~ outputs_adv.max(1)[1].eq(target) & outputs.max(1)[1].eq(target)
            succ_num = succ_ind.sum().item()   

            if not succ_num == 0:
                x_succ, x_adv_succ = x[succ_ind], x_adv[succ_ind]
                succ_tot += succ_num

            succ_tot = succ_tot if not succ_tot == 0 else 1. 

            print(batch_idx, len(test_loader), 'Loss: %.3f| Clean Acc: %.3f%%  | Acc: %.3f%% Successs: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_adv/total, 100.*(total - correct_adv)/total, correct_adv, total))


global_step = 0

def main():
    model = models.__dict__[args.model]()
    model.load_state_dict(torch.load(args.model_path))
    pgd_params = {'attack': args.attack, 'epsilon': args.epsilon, 'num_steps': args.num_steps, 'step_size': args.step_size, 'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': bool(args.subsampling), 'size_groups': args.size_groups}
    run_eval(model, **pgd_params)
    print('******************')

if __name__ == '__main__':

        # settings
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # settings
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # set up data loader
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
        std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
        unnormalize = lambda x: x*std + mu
        normalize = lambda x: (x-mu)/std

        testset = torchvision.datasets.CIFAR10(root='/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

        x_test = torch.cat([x for (x, y) in test_loader], 0)[:args.n_examples]
        y_test = torch.cat([y for (x, y) in test_loader], 0)[:args.n_examples]
        
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.test_batch_size, shuffle=False, num_workers=0)
        

        main() 

        
