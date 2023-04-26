from __future__ import print_function
import os
import sys
import sh
import csv
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import foolbox

sys.path.insert(0, '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10')
import models
from models import *
#from fast_adv.models.mnist.small_cnn import *
from fast_adv.attacks import DDN, CarliniWagnerL2, FW_vanilla_batch, PGD
from fast_adv.utils import requires_grad_
#from fast_adv.models.net_mnist import *
import pdb

sys.path.insert(0, './')
sys.path.insert(0, '../')
import cornersearch_attacks_pt

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--n_examples', type=int, default=1000)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--load_pgd_test_samples', default=None, type=str)
parser.add_argument('--epsilon',  type=float, default=8/255.,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2/255.,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./mnist.pth',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_mnist_smallcnn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_mnist_smallcnn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--attack', default='pgd', 
                    help='model for attack')
parser.add_argument('--mean_samples', default=16)
parser.add_argument('--mean_eps', default=.1)
parser.add_argument('--dict', type=int, default=1)

parser.add_argument('--log-index', type=int, default=5)

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

 CUDA_VISIBLE_DEVICES=1 python CS_attack_cifar10
_robust_FWs_rand.py --test-batch-size 10 --n_examples 10 --loss_type Cross
'''

log_dir = args.attack + str(args.num_steps) + 'eps' + str(args.epsilon) + args.model + args.model_path.split('/')[1].split('.')[0]

sh.rm('-rf', log_dir)






def attack_foolbox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel, criterion=attack_criteria)
    x_adv = attack(X.cpu().detach().numpy(), y.numpy())

    return torch.from_numpy(x_adv)



def _pgd_whitebox(model,
                  x,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size, 
                  sum_dir = None):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    x_pgd = Variable(x.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for i in range(num_steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(normalize(x_pgd)), y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, 0, 1), requires_grad=True)
        if sum_dir:
           loss_clean, loss_Adv, loss_incr, clean_acc, preds_acc, eval_x_adv_linf, eval_x_adv_l0, eval_x_adv_l2 = attack_log(model, x, x_pgd, y, i, sum_dir)
    
    return x_pgd.detach()

def fgsm(model,
         x,
         y,
         epsilon=args.epsilon):
    #err = (out.data.max(1)[1] != y.data).float().sum()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(normalize(x)), y)
    loss.backward()
    # signed gradien
    eta = args.epsilon * x.grad.detach().sign()
    # Add perturbation to original example to obtain adversarial example
    x_adv = x.detach() + eta
    x_adv = torch.clamp(x_adv, 0, 1)
    flattened = (x_adv - x.detach()).view(x.shape[0], -1)
    print("Linf2 norm", torch.norm((x_adv - x.detach()).view(x.shape[0], -1), float('inf'), dim = 1).max())
    linfflatened = torch.norm(flattened, float('inf'), dim = 1).sum()/len(y)
    
    return x_adv.detach()


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    pgd_robust_err_total = 0
    natural_err_total = 0
    
    #start = time.time()
    #cw_time = time.time() - start

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        natural_err_total += err_natural
        pgd_adv = _pgd_whitebox(model, X, y)
        pgd_err_robust = (model(pgd_adv).data.max(1)[1] != y.data).float().sum()
        print('err pgd (white-box): ', pgd_err_robust)
        pgd_robust_err_total += pgd_err_robust
        
    print('natural_err_total: ', natural_err_total)
    print('pgd robust_err_total: ', pgd_robust_err_total)


def eval_adv_test_Carlini(model, device, test_loader):
    """
    evaluate model by Carlini attack
    """
    model.eval()    
    cw_robust_err_total = 0
    natural_err_total = 0
    
    print('Running C&W attack')
    cwattacker = CarliniWagnerL2(device=device,
                                 image_constraints=(0, 1),
                                 num_classes=10)

    #start = time.time()
    #cw_time = time.time() - start

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # cw attack
        X, y = Variable(data), Variable(target)
        X = X.to(device)
        y = y.to(device)
        requires_grad_(model, False)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        natural_err_total += err_natural
        cw_adv = cwattacker.attack(model, X, labels=y, targeted=False)
        cw_err_robust = (model(cw_adv).data.max(1)[1] != y.data).float().sum()
        print('err cw (white-box): ', cw_err_robust)
        cw_robust_err_total += cw_err_robust
    print('natural_err_total: ', natural_err_total)
    print('cw robust_err_total: ', cw_robust_err_total)


def eval_train(model, device, test_loader):
    """
    evaluate model
    """
    model.eval() 
    natural_err_total = 0
    total = 0
    
    #start = time.time()
    #cw_time = time.time() - start

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data), Variable(target)
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        total += target.size(0)
        #print('natural err: ', err_natural)
        natural_err_total += err_natural
    print('natural_err_total: ', 100.*natural_err_total/total)
    print('acc_total: ', 100.*(total - natural_err_total)/total)
    print("error", natural_err_total, "total:", total)


def eval_test(model, device, test_loader):
    """
    evaluate model
    """
    model.eval() 
    natural_err_total = 0
    total = 0
    
    #start = time.time()
    #cw_time = time.time() - start

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data), Variable(target)
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        total += target.size(0)
        #print('natural err: ', err_natural)
        natural_err_total += err_natural
    print('natural_err_total: ', 100.*natural_err_total/total)
    print('acc_total: ', 100.*(total - natural_err_total)/total)
    print("error", natural_err_total, "total:", total)

def test_generalization(model, device, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            inputs, targets = data.to(device), target.to(device)
            with torch.no_grad():
                outputs = model(normalize(inputs))

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rGeneralization... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()
    print("correct", correct, "total:", total)
    return 100. * correct / total



def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def get_loss_and_preds(model, x, y):
        logits = model(normalize(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        _, preds = torch.max(logits, 1)
        return loss, preds

def clip(x, cmin, cmax):
        return torch.min(torch.max(x, cmin), cmax)

def attack_mean(model, x, y, epsilon=args.epsilon):
        clip_min = 0.
        clip_max = 1
        x_advs = _pgd_whitebox(model, x, y, epsilon)
        for _ in tqdm.trange(args.mean_samples, desc='mean attack samples'):
            x_noisy = x + torch.empty_like(x).uniform_(-args.mean_eps, args.mean_eps)
            x_advs += _pgd_whitebox(model, x_noisy, y, epsilon)
        x_advs = x_advs / (args.mean_samples + 1)
        x_advs.clamp_(clip_min, clip_max)
        x_advs = clip(x_advs, x-epsilon, x+epsilon)
        return x_advs

def attack_log(model, x, x_adv, y, step, summaryname):
            loss_clean, preds_clean = get_loss_and_preds(model, x, y)
            print(len(y))
            clean_acc = 100.*preds_clean.eq(y).sum().item()/len(y)

            loss_adv, preds_adv = get_loss_and_preds(model, x_adv, y)
            preds_acc = 100.*preds_adv.eq(y).sum().item()/len(y)
            
            loss_incr = loss_adv - loss_clean
            adv_perturb = (x - x_adv).view(x.size(0), -1)

            eval_x_adv_linf = torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()/len(y)
            eval_x_adv_l0 = torch.norm(adv_perturb, p=0, dim = 1).sum().item()/len(y)
            eval_x_adv_l2 = torch.norm(adv_perturb, p=2, dim = 1).sum().item()/len(y)
            eval_x_adv_lnuc = torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()/len(y) 

            with open(summaryname, 'a') as summaryfile:
               summarywriter = csv.writer(summaryfile, delimiter=',') 
               summarywriter.writerow([loss_clean.item(), loss_adv.item(), loss_incr.item(), clean_acc, preds_acc, 
                                    eval_x_adv_linf, eval_x_adv_l0, eval_x_adv_l2, 
                                    eval_x_adv_lnuc])           

                        
                    
            print('loss_clean', loss_clean.data, step)
            print('loss_adv', loss_adv.data, step)
            print('loss_incr', loss_incr.data, step)
            print('clean_acc', clean_acc, step)
            print('preds_acc', preds_acc, step)
            print('eval_x_adv_linf', eval_x_adv_linf, step)
            print('eval_x_adv_l0', eval_x_adv_l0, step)
            print('eval_x_adv_l2', eval_x_adv_l2, step)
            print('eval_x_adv_lnuc', eval_x_adv_lnuc, step)
            
                   
            #conf_adv = confusion_matrix(preds_clean.cpu(), preds_adv.cpu(), np.arange(nb_classes))
            #conf_adv -= np.diag(np.diag(conf_adv))
            #eval_conf_adv.append(conf_adv)
            return loss_clean, loss_adv, loss_incr, clean_acc, preds_acc, eval_x_adv_linf, eval_x_adv_l0, eval_x_adv_l2
            
            

if args.attack == 'cw':
        cwattacker = CarliniWagnerL2(device=device,
                                 image_constraints=(0, 1),
                                 num_classes=10)
if args.attack == 'L2':
   attacker = PGD(F.cross_entropy, eps=args.epsilon, 
                 alpha=args.step_size,
                 type_projection='l2',
                 iters=args.num_steps,
                 device=device, 
                 p=2, normalizer=normalize)
 
elif args.attack == 'Linf':
   attacker = PGD(nn.CrossEntropyLoss(), eps=args.epsilon, 
                 alpha=args.step_size,
                 type_projection='linfty',
                 iters=args.num_steps,
                 device=device, 
                 p=2, normalizer=normalize)


def run_eval(model, attack, epsilon, num_steps, step_size, p, type_ball,
             mask, subsampling, size_groups, logname, summaryname, with_attack=True):
        logging.info('eval')
        model.eval()
        eval_loss_clean = []
        eval_acc_clean = []
        eval_loss_rand = []
        eval_acc_rand = []
        eval_loss_adv = []
        eval_acc_adv = []
        eval_loss_pand = []
        eval_acc_pand = []
        all_outputs = []
        diffs_rand, diffs_adv, diffs_pand = [], [], []
        eval_preds_clean, eval_preds_rand, eval_preds_adv, eval_preds_pand = [], [], [], []
        norms_clean, norms_adv, norms_rand, norms_pand = [], [], [], []
        norms_dadv, norms_drand, norms_dpand = [], [], []
        eval_important_valid = []
        eval_loss_incr = []
        eval_conf_adv = []
        wdiff_corrs = []
        udiff_corrs = []
        grad_corrs = []
        minps_clean = []
        minps_adv = []
        acc_clean_after_corr = []
        acc_adv_after_corr = []
        eval_det_clean = []
        eval_det_adv = []
        eval_x_adv_l0 = []
        eval_x_adv_l2 = []

        all_eval_important_pixels = []
        all_eval_important_single_pixels = []
        all_eval_losses_per_pixel = []
         
        test_loss = 0
        total = 0
        correct = 0
        correct_pgd = 0
        eval_x_adv_linf = 0
        eval_x_adv_l0 = 0
        eval_x_adv_l2 = 0          
        eval_x_adv_lnuc = 0
        succ_tot = 0


        with open(logname, 'w') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow(['batch_idx', 'Test_Loss', 'Clean_Test_Acc', 'Test_Acc', 'Success_rate', 
                              'Linf', 'L2_norm', 'L0_norm', 'Lnuc_norm', 'Correct', 'Total'])
        
        with open(summaryname, 'w') as logfile:
           summarywriter = csv.writer(logfile, delimiter=',')
           summarywriter.writerow(['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf',
                              'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc', 'FW gap'])


        start = time.time()    

        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            
            #if args.load_pgd_test_samples:
            #    x, y, x_pgd = eval_batch
            #else:
            global global_step 
            global_step = global_step + 1 

            data, target = data.to(device), target.to(device)
            # pgd attack
            x, y = Variable(data, requires_grad=True), Variable(target)
            #x = unnormalize(x).to(device)
            y = y.to(device)

            loss_clean, preds_clean = get_loss_and_preds(model, x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((torch.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if args.log_index == batch_idx:
               summarynameindex = summaryname
            else: 
               summarynameindex = None
         
            if attack == 'pgd':
               x_adv = _pgd_whitebox(model, x, y, 
                                     epsilon=epsilon,
                                     step_size=step_size,
                                     num_steps=num_steps, sum_dir=summarynameindex)#(x, preds_clean, eps=args.eps_eval) 
               #x_adv = attack_foolbox(model, x, y)
               x_adv.to(device)
            elif attack == 'fgsm':
               x_adv = fgsm(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'pgdl2':
               attack_pgdl2 = PGDL2(eps=epsilon, alpha=step_size,
                 type_projection='l2',
                 iters=num_steps, rand='False', device=device,
                 p=2, normalizer=normalize) #attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv = attack_pgdl2.attack(model, x, y)
               x_adv.to(device)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            elif attack == 'CS':
   
               CS_args = {'type_attack': 'L0+sigma',
               'n_iter': 1000,
               'n_max': 150,
               'kappa': 0.4,
               'epsilon': -1,
               'sparsity': 100,
               'size_incr': 5}
    
               CS_attack = cornersearch_attacks_pt.CSattack(model, CS_args)
               x = normalize(x)
               x_adv, _, _ = CS_attack.perturb(x.permute(0, 2, 3, 1).detach().cpu().numpy(), y.detach().cpu().numpy())
               x_adv = torch.from_numpy(x_adv).permute(0, 3, 1, 2).to(device)
            elif attack == 'FW_group':
               print('Computing vanilla FW for batch:')
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
                                 sum_dir=summarynameindex,
                                 normalizer = normalize,
                                 rand=True,
                                 group_norm='group', 
                                 nbr_par=8,
                                 loss_type=args.loss_type,
                                 device=device)
               print("DEviiiiiiiiiiiiice", device)
               x_adv = FW_van.attack(x, y) 
               x_adv.to(device) 
            elif attack == 'L2' or attack == 'Linf':
               x_adv = attacker.attack(model, x, y)      
               x_adv.to(device)         
            elif attack == 'mean':
               x_adv = attack_mean(model, x, y, epsilon=epsilon)
            #x_adv = normalize(x_adv).to(device)   
            #x = normalize(x).to(device)    
            x_adv = x_adv if attack == 'CS' else normalize(x_adv) 
            x = x if attack == 'CS' else normalize(x)
   
            outputs_pgd = model(x_adv)
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs_pgd, y)

            test_loss += loss.item()
            _, predicted_pgd = outputs_pgd.max(1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            correct_pgd += predicted_pgd.eq(target).sum().item()
            

            succ_ind = ~ outputs_pgd.max(1)[1].eq(target) & outputs.max(1)[1].eq(target)
            succ_num = succ_ind.sum().item()   

            if not succ_num == 0:
                x_succ, x_adv_succ = x[succ_ind], x_adv[succ_ind]
                adv_perturb = (x_succ - x_adv_succ).view(x_succ.size(0), -1)
                eval_x_adv_linf += torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()
                eval_x_adv_l0 += torch.sum(torch.max(torch.abs(x_succ - x_adv_succ) > 1e-10, dim=1)[0], dim=(1,2)).sum().item()
                eval_x_adv_l2 += torch.norm(adv_perturb, p=2, dim = 1).sum().item()           
                eval_x_adv_lnuc += torch.norm(x_succ[:, 0, :, :] - x_adv_succ[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()  
                succ_tot += succ_num

            succ_tot = succ_tot if not succ_tot == 0 else 1. 

            print(batch_idx, len(test_loader), 'Loss: %.3f| Clean Acc: %.3f%%  | Acc: %.3f%% Successs: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, correct_pgd, total))

            print(batch_idx, len(test_loader), 'Linf norm %.3f | L2 norm %.3f | L0 norm %.3f | Lnuc norm %.3f | (/%d)' 
            % (eval_x_adv_linf/total, eval_x_adv_l2/total, eval_x_adv_l0/total, eval_x_adv_lnuc/total, total))

            with open(logname, 'a') as logfile:
               logwriter = csv.writer(logfile, delimiter=',')
               logwriter.writerow([batch_idx, test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total,
                                  eval_x_adv_linf/succ_tot, eval_x_adv_l2/succ_tot, eval_x_adv_l0/succ_tot, eval_x_adv_lnuc/succ_tot, correct_pgd, total])
            
        '''    
        attack_time = time.time() - start  
        with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile) 
           logwriter.writerow(["Time: ", attack_time]) 
        '''
'''

def save_checkpoint(state: OrderedDict, filename):
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    torch.save(state, filename)
'''

def load_checkpoint(state, model):
        new_state = {}
        for k in state.keys():
            newk = k.replace('model.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k]
        state = new_state
        print(state.keys())
        model.load_state_dict(state)
        return model

def load_model():

    if args.white_box_attack:
       
        if args.model == 'wide_resnet':
            model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
        else: 
            model = models.__dict__[args.model]()

        #model = ResNet50()
        #model = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=args.drop)
        '''
        model = ResNet18()
        model = model.to(device)
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'), pickle_module=dill)
        print("Keys", checkpoint.keys())
        model.load_state_dict(checkpoint, strict = False)
        '''
        import dill
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'), pickle_module=dill)
        print(checkpoint.keys())
        #print(checkpoint['model'].keys())
        
        
        state_dict_path = 'model'
        if not ('model' in checkpoint):
           state_dict_path = 'state_dict'
        if ('net' in checkpoint):
           checkpoint['model'] = checkpoint['net']
           del checkpoint['net']
        
        #sd = checkpoint[state_dict_path] 
        if args.dict:
           if 'model' in checkpoint:
              if hasattr(checkpoint['model'], 'state_dict'):
                 print("Hi ehsan*******************")
                 sd = checkpoint['model'].state_dict()
              else:
                 sd = checkpoint['model']
           elif 'state_dict' in checkpoint:
              sd = checkpoint['state_dict']
              print ('epoch', checkpoint['epoch'],
                     'arch', checkpoint['arch'],
                     'nat_prec1', checkpoint['nat_prec1'], 
                     'adv_prec1', checkpoint['adv_prec1'])
           else:
              sd = checkpoint 
              print(sd.keys())
           #sd = sd.state_dict()
           #print(sd.keys())
           #sd = {k[len('module.attacker.model.'):]:v for k,v in sd.items()}
           
           sd = {k.replace('module.attacker.model.', '').replace('module.model.','').replace('module.','').replace('model.',''):v for k,v in sd.items()}
        
           keys = model.state_dict().keys()
           new_state = {}
           for k in sd.keys():
              if k in keys:
                 new_state[k] = sd[k]
              else:
                 print(k)
        
           model.load_state_dict(new_state)
        else:
           model = checkpoint['model']
        checkpoint = None
        sd = None
        
        #rng_state = checkpoint['rng_state']
        #torch.set_rng_state(rng_state)
        
        #model = ResNet18()
        #model = load_checkpoint(torch.load(args.model_path, map_location=torch.device('cpu')), model)
        #model = SmallCNN().to(device)
        #model = globals()[args.model]().to(device)
        #model.load_state_dict()
        #model = load_checkpoint(torch.load(args.model_path, map_location=torch.device('cpu')), model)
        
        
        model.eval().to(device)

    return model

global_step = 0

def mk_log_file(attack, num_steps, epsilon):

        if not os.path.isdir('summary'):
            os.mkdir('summary')

        if not os.path.isdir('logger'):
            os.mkdir('logger')

        log_dir = attack + 'iter' + str(num_steps) + 'eps' + str(epsilon) + args.model + args.model_path.split('/')[1].split('.')[0]
        if not os.path.isdir('logger/log_rand_cross_' + attack + '_' + args.model + '/'):
           os.mkdir('logger/log_rand_cross_' + attack + '_' + args.model + '/')
        if not os.path.isdir('summary/sum_rand_cross_' + attack + '_' + args.model + '/'):
           os.mkdir('summary/sum_rand_cross_' + attack + '_' + args.model + '/')

        logname = ('logger/log_rand_cross_' + attack + '_' + args.model + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_rand_cross_' + attack + '_' + args.model + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname

def main():
        
    model = load_model()
        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)
    if adv_atc == 'CS': 
      for eps in [-1]:   
        pgd_params = {'attack': 'CS', 'epsilon': eps, 'num_steps': 20, 'step_size': 0.1,
        'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': False, 'size_groups': False}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("%s parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['attack'], pgd_params['epsilon'], pgd_params['num_steps']))
        run_eval(model, **pgd_params)
        print('******************')

    if adv_atc == 'FW_group':
      for eps in [1.0]:
        step_size =  1
        pgd_params = {'attack': 'FW_group', 'epsilon': eps, 'num_steps': 20, 'step_size': step_size,
        'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': bool(args.subsampling), 'size_groups': args.size_groups}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("pgd parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['epsilon'], pgd_params['num_steps']))
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        run_eval(model, **pgd_params)
        print('******************')
        
        
        '''
        pgd_params['num_steps'] = 100
        print("pgd parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['epsilon'], pgd_params['num_steps']))
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params
        run_eval(model, **pgd_params)
        print('******************')
        '''
        

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

        #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

        x_test = torch.cat([x for (x, y) in test_loader], 0)[:args.n_examples]
        y_test = torch.cat([y for (x, y) in test_loader], 0)[:args.n_examples]
        
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.test_batch_size, shuffle=False, num_workers=0)
        

        for adv_atc in ['CS']:

            args.model = 'ResNet50'
            args.model_path = '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/Madry-PGD-L2/ResNet50/cifar_l2_0_5.pt'
            main()
        
        pdb.set_trace()
        for adv_atc in ['FW_group', 'CS']:
            args.model = 'ResNet18'
            args.model_path = '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/Madry-PGD-L2/ResNet18/checkpoint.pt.best'
            main()

        for adv_atc in ['FW_group', 'CS']:
            args.model = 'WideResNet'
            args.model_path = '/data/ehsan/Adv-Ex/test_attack2/test-attackYan/attack_cifar10/Madry-PGD-L2/WideResNet/checkpoint.pt.best'
            main()

        

        '''
        args.model = 'Net'
        args.model_path = 'mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        main(eps=3)

        args.model = 'SmallCNN'
        args.model_path = 'mnist_virginia/mnist/LeNet-inf2/mnist/f383270a-4119-4782-922a-a7a80444c226/checkpoint.pt.best'
        main(eps=5)

        args.model = 'Net'
        args.model_path = 'mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        main(eps=5)
        '''
