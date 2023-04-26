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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from pgd import  PGD

import models
from models import *
#from fast_adv.models.mnist.small_cnn import *
from FW_vanilla_batch import *
#from fast_adv.models.net_mnist import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 200)')
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

python pgd_attack_cifar10_robust_FWs.py  --loss Log

python pgd_attack_cifar10.py --test-batch-size 64 --attack FW --model-path Madry-resnet50/cifar_linf_8.pt  --num-steps 100  --model ResNet50  --epsilon 5 --step-size 1

python pgd_attack_cifar10.py --test-batch-size 64 --attack pgd --model-path Madry-PGD-Linf/ResNet18/checkpoint_lr_0.1.pt.best --model ResNet50 --epsilon 0.03
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
         epsilon):
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


def run_eval(model, ex_loader, attack, epsilon, num_steps, step_size, p, type_ball,
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

        adv_examples = []
        start = time.time()    

        for batch_idx, example in enumerate(tqdm(ex_loader)):
            
            #if args.load_pgd_test_samples:
            #    x, y, x_pgd = eval_batch
            #else:
            global global_step 
            global_step = global_step + 1 

            data, target = example
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
               x_adv = _pgd_whitebox(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'pgdnucl':
               PGDnuclattacker = PGD(nn.CrossEntropyLoss(), eps=epsilon, 
                 alpha=step_size,
                 type_projection='nuc',
                 iters=20,
                 device= device, 
                 p=2, normalizer=normalize)
               x_adv = PGDnuclattacker.attack(model, data, target)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            elif attack == 'FW':
               print('Computing vanilla FW for batch:')
               # Forward pass the data through the model
               output = model(normalize(x))
               init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

               # If the initial prediction is wrong, dont bother attacking, just move on
               if init_pred.item() != y.item():
                  continue
               FW_van = FW_vanilla_batch(model, type_ball=type_ball,
                                 p=p,
                                 mask=mask,
                                 radius=epsilon,
                                 eps=2e-1,
                                 T_max=num_steps,
                                 step_size=step_size,
                                 channel_subsampling=subsampling,
                                 size_groups=size_groups,
                                 group_subsampling=subsampling,
                                 sum_dir=None,
                                 normalizer = normalize,
                                 rand=bool(args.random_start),
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
            outputs_pgd = model(normalize(x_adv))
            outputs = model(normalize(x))
            loss = nn.CrossEntropyLoss()(outputs_pgd, y)

            test_loss += loss.item()
            _, predicted_pgd = outputs_pgd.max(1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            correct_pgd += predicted_pgd.eq(target).sum().item()
            
            adv_perturb = (x - x_adv)
            '''
            if batch_idx < 1000:
               tresh_norm = 0.20
            elif batch_idx < 2000:
               tresh_norm = 0.25
            elif batch_idx < 3000:
               tresh_norm = 0.3
            elif batch_idx < 4000:
               tresh_norm = 0.35
            elif batch_idx < 5000:
               tresh_norm = 0.40 
            '''
            tresh_norm = 1.00
            if predicted_pgd.item() == target.item():       
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 7):
                    perturbm = adv_perturb.squeeze()
                    adv_ex = x_adv.squeeze().detach().cpu().numpy()
                    adv_examples.append( (target.item(), predicted_pgd.item(), adv_ex, torch.norm(perturbm, 'fro'),
                             torch.norm(perturbm, inf), torch.norm(perturbm, 'nuc')) )
                elif (epsilon == 0):
                    break
            else:
                # Save some adv examples for visualization later
                perturbm = adv_perturb.squeeze()
                print("psilon: ", epsilon, "perturb norm: ", torch.norm(perturbm, p = float("inf")))
                if len(adv_examples) < 7 and torch.norm(perturbm, p = float("inf")) < tresh_norm:                     
                   print('**********************************************')                 
                   print("this is me:", epsilon)
                   print('**********************************************')
                   adv_ex = x_adv.squeeze().detach().cpu().numpy()
                   adv_examples.append( (target.item(), predicted_pgd.item(), adv_ex, torch.norm(perturbm, p='fro'),
                             torch.norm(perturbm, p = float("inf")), torch.norm(perturbm[0, :, :], p='nuc')) )
                elif len(adv_examples) >= 7:
                   break  
        return adv_examples

def sample_test(model, attack, epsilon, num_steps, step_size, p, type_ball,
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

        adv_examples = []
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
            output = model(normalize(x))
            init_pred = output.max(1, keepdim=True)[1]
            if y.item() != init_pred.item():
               continue
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
            x_advpgd = _pgd_whitebox(model, x, y, 
                                     epsilon=8/255.,
                                     step_size=2.0 / 255,
                                     num_steps=num_steps, sum_dir=summarynameindex)#(x, preds_clean, eps=args.eps_eval) 
            PGDnuclattacker = PGD(nn.CrossEntropyLoss(), eps=epsilon, 
                 alpha=step_size,
                 type_projection='nuc',
                 iters=20,
                 device= device, 
                 p=2, normalizer=normalize)
            x_advpgdnucl = PGDnuclattacker.attack(model, data, target)
            
            x_advfgsm = fgsm(model, x, y, epsilon=8/255.)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
            
            if attack == 'pgdl2':
               x_advpgdnucl = _pgd_whitebox(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            elif attack == 'L2' or attack == 'Linf':
               x_adv = attacker.attack(model, x, y)      
               x_adv.to(device)         
            elif attack == 'mean':
               x_adv = attack_mean(model, x, y, epsilon=epsilon)
            output = model(normalize(x))
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            FW_van = FW_vanilla_batch(model, type_ball=type_ball,
                                 p=p,
                                 mask=mask,
                                 radius=epsilon,
                                 eps=2e-1,
                                 T_max=num_steps,
                                 step_size=step_size,
                                 channel_subsampling=subsampling,
                                 size_groups=size_groups,
                                 group_subsampling=subsampling,
                                 sum_dir=None,
                                 normalizer = normalize,
                                 rand=bool(args.random_start),
                                 loss_type=args.loss_type,
                                 device=device)
            print("DEviiiiiiiiiiiiice", device)
            x_advFW = FW_van.attack(x, y) 
            x_advFW.to(device) 
            
            #x_adv = normalize(x_adv).to(device)   
            #x = normalize(x).to(device)          
            outputs_FW = model(normalize(x_advFW))
            outputs_fgsm = model(normalize(x_advfgsm))
            outputs_pgd = model(normalize(x_advpgd))
            outputs_pgdnucl = model(normalize(x_advpgdnucl))
            outputs = model(normalize(x))

            _, predicted_FW = outputs_FW.max(1)
            _, predicted_fgsm = outputs_fgsm.max(1)
            _, predicted_pgd = outputs_pgd.max(1)
            _, predicted_pgdnucl = outputs_pgdnucl.max(1)
            

            tresh_norm = 0.40 
            print('Target: ', target.item(), 'FW:', predicted_FW.data, 'fgsm: ', predicted_fgsm, 'PGD: ', predicted_pgd, 'PGDnucl: ', predicted_pgdnucl) 
            if predicted_FW.item() != target.item() and predicted_fgsm.item() != target.item() and predicted_pgd.item() != target.item() and predicted_pgdnucl.item() != target.item():
                # Save some adv examples for visualization later
                if len(adv_examples) < picnums:                     
                   print('**********************************************')                 
                   print("this is me:", epsilon)
                   print('**********************************************')
                   #adv_ex = x.squeeze().detach().cpu().numpy()
                   adv_examples.append( (x.detach().cpu(), target.detach().cpu())) 
                elif len(adv_examples) >= picnums:
                   break  
        return adv_examples
        '''
            eval_x_adv_linf += torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()
            eval_x_adv_l0 += torch.norm(adv_perturb, p=0, dim = 1).sum().item()
            eval_x_adv_l2 += torch.norm(adv_perturb, p=2, dim = 1).sum().item()           
            eval_x_adv_lnuc += torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()     
            print("mean nuc norm", torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim = (1,2)).mean().item())

            print(batch_idx, len(test_loader), 'Loss: %.3f| Clean Acc: %.3f%%  | Acc: %.3f%% Successs: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, correct_pgd, total))

            print(batch_idx, len(test_loader), 'Linf norm %.3f | L2 norm %.3f | L0 norm %.3f | Lnuc norm %.3f | (/%d)' 
            % (eval_x_adv_linf/total, eval_x_adv_l2/total, eval_x_adv_l0/total, eval_x_adv_lnuc/total, total))
        '''
            
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
        log_dir = attack + 'iter' + str(num_steps) + 'eps' + str(epsilon) + args.model + args.model_path.split('/')[1].split('.')[0]
        if not os.path.isdir('logger/log_' + attack + '_' + args.model + '/'):
           os.mkdir('logger/log_' + attack + '_' + args.model + '/')
        if not os.path.isdir('summary/sum_' + attack + '_' + args.model + '/'):
           os.mkdir('summary/sum_' + attack + '_' + args.model + '/')

        logname = ('logger/log_' + attack + '_' + args.model + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_' + attack + '_' + args.model + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname
def plot_images(epsilons, examples):
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(11, 5))
    gs1 = gridspec.GridSpec(4, picnums)
    gs1.update(wspace=0.085, hspace=0.05) # set the spacing between axes. 
    print("length of examples are: ", )
    for i in range(len(epsilons)):
       print("i: ", i, "epsilon", epsilons)
       print("length of examples are: ", len(examples), "len0: ", len(examples[i]))
       for j in range(len(examples[i])):
        #cnt += 1
        ax = plt.subplot(gs1[cnt])
        #plt.subplot(len(epsilons),len(examples[0]),gs1[cnt])
        plt.xticks([], [])
        plt.yticks([], [])
        #if j == 0:
        #    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=20)
        orig,adv,ex, distfro, distinf, distnuc = examples[i][j]
        #plt.title("{} -> {}\n {:0.2f}, {:0.2f}, {:0.2f}".format(orig, adv, distfro, distinf, distnuc))
        plt.imshow(ex.transpose(1,2,0))
        rect = patches.Rectangle((0,27),14,4,linewidth=0.01,edgecolor='none',facecolor='gray')
        ax.add_patch(rect)
        plt.text(7, 29, "{}".format(classes[adv]), size=15, ha='center', va='center', color='red')
        #, bbox=dict(facecolor='gray', edgecolor='red', pad=2)
        cnt += 1
    plt.tight_layout(h_pad=0)
    plt.savefig('cifar10_resnet50.png', bbox_inches='tight',pad_inches = 0)
    plt.show()

def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)  
            orig,adv,ex, distfro, distinf, distnuc = xlist[h][w]              
            ax.imshow(ex.squeeze().transpose(1,2,0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main(eps):
        
        model = load_model()
        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)
        step_size =  0.3
        examples = []        
        pgd_params = {'attack': 'FW', 'epsilon': 1, 'num_steps': 20, 'step_size': step_size,
           'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': bool(args.subsampling), 'size_groups': args.size_groups}
        pgd_params['summaryname'], pgd_params['logname'] = None, None #mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        ex_loader = sample_test(model, **pgd_params)
        pgd_params['epsilon'] = 3
        params =[]
        params.append(pgd_params)
        pgd_params['attack'], pgd_params['step_size'] = 'pgdnucl', 2./255
        params.append(pgd_params)
        pgd_params['attack'], pgd_params['epsilon'] = 'pgd', 8./255
        params.append(pgd_params)
        pgd_params['attack'] = 'fgsm'
        params.append(pgd_params)
        #epsilons = [1, 3, 5]
        #for eps in epsilons:
        for par in params:
           #pgd_params['epsilon'] = eps
           print("pgd parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['epsilon'], pgd_params['num_steps']))
           #ex = run_eval(model, **pgd_params)           
           ex = run_eval(model, ex_loader, **par)
           print('******************')
           examples.append(ex)
        plot_images(epsilons, examples)
        #show_image_row(examples)
        sys.exit()
       
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == '__main__':

        picnums = 1
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

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        sampler = torch.utils.data.sampler.RandomSampler( testset, replacement=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, sampler=sampler, num_workers=2)

        #args.model = 'ResNet18'
        #args.model_path = 'checkpoint/resnet18_epoch_347_acc_94.77.pth'
        #main(eps=1)

        

        args.model = 'ResNet50'
        args.model_path = 'checkpoint/cifar_linf_8.pt'
        main(eps=1)
        '''
        
        args.model = 'ResNet18'
        args.model_path = 'Madry-PGD-Linf/ResNet18/checkpoint.pt.best'
        main(eps=1)

        args.model = 'WideResNet'
        args.model_path = 'Madry-PGD-Linf/WideResNet/checkpoint.pt.best'
        main(eps=1)

        args.model = 'ResNet50'
        args.model_path = 'Madry-PGD-Linf/ResNet50/cifar_linf_8.pt'
        main(eps=3)
        
        
        args.model = 'ResNet18'
        args.model_path = 'Madry-PGD-Linf/ResNet18/checkpoint.pt.best'
        main(eps=3)

        args.model = 'WideResNet'
        args.model_path = 'Madry-PGD-Linf/WideResNet/checkpoint.pt.best'
        main(eps=3)

        args.model = 'ResNet50'
        args.model_path = 'Madry-PGD-Linf/ResNet50/cifar_linf_8.pt'
        main(eps=5)
        
        
        args.model = 'ResNet18'
        args.model_path = 'Madry-PGD-Linf/ResNet18/checkpoint.pt.best'
        main(eps=5)

        args.model = 'WideResNet'
        args.model_path = 'Madry-PGD-Linf/WideResNet/checkpoint.pt.best'
        main(eps=5)
        

        
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
