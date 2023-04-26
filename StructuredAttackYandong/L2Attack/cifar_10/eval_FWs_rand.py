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

sys.path.insert(0, '../../../')
#from fast_adv.models.mnist.small_cnn import *
from fast_adv.attacks import DDN, CarliniWagnerL2, FW_vanilla_batch, PGD
from fast_adv.utils import requires_grad_
#from fast_adv.models.net_mnist import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from tensorboardX.writer import SummaryWriter

import pdb

'''
python3 pgd_attack_mnist_robust.py --test-batch-size 512 --attack pgd --model-path mnist_virginia/mnist_Net/LeNet-inf2/mnist/f3808301-0c8c-4702-a4dd-7c830e4afb07/checkpoint.pt.best  --model Net --no-cuda --num-steps 10
'''



def attack_foolbox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel, criterion=attack_criteria)
    x_adv = attack(X.cpu().detach().numpy(), y.numpy())

    return torch.from_numpy(x_adv)



def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size, 
                  sum_dir = None):

    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        loss_clean, loss_Adv, loss_incr, clean_acc, preds_acc, eval_x_adv_linf, eval_x_adv_l0, eval_x_adv_l2 = attack_log(model, X, X_pgd, y, i, sum_dir)
    return X_pgd.detach() 

def _pgd_whitebox1(model,
                  inputs,
                  y,
                  epsilon,
                  num_steps,
                  step_size, 
                  sum_dir = None):


        x = inputs.detach()
        if args.random:
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, 0, 1)
    
        return x

def fgsm(model,
         X,
         y,
         epsilon):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    # signed gradien
    eta = epsilon * X.grad.detach().sign()
    # Add perturbation to original example to obtain adversarial example
    x_adv = X.detach() + eta
    x_adv = torch.clamp(x_adv, 0, 1)
    
    
    return x_adv.detach()


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
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
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        _, preds = torch.max(logits, 1)
        return loss, preds

def clip(x, cmin, cmax):
        return torch.min(torch.max(x, cmin), cmax)

def attack_mean(model, x, y, epsilon):
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
            clean_acc = 100.*preds_clean.eq(y).sum().item()/len(y)

            loss_adv, preds_adv = get_loss_and_preds(model, x_adv, y)
            preds_acc = 100.*preds_adv.eq(y).sum().item()/len(y)
            
            loss_incr = loss_adv - loss_clean

            eval_x_adv_linf = torch.norm(x - x_adv, p = float("inf"), dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_l0 = torch.norm(x - x_adv, p=0, dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_l2 = torch.norm(x - x_adv, p=2, dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_lnuc = torch.norm(x - x_adv, p='nuc', dim = (2,3)).sum().item()/len(y)
            
          
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
            
            


def run_eval(model, test_loader, attack, epsilon, num_steps, step_size, p, type_ball,
             mask, subsampling, size_groups, logname, summaryname, with_attack=True, **kwargs):
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
        adversary = []
         
        test_loss = 0
        total = 0
        correct = 0
        correct_pgd = 0
        eval_x_adv_linf = 0
        eval_x_adv_l0 = 0
        eval_x_adv_l2 = 0          
        eval_x_adv_lnuc = 0

        with open(logname, 'w') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow(['batch_idx', 'Test_Loss', 'Clean_Test_Acc', 'Test_Acc', 'Success_rate', 
                              'Linf', 'L2_norm', 'L0_norm', 'Lnuc_norm', 'Correct', 'Total'])

        with open(summaryname, 'w') as logfile:
           summarywriter = csv.writer(logfile, delimiter=',')
           summarywriter.writerow(['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf',
                              'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc', 'FW gap'])
        device = kwargs.get('device')
        model.eval().to(device)
        start = time.time()    

        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            
            #if args.load_pgd_test_samples:
            #    x, y, x_pgd = eval_batch
            #else:
            data, target = data.to(device), target.to(device)
            # pgd attack
            x, y = Variable(data, requires_grad=True), Variable(target)
            x = x.to(device)
            y = y.to(device)

            loss_clean, preds_clean = get_loss_and_preds(model, x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((torch.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if attack == 'pgd':
               x_adv = _pgd_whitebox(model, x, y, 
                                     epsilon=epsilon,
                                     step_size=step_size,
                                     num_steps=num_steps, sum_dir=summaryname)#(x, preds_clean, eps=args.eps_eval) 
               #x_adv = attack_foolbox(model, x, y)
               x_adv.to(device)
            elif attack == 'fgsm':
               x_adv = fgsm(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'pgdl2':
               x_adv = _pgd_whitebox(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            if attack == 'FW':
               print('Computing vanilla FW for batch')
               FW_van = FW_vanilla_batch(model, type_ball=type_ball,
                                 p=p,
                                 mask=mask,
                                 radius=epsilon,
                                 eps=1e-10,
                                 rand=True,
                                 T_max=num_steps,
                                 step_size=3.0,
                                 channel_subsampling=subsampling,
                                 size_groups=size_groups,
                                 group_subsampling=subsampling,
                                 sum_dir=summaryname, 
                                 device=device)
               x_adv = FW_van.attack(x, y) 
               x_adv.to(device) 
            elif attack == 'L2' or attack == 'Linf':
               x_adv = attacker.attack(model, x, y)      
               x_adv.to(device)                 
            elif attack == 'mean':
               x_adv = attack_mean(model, x, y, epsilon=epsilon)
                        
            outputs_pgd = model(x_adv)
            outputs = model(x)
            adversary.append(x_adv.detach().cpu())
            loss = nn.CrossEntropyLoss()(outputs_pgd, y)

            adv_perturb = (x - x_adv).view(x.size(0), -1)
            eval_x_adv_linf += torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()
            eval_x_adv_l0 += torch.sum(torch.max(torch.abs(x - x_adv) > 1e-10, dim=1)[0], dim=(1,2)).sum().item()
            eval_x_adv_l2 += torch.norm(adv_perturb, p=2, dim = 1).sum().item()           
            eval_x_adv_lnuc += torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()    


            test_loss += loss.item()
            _, predicted_pgd = outputs_pgd.max(1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            correct_pgd += predicted_pgd.eq(target).sum().item()
            

            print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% Successs: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, correct_pgd, total))
            with open(logname, 'a') as logfile:
               logwriter = csv.writer(logfile, delimiter=',')
               logwriter.writerow([batch_idx, test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, eval_x_adv_linf/total, eval_x_adv_l2/total, eval_x_adv_l0/total, eval_x_adv_lnuc/total, correct_pgd, total])
            
        '''    
        attack_time = time.time() - start  
        with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile) 
           logwriter.writerow(["Time: ", attack_time]) 
        '''
        adversary = torch.cat(adversary, dim=0)
        return adversary
 

def load_checkpoint(state, model):
        new_state = {}
        for k in state.keys():
            newk = k.replace('model.', '')  # remove module. if model was trained using DataParallel
            newk = newk.replace('module.', '')       
            new_state[newk] = state[k]
        state = new_state
        model.load_state_dict(state)
        return model


def mk_log_file(attack, num_steps, epsilon):

        log_dir = attack + '_step_' + str(num_steps) + '_eps_' + str(epsilon)
        if not os.path.isdir('logger/'):
            os.mkdir('logger/') 
        if not os.path.isdir('summary/'):
            os.mkdir('summary/')
        if not os.path.isdir('logger/log_rand_' + attack + '/'):
           os.mkdir('logger/log_rand_' + attack + '/')
        if not os.path.isdir('summary/sum_rand_' + attack + '/'):
           os.mkdir('summary/sum_rand_' + attack + '/')

        logname = ('logger/log_rand_' + attack + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_rand_' + attack + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname

def FW_attack(model, test_loader, eps, device='cpu'):
        
        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)

        pgd_params = {'attack': 'FW', 'epsilon': eps, 'num_steps': 10, 'step_size': 3,
        'p': 2, 'type_ball': 'nuclear', 'mask': 'True', 'subsampling': False, 'size_groups': False, 'device': device}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("%s parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['attack'], pgd_params['epsilon'], pgd_params['num_steps']))
        adversary = run_eval(model, test_loader, **pgd_params)
        print('******************')
        return adversary
        

        print('******************')

        unnormalize = lambda x: x
        normalize = lambda x: x
        

        

    
