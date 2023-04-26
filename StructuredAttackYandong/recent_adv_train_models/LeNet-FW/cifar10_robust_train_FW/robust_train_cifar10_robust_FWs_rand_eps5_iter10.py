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

sys.path.insert(0, '../../../../attack_cifar10/')
import models
from models import *
#from fast_adv.models.mnist.small_cnn import *
from fast_adv.attacks import DDN, CarliniWagnerL2, PGD
from FW_vanilla_batch import FW_vanilla_batch
from fast_adv.utils import requires_grad_
#from fast_adv.models.net_mnist import *

import pdb
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default=0.1)')
parser.add_argument('--decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--step_lr', '-LR', type=int, default=50)
parser.add_argument('--epoch', type=int, default=200, help='total epochs')
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
parser.add_argument('--save-dir',
                    default=None,
                    help='save directory to save the model')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_mnist_smallcnn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--num_ckpt_steps', type=int, default=10, help='save checkpoint steps (default: 10)')
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


class AttackFW(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_inf bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config):
        super(AttackFW, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.FW_van = FW_vanilla_batch(self.model, type_ball=config['type_ball'],
                                 p=config['p'],
                                 mask=config['mask'],
                                 radius=config['epsilon'],
                                 eps=1e-10,
                                 rand=True,
                                 T_max=config['num_steps'],
                                 step_size=0.1,
                                 channel_subsampling=config['subsampling'],
                                 size_groups=config['size_groups'],
                                 group_subsampling=config['subsampling'],
                                 normalizer = normalize,
                                 loss_type=args.loss_type,
                                 device=device)
        

    def forward(self, inputs, targets=None, make_adv=False):
       x = inputs
       if make_adv:
           self.FW_van.model = self.model
           x = self.FW_van.attack(x, targets)
            
       return self.model(x), x




def train(save_dir):
    
    net.train()
    globe_train = True
    train_loss = 0
    correct = 0
    total = 0
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma = 0.1)    
    for epoch in range(args.epoch):
        print('\nEpoch: %d' % epoch)
        correct = 0
        total = 0
        train_loss = 0.0
        #eval_test(net, device, test_loader)
        #adjust_lr(optimizer, epoch)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
           inputs, targets = inputs.to(device), targets.to(device)
           net.eval()
           #adv = dct_pgd_whitebox_1(net, inputs, targets, num_steps=40) #dct_pgd_whitebox(net, net, inputs, targets)
           #dct_attack_batch(model_src, model_tgt, inputs, targets, 10) #net(inputs)   
           
           #adv = attacker(inputs, target=targets, **attack_kwargs) 
           #net.train()
           #outputs = net(adv)
           #outputs, final_inp = net(inputs, target=targets, make_adv=1,
           #                       **attack_kwargs)
           outputs, final_inp = net(inputs, targets, make_adv=True)
           # outputs = net(inputs.cuda())
           net.train()
           loss = criterion(outputs, targets)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           train_loss += loss.item()
           _, pred_idx = torch.max(outputs.data, 1)

           total += targets.size(0)
           correct += pred_idx.eq(targets.data).cpu().sum().float()

           # Bar visualization
           print(batch_idx, len(train_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if epoch % args.num_ckpt_steps == 0:
            # save_checkpoint(net, 0.0, epoch, 'net_robust_adversarial_FW/')
            save_checkpoint(net, 0.0, epoch, save_dir + '/')
        schedule.step()


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


        with open(logname, 'w') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow(['batch_idx', 'Test Loss', 'Clean Test Acc', 'Test Acc', 'Success rate', 
                              'Linf', 'L2 norm', 'L0 norm', 'Lnuc norm', 'Correct', 'Total'])
        
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
               x_adv = _pgd_whitebox(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            elif attack == 'FW':
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
            outputs_pgd = model(x_adv)
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs_pgd, y)

            test_loss += loss.item()
            _, predicted_pgd = outputs_pgd.max(1)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            correct_pgd += predicted_pgd.eq(target).sum().item()
            
            adv_perturb = (x - x_adv).view(x.size(0), -1)
            eval_x_adv_linf += torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()
            eval_x_adv_l0 += torch.norm(adv_perturb, p=0, dim = 1).sum().item()
            eval_x_adv_l2 += torch.norm(adv_perturb, p=2, dim = 1).sum().item()           
            eval_x_adv_lnuc += torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()    

            print(batch_idx, len(test_loader), 'Loss: %.3f| Clean Acc: %.3f%%  | Acc: %.3f%% Successs: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, correct_pgd, total))

            print(batch_idx, len(test_loader), 'Linf norm %.3f | L2 norm %.3f | L0 norm %.3f | Lnuc norm %.3f | (/%d)' 
            % (eval_x_adv_linf/total, eval_x_adv_l2/total, eval_x_adv_l0/total, eval_x_adv_lnuc/total, total))

            with open(logname, 'a') as logfile:
               logwriter = csv.writer(logfile, delimiter=',')
               logwriter.writerow([batch_idx, test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_pgd/total, 100.*(total - correct_pgd)/total,
                                  eval_x_adv_linf/total, eval_x_adv_l2/total, eval_x_adv_l0/total, eval_x_adv_lnuc/total, correct_pgd, total])
            
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

def save_checkpoint(model, acc, epoch, filename):
    print('=====> Saving checkpoint...')
    state = {
        'model': model.model,
        'acc': acc,
        'epoch': epoch
    }
    torch.save(state, filename + 'net_epoch' + str(epoch) + '.ckpt')

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
        if not os.path.isdir('logger/log_rand_cross_' + attack + '_' + args.model + '/'):
           os.mkdir('logger/log_rand_cross_' + attack + '_' + args.model + '/')
        if not os.path.isdir('summary/sum_rand_cross_' + attack + '_' + args.model + '/'):
           os.mkdir('summary/sum_rand_cross_' + attack + '_' + args.model + '/')

        logname = ('logger/log_rand_cross_' + attack + '_' + args.model + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_rand_cross_' + attack + '_' + args.model + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname

def main(eps):
        
        model = load_model()
        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)
        step_size =  1
        pgd_params = {'attack': 'FW', 'epsilon': eps, 'num_steps': 10, 'step_size': step_size,
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
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
        std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
        unnormalize = lambda x: x
        normalize = lambda x: x

        trainset = torchvision.datasets.CIFAR10(root='../../../../attack_cifar10/data/', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='../../../../attack_cifar10/data/', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

        attack_kwargs = {
            'constraint': "inf",
            'eps': 0.3,
            'step_size': 0.01,
            'iterations': 40,
            'random_start': 0,
            'random_restarts': 0,
            'use_best': 1,
            'should_normalize': 0
        }

        config = {
                'epsilon': 0.3,
                'num_steps': 40,
                'step_size': 0.01,
                'random_start': False,
                'loss_func': 'xent'
               } 
        pgd_params = {'attack': 'FW', 'epsilon': args.epsilon, 'num_steps': 10, 'step_size': 3,
        'p': 2, 'type_ball': 'nuclear', 'mask': 'True', 'subsampling': False, 'size_groups': False, 'device': device}

        config.update(pgd_params)
        
        args.model = 'ResNet18'
        # args.model_path = '../../../../attack_cifar10/Madry-PGD-Linf/ResNet18/checkpoint.pt.best'

        # args.model_path = 'net_robust_adversarial_FW_eps5_iter10/net_epoch170.ckpt' # 'net_robust_adversarial_eps3_FW/net_epoch20.ckpt'
        #args.model_path = 'net_robust_adversarial_FW/net_epoch30.ckpt'
        net_src = load_model()
        net_src = net_src.to(device)
        eval_test(net_src, device, test_loader)
        net = AttackFW(net_src, config)
        if not os.path.isdir(args.save_dir):
           os.mkdir(args.save_dir)

        train(args.save_dir)
        # main(1.0)
        

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
