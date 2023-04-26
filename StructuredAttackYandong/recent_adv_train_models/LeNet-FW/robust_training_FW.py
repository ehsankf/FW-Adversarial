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
from FW_vanilla_batch import FW_vanilla_batch

sys.path.insert(0, '../../..')
from models.net_mnist import *
from models.small_cnn import *
#from fast_adv.models.mnist.small_cnn import *
from fast_adv.attacks import DDN, CarliniWagnerL2, PGD
from fast_adv.utils import requires_grad_
#from fast_adv.models.net_mnist import *
from utils import progress_bar

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from tensorboardX.writer import SummaryWriter
import pdb

'''
python3 pgd_attack_mnist_robust.py --test-batch-size 512 --attack pgd --model-path mnist_virginia/mnist_Net/LeNet-inf2/mnist/f3808301-0c8c-4702-a4dd-7c830e4afb07/checkpoint.pt.best  --model Net --no-cuda --num-steps 10
'''



parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', choices=['SmallCNN','Net', 'ModelNet'],
                    help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default=0.1)')
parser.add_argument('--decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--step_lr', '-LR', type=int, default=50)
parser.add_argument('--epoch', type=int, default=200, help='total epochs')
parser.add_argument('--load_pgd_test_samples', default=None, type=str)
parser.add_argument('--epsilon',  type=float, default=0.3,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.1,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default=None,
                    help='model for white-box attack evaluation')
parser.add_argument('--save-dir',
                    default=None,
                    help='save directory to save the model')
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
parser.add_argument('--num_ckpt_steps', type=int, default=10, help='save checkpoint steps (default: 10)')

#Frank-Wolf parameters
parser.add_argument('--p', default=2, type=float, 
                    help='p-wasserstein distance or for the lp ball.')
parser.add_argument('--subsampling', default='True',
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


args = parser.parse_args()

def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 1, size, size)
    z = torch.zeros(batch_size, 1, 28, 28)
    z[:, :, :size, :size] = x
    return z


def get_probs(model, x, y):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    probs = torch.index_select(torch.nn.Softmax()(output).data, 1, y)
    return torch.diag(probs)

def get_preds(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    _, preds = output.data.max(1)
    return preds

def get_logit(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    return output

def xent_loss(logit, label, target=None):
        return F.cross_entropy(logit, label, reduction='none')
def dct_pgd(model_src, model,
                  inputs,
                  y,
                  epsilon=args.epsilon,
                  num_steps=10,
                  step_size=0.1, 
                  sum_dir = None):


        x = inputs.detach()
        #if args.random:
        #    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(normalize(x.cuda())).cpu(), y) - F.cross_entropy(model_src(normalize(x.cuda())).cpu(), y)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            eta = step_size * torch.sign(grad.detach())
            #x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            #x = torch.clamp(x, 0, 1)
    
        return eta

def dct_pgd_whitebox(model_src, model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=10,
                  step_size=0.01, 
                  sum_dir = None):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    model.eval()
    model_src.eval()
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
             #loss = F.cross_entropy(model(normalize(X_pgd.cuda())), y) - F.cross_entropy(model_src(normalize(X_pgd.cuda())), y)
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd.detach()

criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
def dct_pgd_whitebox_1(model, orig_input, target,
                  epsilon=args.epsilon,
                  num_steps=10,
                  step_size=0.01, 
                  sum_dir = None):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    model.eval()
    x = orig_input
    for i in range(num_steps):        
        x = x.clone().detach().requires_grad_(True)
        output = model(x)
        losses = criterion(output, target)
        loss = torch.mean(losses) 
        grad, = torch.autograd.grad( loss, [x])
        with torch.no_grad():
           step = torch.sign(grad) * step_size
           diff = x + step - orig_input
           diff = torch.clamp(diff, -epsilon, epsilon)
           x = torch.clamp(diff + orig_input, 0, 1)
    return x



def dct_fgsm(model_src, model, 
         X,
         y,
         epsilon=args.epsilon):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X.requires_grad = True
    with torch.enable_grad():
        loss = F.cross_entropy(model(normalize(X.cuda())).cpu(), y) - F.cross_entropy(model_src(normalize(X.cuda())).cpu(), y)
        #loss = - F.cross_entropy(model_src(X.cuda()).cpu(), y)
    loss.backward()
    # signed gradient
    eta = args.epsilon * X.grad.detach().sign()
    # Add perturbation to original example to obtain adversarial example
    return eta

# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>
def dct_attack_batch(model_src, model, images_batch, labels_batch, max_iters, stride=0, epsilon=args.epsilon, order='rand', targeted=False, pixel_attack=True, log_every=100):
    batch_size = images_batch.size(0)
    image_size = images_batch.size(2)
    #print("image size: ", images_batch.size())
    # sample a random ordering for coordinates independently per batch element
    images_batch, labels_batch = images_batch.cpu(), labels_batch.cpu()
    if order == 'rand':
        indices = torch.randperm(images_batch.size(1) * images_batch.size(2) * images_batch.size(3))[:max_iters]
    else:
        indices = utils.block_order(image_size, 3)[:max_iters]
    if order == 'rand':
        expand_dims = images_batch.size(2)
    else:
        expand_dims = image_size

    model_src.eval()
    model.eval()
    n_dims = 1 * expand_dims * expand_dims
    x = torch.zeros(batch_size, n_dims)
    expanded = images_batch
    # logging tensors
    probs = torch.zeros(batch_size, max_iters)
    succs = torch.zeros(batch_size, max_iters)
    queries = torch.zeros(batch_size, max_iters)
    l2_norms = torch.zeros(batch_size, max_iters)
    linf_norms = torch.zeros(batch_size, max_iters)
    prev_probs = get_probs(model, images_batch, labels_batch)
    preds = get_preds(model, images_batch)
    preds_src = get_preds(model_src, images_batch)
    if pixel_attack:
        trans = lambda z: z.clamp(-epsilon, epsilon)
    else:
        trans = lambda z: utils.block_idct(z, block_size=image_size)
    remaining_indices = torch.arange(0, batch_size).long()
    print("batch_size: ",  batch_size)


    for k in range(max_iters):
        #dim = indices[k]
        expanded[remaining_indices] = (images_batch[remaining_indices] + trans(expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
        perturbation = trans(expand_vector(x, expand_dims))
        l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
        linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
        preds_next = get_preds(model, expanded[remaining_indices])
        preds[remaining_indices] = preds_next
        preds_next_src = get_preds(model_src, expanded[remaining_indices])
        preds_src[remaining_indices] = preds_next_src
        if targeted:
            remaining = preds.ne(labels_batch).tolist() | preds_src.ne(labels_batch).tolist()
        else:
            remaining = preds.eq(labels_batch) | preds_src.ne(labels_batch)
        # check if all images are misclassified and stop early
        if remaining.sum() == 0:
            adv = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
            probs_k = get_probs(model, adv, labels_batch)
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(batch_size, max_iters - k)
            break
        remaining_indices = torch.arange(0, batch_size)[remaining].long()
        if k > 0:
            succs[:, k-1] = ~ remaining
        diff = torch.zeros(remaining.sum(), n_dims)
        #diff = epsilon
        #diff[:, dim] = epsilon
        #diff =   epsilon * get_src_grad(model_src, expanded[remaining_indices], labels_batch[remaining_indices]).sign().view(-1, n_dims)
        diff = dct_pgd(model_src, model, expanded[remaining_indices], labels_batch[remaining_indices]).view(-1, n_dims)
        #diff =  diff.view(-1, n_dims)
        #diff /= torch.norm(diff, p=2, dim=1)
        #diff *= epsilon
        #diff = epsilon * get_tgt_fdm_grad(model, expanded[remaining_indices], labels_batch[remaining_indices], expand_vector(diff, expand_dims))
       
        diff =  diff.view(-1, n_dims)
        left_vec = x[remaining_indices] - diff
        right_vec = x[remaining_indices] + diff
        # trying negative direction
        adv = (images_batch[remaining_indices] + trans(expand_vector(left_vec, expand_dims))).clamp(0, 1)
        left_probs = get_probs(model, adv, labels_batch[remaining_indices])
        left_preds_src = get_preds(model_src, adv)
        queries_k = torch.zeros(batch_size)
        # increase query count for all images
        queries_k[remaining_indices] += 1
        if targeted:
            improved = left_probs.gt(prev_probs[remaining_indices]) & left_preds_src.eq(labels_batch[remaining_indices])
            #improved = left_preds_src.eq(labels_batch[remaining_indices])
        else:
            improved = left_probs.lt(prev_probs[remaining_indices]) & left_preds_src.eq(labels_batch[remaining_indices])
            #improved = left_preds_src.eq(labels_batch[remaining_indices])
        # only increase query count further by 1 for images that did not improve in adversarial loss
        if improved.sum() < remaining_indices.size(0):
            queries_k[remaining_indices[~improved]] += 1
        # try positive directions
        adv = (images_batch[remaining_indices] + trans(expand_vector(right_vec, expand_dims))).clamp(0, 1)
        right_probs = get_probs(model, adv, labels_batch[remaining_indices])
        right_preds_src = get_preds(model_src, adv)
        if targeted:
            right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs)) & right_preds_src.eq(labels_batch[remaining_indices])
            #right_improved = right_preds_src.eq(labels_batch[remaining_indices]) and ~improved
        else:
            right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs)) & right_preds_src.eq(labels_batch[remaining_indices])
            #right_improved = right_preds_src.eq(labels_batch[remaining_indices]) and ~improved
        probs_k = prev_probs.clone()
        # update x depending on which direction improved
        if improved.sum() > 0:
            left_indices = remaining_indices[improved]
            left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
            x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
            probs_k[left_indices] = left_probs[improved]
        if right_improved.sum() > 0:
            right_indices = remaining_indices[right_improved]
            right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
            x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
            probs_k[right_indices] = right_probs[right_improved]
        if right_improved.sum() == 0 and improved.sum() == 0:
            not_improved = ~improved
            not_indices = remaining_indices[not_improved]
            not_mask_remaining = not_improved.unsqueeze(1).repeat(1, n_dims)
            x[not_indices] = right_vec[not_mask_remaining].view(-1, n_dims)
            probs_k[not_indices] = right_probs[not_improved] 
        probs[:, k] = probs_k
        queries[:, k] = queries_k
        prev_probs = probs[:, k]
        #print("left improved: ", improved.sum(), "right improved: ", right_improved.sum())
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        
    expanded = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
    preds = get_preds(model, expanded)  
    preds_src = get_preds(model_src, expanded)  
    if targeted:
        remaining = preds.ne(labels_batch) | preds_src.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch) | preds_src.ne(labels_batch)
    succs[:, max_iters-1] = ~ remaining
    return expanded.cuda()

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


'''
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size=args.step_size, 
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
    return X_pgd.detach() 
'''


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
         epsilon=args.epsilon):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    # signed gradien
    eta = args.epsilon * X.grad.detach().sign()
    # Add perturbation to original example to obtain adversarial example
    x_adv = X.detach() + eta
    x_adv = torch.clamp(x_adv, 0, 1)
    
    
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

class AttackPGD(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_inf bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        

    def forward(self, inputs, targets, make_adv=False):
       x = inputs.detach()
       if make_adv:
        if not args.attack:
            return self.model(inputs), inputs

        
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

       return self.model(x), x


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
                                 step_size=3.0,
                                 channel_subsampling=config['subsampling'],
                                 size_groups=config['size_groups'],
                                 group_subsampling=config['subsampling'],
                                 device=device)
        

    def forward(self, inputs, targets, make_adv=False):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
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
           #outputs = net(pert_inputs.cuda())
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
            save_checkpoint(net, 0.0, epoch, save_dir)
        schedule.step()


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


def run_eval(model, attack, epsilon, num_steps, step_size, p,
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

        with open(logname, 'w') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow(['batch_idx', 'Clean Acc', 'Test Loss', 'Test Acc', 'Success rate', 'Correct', 'Total'])

        with open(summaryname, 'w') as logfile:
           summarywriter = csv.writer(logfile, delimiter=',')
           summarywriter.writerow(['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf',
                              'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc'])
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
                                 T_max=num_steps,
                                 channel_subsampling=subsampling,
                                 size_groups=size_groups,
                                 group_subsampling=subsampling,
                                 sum_dir=summaryname)

               x_adv = FW_van.attack(x, y) 
               x_adv.to(device) 
            elif attack == 'L2' or attack == 'Linf':
               x_adv = attacker.attack(model, x, y)      
               x_adv.to(device)                 
            elif attack == 'mean':
               x_adv = attack_mean(model, x, y, epsilon=epsilon)
                        
            outputs_pgd = model(x_adv)
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs_pgd, y)

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
               logwriter.writerow([batch_idx, 100.*correct/total, test_loss/(batch_idx+1), 100.*correct_pgd/total, 100.*(total - correct_pgd)/total, correct_pgd, total])
            
        '''    
        attack_time = time.time() - start  
        with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile) 
           logwriter.writerow(["Time: ", attack_time]) 
        '''
 

def save_checkpoint(model, acc, epoch, filename):
    print('=====> Saving checkpoint...')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch
    }
    torch.save(state, filename + '/net_epoch' + str(epoch) + '.ckpt')


def load_checkpoint(state, model):
        new_state = {}
        for k in state.keys():
            newk = k.replace('model.', '')  # remove module. if model was trained using DataParallel
            newk = newk.replace('module.', '')       
            new_state[newk] = state[k]
        state = new_state
        model.load_state_dict(state)
        return model


def load_model():

    if args.white_box_attack:
        #model = SmallCNN().to(device)
        model = globals()[args.model]().to(device)
        if not args.model_path:
            return model
        #model.load_state_dict()
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

        model.eval().to(device)

        
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = SmallCNN().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = SmallCNN().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)

    return model

global_step = 0

def mk_log_file(attack, num_steps, epsilon):
        log_dir = attack + 'iter' + str(num_steps) + 'eps' + str(epsilon) + args.model + args.model_path.split('/')[1].split('.')[0]
        if not os.path.isdir('logger/log_adversarialgame_' + attack + '/'):
           os.mkdir('logger/log_adversarialgame_' + attack + '/')
        if not os.path.isdir('summary/sum_adversarialgame_' + attack + '/'):
           os.mkdir('summary/sum_adversarialgame_' + attack + '/')

        logname = ('logger/log_adversarialgame_' + attack + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_adversarialgame_' + attack + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname

def main():


        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)

        model = load_model()
        model = net.to(device)

        pgd_params = {'attack': 'pgd', 'epsilon': 0.3, 'num_steps': 7, 'step_size': 0.1,
        'p': args.p, 'mask': args.mask, 'subsampling': args.subsampling, 'size_groups': args.size_groups}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("pgd parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['epsilon'], pgd_params['num_steps']))
        run_eval(model, **pgd_params)
        print('******************')

        pgd_params = {'attack': 'pgd', 'epsilon': 0.3, 'num_steps': 20, 'step_size': 0.1, 
        'p': args.p, 'mask': args.mask, 'subsampling': args.subsampling, 'size_groups': args.size_groups}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("pgd parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['epsilon'], pgd_params['num_steps']))
        run_eval(model, **pgd_params)
        print('******************')

        

if __name__ == '__main__':

        # settings
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # set up data loader
        transform_test = transforms.Compose([transforms.ToTensor(),])
        trainset = torchvision.datasets.MNIST(root='../../../../data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root='../../../../data', train=False, download=True, transform=transform_test)
        

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        unnormalize = lambda x: x
        normalize = lambda x: x

        #args.model = 'SmallCNN'
        #args.model_path = 'mnist_virginia/mnist/LeNet-inf2/mnist/f383270a-4119-4782-922a-a7a80444c226/checkpoint.pt.best'

        
        #ds = MNIST('mnist/')
        #net, _ = model_utils.make_and_restore_model(arch='SmallCNN', dataset=ds)
        #net = net.cuda()
        

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

        # args.model_path = 'net_robust_adversarial_eps3_FW/net_adver_game_epoch190.ckpt' # '../../../mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        # args.model_path = 'mnist/12552106-b5d1-4376-8af5-7d7e8896ce68/checkpoint.pt.best'
        # args.save_dir = 'net_robust_adversarial_FW/net_epoch190.ckpt'
        
        net_src = load_model()
        net_src = net_src.to(device)
        net = AttackFW(net_src, config)
        
        if not os.path.isdir(args.save_dir):
           os.mkdir(args.save_dir)

        
        train(args.save_dir)

        #main()
        
        
        #main()


    
