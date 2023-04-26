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
from utils import CAM

sys.path.append('../../')
from models.net_mnist import *
from models.small_cnn import *
#from fast_adv.models.mnist.small_cnn import *
from fast_adv.attacks import DDN, CarliniWagnerL2, PGD
from FW_vanilla_batch import FW_vanilla_batch
from StrAttack import LADMMSTL2
from fast_adv.utils import requires_grad_

from saliency_adversarial_example import saliency_map_mat, compute_salien_map
#from fast_adv.models.net_mnist import *
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from tensorboardX.writer import SummaryWriter

from PIL import Image
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt
import copy

import pdb


'''
python3 pgd_attack_mnist_robust.py --test-batch-size 512 --attack pgd --model-path mnist_virginia/mnist_Net/LeNet-inf2/mnist/f3808301-0c8c-4702-a4dd-7c830e4afb07/checkpoint.pt.best  --model Net --no-cuda --num-steps 10
'''



parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--model', default='SmallCNN', choices=['SmallCNN','Net', 'ModelNet'],
                    help='models to train (default small_cnn')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
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


RESULTS_FOLDER = './results'


def normalize_array(arr):
    """
        Sets array values to span 0-1
    """
    arr = arr - arr.min()
    arr /= arr.max()
    return arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def plt_show_fig(image):
        pdb.set_trace()
        plt.imshow(np.asarray(image))
        plt.imshow(saliency[0], cmap=plt.cm.hot)
        plt.axis('off')
        plt.show()

def save_vanilla_gradient(image, raw_saliency_map, filename):
        """ Implements gradient visualization with vanilla backprop. """

        saliency_map = normalize_array(raw_saliency_map)

        # Process data image for rendering
        image = normalize_array(image)
        image = format_np_output(image)
        image = Image.fromarray(image)
        save_gradient_images(image, saliency_map[0], filename)

        print("Saved Vanilla Gradient image to results folder")




def save_gradient_images(org_img, saliency_map, file_name):
    """
        Saves saliency map and overlay on the original image

    Args:
        org_img (PIL img): Original image
        saliency_map (numpy arr): Saliency map (grayscale) 0-255
        file_name (str): File name of the exported image
    """

    # Grayscale saliency map
    heatmap, heatmap_on_image = apply_colormap_on_image(
        org_img, saliency_map, 'RdBu')

    # Save original
    path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_base.png')
    save_image(org_img, path_to_file)

    # Save heatmap on image
    path_to_file = os.path.join(
        RESULTS_FOLDER, file_name+'_saliency_overlay.png')
    save_image(heatmap_on_image, path_to_file)

    # Save colored heatmap
    path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_saliency.png')
    save_image(heatmap, path_to_file)

    # Save grayscale heatmap
    # path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_grayscale.png')
    # save_image(saliency_map, path_to_file)


def apply_colormap_on_image(org_im, saliency_map, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        saliency_map (numpy arr): Saliency map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(saliency_map)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


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

def compute_IS_numpy(delta, sali_map, perc=30):

        del_norm = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=(1))
        ASM = sali_map # sali_map.norm(p=2, dim=(1,2,3))
        nu = np.percentile(ASM, perc)
        B = np.greater_equal(ASM, nu).astype(np.float)
        C = np.multiply(B, delta)
        IS = np.linalg.norm(C.reshape(delta.shape[0], -1), axis=(1)) / del_norm

        return IS.mean()

def compute_IS(delta, sali_map, perc=30):

        del_norm = delta.norm(p=2, dim=(1,2,3))
        ASM = sali_map # sali_map.norm(p=2, dim=(1,2,3))
        nu = np.percentile(ASM.numpy(), perc)
        B = ASM.ge(nu).float()
        C = torch.mul(B, delta)
        IS = C.norm(p=2, dim=(1,2,3)) / del_norm

        return IS.mean()
        
def run_eval(model, attack, epsilon, num_steps, step_size, p, type_ball,
             mask, subsampling, size_groups, logname, summaryname, with_attack=True, **args):
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
                              'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc', 'FW gap'])
        model.eval().to(device)
        network = CAM(model).to(device)

        start = time.time()  

        nb_classes = 10
        Identity = np.eye(nb_classes) 
       
        X, X_ADV, SAL_MAT, Y, Y_ADV = [], [], [], [], []
        detach = lambda x : x.cpu().detach().numpy() 
        cat = lambda x : np.concatenate(x, axis=0)

            
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
               from saliency_adversarial_example.craft import jsma
               x_adv = fgsm(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
               outputs_pgd = model(x_adv)
               _, pred_tgt = outputs_pgd.max(1)
               ind  = pred_tgt != y
               delta = x_adv - x 
               sali_map = compute_salien_map(model, x, pred_tgt)
               # IS = compute_IS(delta[ind], sali_map, perc=30)
               X.append(detach(x)), X_ADV.append(detach(x_adv)), SAL_MAT.append(detach(sali_map)), Y.append(detach(y)), Y_ADV.append(detach(pred_tgt))
               continue
               save_vanilla_gradient(x[0].detach().cpu().numpy(), sali_map[0].detach().cpu().numpy(), 'sali_map')
               # crafted = jsma(model, x[0:1], y[0:1], 0.1)
               pdb.set_trace()
               output = model(x)
               sail_mat = compute_salien_map(model, x, y)
               from saliency_adversarial_example.craft import fgsm_new
               pdb.set_trace()
            elif attack == 'pgdl2':
               x_adv = _pgd_whitebox(model, x, y)#attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
               x_adv.to(device)
            elif attack == 'cw':
               x = x.detach().to(device)
               requires_grad_(model, False)               
               x_adv = cwattacker.attack(model, x, labels=y, targeted=False)#(x, preds_clean)
               x_adv.to(device)
            elif attack == 'FW':
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
            elif attack == 'StrAttack':
                x = x[:]
                target = target[:]
                one_hot_targets = Identity[target.dsave_vanilla_gradientata]
                x_n = x.detach().cpu().numpy()
                Strattack = LADMMSTL2(model, x.size(0), x.size(2), x.size(1), max_iterations=args['maxiter'],
                               confidence=args['conf'], binary_search_steps=args['iteration_steps'], ro=args['ro'],
                               abort_early=args['abort_early'],retrain=args['retrain']) 
                x_adv = Strattack.attack(x_n, one_hot_targets)
                x_adv = torch.from_numpy(x_adv).float()
                model = model.to(device)
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

        np.savez(f'{attack}_eps_{epsilon}.npz', X=cat(X), X_ADV=cat(X_ADV), SAL_MAT=cat(SAL_MAT), Y=cat(Y), Y_ADV=cat(Y_ADV))
        load_file = np.load(f'{attack}_eps_{epsilon}.npz')

        keys = ['X', 'X_ADV', 'SAL_MAT', 'Y', 'Y_ADV']

        data_dict = {key:load_file[key] for key in keys}

        ind = data_dict['Y'] == data_dict['Y_ADV']
        delta = data_dict['X_ADV'] - data_dict['X']
        pdb.set_trace()
        compute_IS_numpy(delta[~ind], data_dict['SAL_MAT'][~ind])

        pdb.set_trace()
        

            
        '''    
        attack_time = time.time() - start  
        with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile) 
           logwriter.writerow(["Time: ", attack_time]) 
        '''
 

def save_checkpoint(state: OrderedDict, filename):
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    torch.save(state, filename)

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
        if not os.path.isdir('logger/log_rand_' + attack + '/'):
           os.mkdir('logger/log_rand_' + attack + '/')
        if not os.path.isdir('summary/sum_rand_' + attack + '/'):
           os.mkdir('summary/sum_rand_' + attack + '/')

        logname = ('logger/log_rand_' + attack + '/' + log_dir + '.csv')
        summaryname = ('summary/sum_rand_' + attack + '/' + log_dir + '.csv')

        sh.rm('-rf', logname)
        sh.rm('-rf', summaryname)
        return summaryname, logname

def main(eps):
        
        model = load_model()
        #eval_adv_test_whitebox(model, device, test_loader)
        #eval_adv_test_Carlini(model, device, test_loader)
        #eval_test(model, device, test_loader)

        """
        pgd_params = {'attack': 'FW', 'epsilon': eps, 'num_steps': 10, 'step_size': 3,
        'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': False, 'size_groups': False}
        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("%s parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['attack'], pgd_params['epsilon'], pgd_params['num_steps']))
        run_eval(model, **pgd_params)
        print('******************')
        

        pgd_params['num_steps'] = 20
        """

        pgd_params = {'attack': 'fgsm', 'epsilon': eps, 'num_steps': 10, 'step_size': 3,
        'p': args.p, 'type_ball': 'nuclear', 'mask': args.mask, 'subsampling': False, 'size_groups': False, 'batch_size': 9, 'maxiter': 1000,
        'conf': 0, 'iteration_steps': 9, 'ro': 15, 'abort_early': True, 'retrain': False}

        pgd_params['summaryname'], pgd_params['logname'] = mk_log_file(pgd_params['attack'], pgd_params['num_steps'], pgd_params['epsilon'])
        print("%s parameters... epsilon: %.3f%% number of steps: %d"% (pgd_params['attack'], pgd_params['epsilon'], pgd_params['num_steps']))
        run_eval(model, **pgd_params)
        print('******************')
        

if __name__ == '__main__':

        # settings
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # set up data loader
        transform_test = transforms.Compose([transforms.ToTensor(),])
        testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        unnormalize = lambda x: x
        normalize = lambda x: x
        
        args.model = 'SmallCNN'
        args.model_path = '../../mnist_virginia/mnist/LeNet-inf2/mnist/f383270a-4119-4782-922a-a7a80444c226/checkpoint.pt.best'
        main(eps=0.1)
        
        args.model = 'Net'
        args.model_path = '../../mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        main(eps=1)

        args.model = 'SmallCNN'
        args.model_path = '../../mnist_virginia/mnist/LeNet-inf2/mnist/f383270a-4119-4782-922a-a7a80444c226/checkpoint.pt.best'
        main(eps=3)
        
        args.model = 'Net'
        args.model_path = '../../mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        main(eps=3)

        args.model = 'SmallCNN'
        args.model_path = '../../mnist_virginia/mnist/LeNet-inf2/mnist/f383270a-4119-4782-922a-a7a80444c226/checkpoint.pt.best'
        main(eps=5)

        args.model = 'Net'
        args.model_path = '../../mnist_virginia/mnist_Net/LeNet/mnist/36afc8e6-f18b-49d5-a5e8-b18cd7ce1be5/checkpoint.pt.best'
        main(eps=5)


    
