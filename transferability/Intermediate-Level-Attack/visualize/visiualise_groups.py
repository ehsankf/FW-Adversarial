
import json
import csv
#import required libs
import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid

from torch.autograd import Variable

import sys
import argparse
from FW_attack import FW_vanilla_batch, FW_vanilla_batch_un
from pgd import PGD

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', help='<Required> source models', required=True)
parser.add_argument('--file', default='ex4.jpg', help='<Required> image file')
args = parser.parse_args()

#url = "https://savan77.github.io/blog/images/ex4.jpg"  #tiger cat #i have uploaded 4 images to try- ex/ex2/ex3.jpg
#response = requests.get(url)

'''
Usage:

python visiualise.py --models pgd  --file pics/ILSVRC2012_val_00049144.JPEG
'''

#model = models.inception_v3(pretrained=True).cuda()#download and load pretrained inceptionv3 model
#model = models.resnet50(pretrained=True).cuda()
model = models.densenet121(pretrained=True).cuda()
#model = models.resnet18(pretrained=True).cuda()
#model = models.googlenet(pretrained=True).cuda()
model.eval();

class NegLLLoss(torch.nn.Module):
    def __init__(self, min_problem=True):
        super(NegLLLoss, self).__init__()
        assert type(min_problem) == bool
        self.min_problem = min_problem

    def forward(self, x, target):
        if self.min_problem:
            return torch.nn.NLLLoss()(x, target)
        else:
            return -torch.nn.NLLLoss()(x, target)

def show_image(x):
    
    x = x.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    figure, ax = plt.subplots(1,1, figsize=(10,3))
    ax.imshow(x, extent = [0, 1, 0, 1])
    ax.set_title('Clean Example', fontsize=20) 
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.050, 'h', size=24, ha='center', va='center', color='r')

    plt.tight_layout()
    plt.show()

def plot_histogram(x, name, picname):
    
    x = x.squeeze(0)
    ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #print(ndarr[:128, :128, 0])
    image = ndarr
    newvalues = [x for x in image.ravel() if x != 0]
    fig = plt.figure(figsize=(6, 5))
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', )
    '''
    newvalues = image[:, :, 0].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'red', alpha = 0.5)
    newvalues = image[:, :, 1].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Green', alpha = 0.5)
    newvalues = image[:, :, 2].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', alpha = 0.5)
    '''
    figname = 'IMAGES/hist/' + name + 'hist' +  picname.split('/')[1] + '.png'     
    _ = plt.xlabel('Intensity Value', fontsize=18)
    _ = plt.ylabel('Count', fontsize=18)
    #_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.margins(0,0)
    plt.tight_layout(h_pad=0) 
    plt.savefig(figname, pad_inches = 0)
    #plt.show()
    plt.close()

def plot_histogram_name(x, name, picname):
    
    x = x.squeeze(0)
    ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #print(ndarr[:128, :128, 0])
    image = ndarr
    newvalues = [x for x in image.ravel() if x != 0]
    fig = plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 18})
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', )
    '''
    newvalues = image[:, :, 0].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'red', alpha = 0.5)
    newvalues = image[:, :, 1].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Green', alpha = 0.5)
    newvalues = image[:, :, 2].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', alpha = 0.5)
    '''
    figname = 'IMAGES/hist_group/' + picname + name + '.png'     
    _ = plt.xlabel('Intensity Value', fontsize=18)
    _ = plt.ylabel('Count', fontsize=18)
    #_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.margins(0,0)
    plt.tight_layout(h_pad=0) 
    plt.savefig(figname, pad_inches = 0)
    #plt.show()
    plt.close()

def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    figure, ax = plt.subplots(1,3, figsize=(10,3))
    ax[0].imshow(x, extent = [0, 15, 0, 15])
    ax[0].set_title('Clean Example', fontsize=20)
    
    '''
    #ax[1].imshow(x_grad)
    ax[1].imshow(np.abs(x_grad))
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    '''
    
    
    im = ax[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    #im = ax[1].imshow(np.abs(x_grad), cmap='hot', interpolation='nearest')
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax1)
    #figure.colorbar(im, ax=cax)

    '''
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    #ax[1][0].colorbar()
    plt.colorbar(im, ax=ax[1])
    #figure.colorbar(im, ax=ax[1])
    ax[1].set_aspect('auto')
    '''
    
    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)
    
    ax[0].axis('off')
    ax[2].axis('off')

    #ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center", 
    #        transform=ax[0].transAxes)
    
    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center", 
         transform=ax[0].transAxes)
    
    
    #ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center", 
         transform=ax[2].transAxes)
    
    
    plt.tight_layout()    
    plt.show()



def plotting(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob, name):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    fig=plt.figure(figsize=(10, 3))

    plt.subplot(131)
    plt.imshow(x)
    plt.axis('off')
    plt.title('Origianl with Pred %d' % clean_prob, fontsize=20)

    plt.subplot(132)
    plt.imshow(x_adv)
    plt.axis('off')
    plt.title(name + ' Perturbed with Pred %d' % adv_prob)

    plt.subplot(133)
    imc = plt.imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.title('Perturbation')
    plt.colorbar()
    imc.set_clim(vmin=0, vmax=0.06)
   
    plt.tight_layout()   

    plt.savefig('image' + name + args.file + '.png')
    plt.show()

def plottingp2(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob, name):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    #fig=plt.figure(figsize=(6.5, 3))
    #fig=plt.figure(figsize=(8, 3))
    #fig=plt.figure(figsize=(12, 5))
    fig = plt.figure(tight_layout=True)
    #plt.subplot(131)
    #plt.imshow(x)
    #plt.axis('off')
    #plt.title('Origianl with Pred %d' % clean_prob)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.05, hspace=0.005)

    plt.subplot(gs1[0])
    plt.imshow(x_adv)
    plt.axis('off')
    if name == 'FW':
       plt.title( 'FWnucl' + ' Perturbed', fontsize=20)
    else:
       plt.title(name.upper() + ' Perturbed', fontsize=20)

    ax1 = plt.subplot(gs1[1])
    imc = plt.imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    plt.axis('off')
    if name == 'FW':
       plt.title('FWnucl' + ' Perturbation', fontsize=20)
    else:
       plt.title(name.upper() + ' Perturbation', fontsize=20)
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    imc.set_clim(vmin=0, vmax=0.15)

    #plt.tight_layout(pad=0.0)  
    #plt.tight_layout()
    cbar_ax = plt.gcf().add_axes([0.93, 0.22, 0.02, .55])
    plt.colorbar(imc, cax=cbar_ax)
    plt.tight_layout(h_pad=0) 
    figname = args.file.split('/')[0]  +  '/image' + name + args.file.split('/')[1] + '.png' if '/' in args.file else 'image' + name + args.file + '.png' 

    plt.savefig(figname)
    #plt.savefig('image' + name + args.file + '.png')
    plt.show()

def plotting2(x, x_adv, x_grad, epsilon, name, picname):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    #fig=plt.figure(figsize=(6.5, 3))
    #fig=plt.figure(figsize=(8, 3))
    #fig=plt.figure(figsize=(12, 5))
    fig = plt.figure(figsize=(12, 5))
    grid = ImageGrid(fig, 111,
                nrows_ncols = (1,2),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )
    #plt.subplot(131)
    #plt.imshow(x)
    #plt.axis('off')
    #plt.title('Origianl with Pred %d' % clean_prob)
    grid[0].imshow(x)
    grid[0].axis('off')    
    

    if name == 'FW':
       grid[0].set_title( 'FWnucl' + ' Perturbed', fontsize=20)
    elif name == 'pgdnucl':
       grid[0].set_title( 'PGDnucl' + ' Perturbed', fontsize=20)
    else:
       grid[0].set_title(name.upper() + ' Perturbed', fontsize=20)

    imc = grid[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    grid[1].axis('off')


    if name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)

    if name == 'FW':
       grid[1].set_title('FWnucl' + ' Perturbation', fontsize=20)
    elif name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)
    else:
       grid[1].set_title(name.upper() + ' Perturbation', fontsize=20)
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    imc.set_clim(vmin=0, vmax=0.15)
    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    plt.margins(0,0)
    cbar.ax.tick_params(labelsize=14) 
    plt.tight_layout(h_pad=0) 
    figname =  picname.split('/')[0] + '/generates' + '/image' + name + picname.split('/')[1] + '.png' if '/' in picname else 'image' + name + picname + '.png' 
    print("figname: ", figname)
    #plt.show()
    plt.savefig(figname, bbox_inches='tight',pad_inches = 0)
    #plt.savefig('image' + name + args.file + '.png')
    #plt.show()
    plt.close()

def plotting_name(x, x_adv, x_grad, epsilon, name, picname):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    #fig=plt.figure(figsize=(6.5, 3))
    #fig=plt.figure(figsize=(8, 3))
    #fig=plt.figure(figsize=(12, 5))
    fig = plt.figure(figsize=(12, 5))
    grid = ImageGrid(fig, 111,
                nrows_ncols = (1,2),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )
    #plt.subplot(131)
    #plt.imshow(x)
    #plt.axis('off')
    #plt.title('Origianl with Pred %d' % clean_prob)
    grid[0].imshow(x)
    grid[0].axis('off')    
    

    if name == 'FW':
       grid[0].set_title( 'FWnucl' + ' Perturbed', fontsize=20)
    elif name == 'pgdnucl':
       grid[0].set_title( 'PGDnucl' + ' Perturbed', fontsize=20)
    else:
       grid[0].set_title(name.upper() + ' Perturbed', fontsize=20)

    imc = grid[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    grid[1].axis('off')


    if name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)

    if name == 'FW':
       grid[1].set_title('FWnucl' + ' Perturbation', fontsize=20)
    elif name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)
    else:
       grid[1].set_title(name.upper() + ' Perturbation', fontsize=20)
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    imc.set_clim(vmin=0, vmax=0.06)
    #imc.set_clim(vmin=0, vmax=0.15)
    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    plt.margins(0,0)
    cbar.ax.tick_params(labelsize=14) 
    plt.tight_layout(h_pad=0) 
    figname =  'IMAGES/generates_group/' + picname + name + '.png' 
    print("figname: ", figname)
    #plt.show()
    plt.savefig(figname, bbox_inches='tight',pad_inches = 0)
    #plt.savefig('image' + name + args.file + '.png')
    #plt.show()
    plt.close()

def plotting3(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob, name):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    fig=plt.figure(figsize=(6.5, 3))
    #fig=plt.figure(figsize=(8, 3))
    #fig=plt.figure(figsize=(12, 5))
    #plt.subplot(131)
    #plt.imshow(x)
    #plt.axis('off')
    #plt.title('Origianl with Pred %d' % clean_prob)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.005, hspace=0.005)

    plt.subplot(gs1[0])
    plt.imshow(x_adv)
    plt.axis('off')
    if name == 'FW':
       plt.title( 'dog')
    else:
       plt.title('dog')

    ax1 = plt.subplot(gs1[1])
    imc = plt.imshow(x_adv, cmap='hot', interpolation='nearest')
    plt.axis('off')
    if name == 'FW':
       plt.title('dog')
    else:
       plt.title('dog')
    #axes = plt.subplot(gs1[2])
    #cbar_ax = fig.add_axes([0.90, 0.12, 0.02, .750])
    #fig.colorbar(imc, cax=cbar_ax)
    fig.colorbar(imc)
    #plt.colorbar(imc, cax=cax)
    #imc.set_clim(vmin=0, vmax=0.06)
    imc.set_clim(vmin=0, vmax=0.07)
    #imc.set_clim(vmin=0, vmax=0.15)

    #plt.tight_layout(pad=0.0)  
    plt.tight_layout() 
    figname = args.file.split('/')[0] + '/ image' + name + args.file.split('/')[1] + '.png' if '/' in args.file else 'image' + name +    args.file + '.png' 

    plt.savefig(figname)
    #plt.savefig('image' + name + args.file + '.png')
    plt.show()




def OpenImage(filename, picname='1'):
   img = Image.open(filename)
   image_tensor = preprocess(img) #preprocess an i
   img.save('IMAGES/generates/' + picname + '.png' , "png")
   image_tensor = image_tensor.unsqueeze(0).cuda() # add batch dimension.  C X H X W ==> B X C X H X W


   img_variable = Variable(image_tensor, requires_grad=True).cuda() #convert tensor into a variable



   output = model.forward(img_variable)
   label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
   print("True lable:", label_idx)



   print("label_idx", label_idx)
   x_pred = labels[label_idx.data.item()]
   print("True pred:", x_pred)
   #get probability dist over classes
   output_probs = F.softmax(output, dim=1)
   x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]).item() * 100,4)
   print(x_pred_prob)

   return img_variable, label_idx


#labels_link = "https://savan77.github.io/blog/files/labels.json"    
#labels_json = requests.get(labels_link).json()
json_file = open("labels.json")
labels_json = json.load(json_file)
labels = {int(idx):label for idx, label in labels_json.items()}
#mean and std will remain same irresptive of the model you use
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

'''
preprocess = transforms.Compose([
                transforms.Resize((299,299)),  
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
'''
preprocess = transforms.Compose([
                transforms.Resize((299,299)),  
                transforms.ToTensor()
            ])

mu = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1).cuda()
stdv = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1).cuda()

unnormalize = lambda x: x*stdv + mu
normalize = lambda x: (x-mu)/stdv



eps = 0.02

def gen_adv_ex(img_variable):

   #y_true = 282   #tiger cat  ##change this if you change input image
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target)
   #perform a backward pass in order to get gradients
   loss = torch.nn.CrossEntropyLoss()
   output = model.forward(img_variable)
   loss_cal = loss(output, target)
   loss_cal.backward(retain_graph=True)    #this will calculate gradient of each variable (with requires_grad=True) and can be accessed by "var.grad.data"


   
   x_grad = torch.sign(img_variable.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
   x_adversarial = img_variable.data + eps * x_grad          #find adv example using formula shown above
   x_grad = x_adversarial - img_variable
   x_grad = x_grad.detach()
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print(x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, lable_adv, adv_pred_prob



def pgd_nuc(inputs, targets, rand, step_size, epsilon, num_steps, mod=None):
        attacker = PGD(F.cross_entropy, eps=epsilon, 
                 alpha=step_size,
                 type_projection='nuc',
                 iters=num_steps,
                 device='cuda', 
                 p=2, normalizer=normalize)
        x = attacker.attack(mod, inputs, targets) 
        return x
def gen_adv_ex_pgd_nuc(img_variable):
    eps = 3.0
    #y_true = 282   #tiger cat  ##change this if you change input image
    niters = 20
    target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
    x_adversarial = pgd_nuc(inputs=img_variable, targets=target, rand=False, step_size= 0.3, epsilon=eps, num_steps=niters, mod = model)
    x_grad = x_adversarial - img_variable
    lable_adv, adv_pred_prob, x_adv_pred = calculate_prob(x_adversarial)

    print("pgd nuc pred: ", x_adv_pred)

    return x_adversarial, x_grad, lable_adv, adv_pred_prob

def pgd_white(inputs, targets, rand, step_size, epsilon, num_steps, mod=None, **kwargs):
        
        x = inputs.detach()
        if rand:
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = mod(normalize(x))
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, 0, 1)

        return mod(x), x

def calculate_prob(x_adversarial):
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)
   
   
   return lable_adv, adv_pred_prob, x_adv_pred

def gen_adv_ex_pgd(img_variable):
    eps = 0.02
    #y_true = 282   #tiger cat  ##change this if you change input image
    niters = 20
    target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
    _, x_adversarial = pgd_white(inputs=img_variable, targets=target, rand=False, step_size=0.01, epsilon=eps, num_steps=niters, mod = model)
    x_grad = x_adversarial - img_variable
    lable_adv, adv_pred_prob, x_adv_pred = calculate_prob(x_adversarial)
    print("pgd pred: ", x_adv_pred)
    return x_adversarial, x_grad, lable_adv, adv_pred_prob

def gen_adv_ex_FW(img_variable):

   #y_true = 336   #tiger cat  ##change this if you change input image
   niters = 20
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target)
   
   eps = 3
   FW = FW_vanilla_batch(radius=eps, normalizer=normalize, T_max=niters, rand=True, device='cuda')
   x_adversarial = FW.attack(model, img_variable, target, niters)
   x_grad = x_adversarial - img_variable
   
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print("FW pred: ", x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, lable_adv, adv_pred_prob


def gen_adv_ex_FW_group(img_variable):

   #y_true = 336   #tiger cat  ##change this if you change input image
   niters = 30
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target) 

   eps = 5
   FW = FW_vanilla_batch(radius=eps, normalizer=normalize, T_max=niters, device='cuda', group_norm='group', nbr_par=50, rand=True)
   x_adversarial = FW.attack(model, img_variable, target, niters)
   x_grad = x_adversarial - img_variable
   
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print("Group-FW pred: ", x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, lable_adv, adv_pred_prob

def gen_adv_ex_FW(img_variable):

   #y_true = 336   #tiger cat  ##change this if you change input image
   niters = 20
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target)
   
   eps = 3
   FW = FW_vanilla_batch(radius=eps, normalizer=normalize, T_max=niters, device='cuda')
   x_adversarial = FW.attack(model, img_variable, target, niters)
   x_grad = x_adversarial - img_variable
   
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print("FW pred: ", x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, lable_adv, adv_pred_prob





model_configs = { 
    "fgsm": ("fgsm", gen_adv_ex),
    "pgd": ("pgd", gen_adv_ex_pgd),
    "pgdnucl": ("pgdnucl", gen_adv_ex_pgd_nuc),
    "FW": ("FW", gen_adv_ex_FW),
    "FW_group": ("FW_group", gen_adv_ex_FW_group)
    }
models = list(map(lambda model_name: model_configs[model_name], args.models))
print("transfer_models: ", models)
#y_true = 89

#images = ['IMAGES/ILSVRC2012_val_00017951.JPEG', 'IMAGES/ILSVRC2012_val_00001478.JPEG']
images = ['IMAGES/ILSVRC2012_val_00020582.JPEG', 'IMAGES/ILSVRC2012_val_00001478.JPEG', 'IMAGES/ILSVRC2012_val_00022457.JPEG',
'IMAGES/ILSVRC2012_val_00001543.JPEG', 'IMAGES/ILSVRC2012_val_00023069.JPEG', 'IMAGES/ILSVRC2012_val_00002805.JPEG',
'IMAGES/ILSVRC2012_val_00024835.JPEG', 'IMAGES/ILSVRC2012_val_00003563.JPEG', 'IMAGES/ILSVRC2012_val_00024991.JPEG',
'IMAGES/ILSVRC2012_val_00007098.JPEG', 'IMAGES/ILSVRC2012_val_00040576.JPEG', 'IMAGES/ILSVRC2012_val_00009564.JPEG',
'IMAGES/ILSVRC2012_val_00043769.JPEG', 'IMAGES/ILSVRC2012_val_00009606.JPEG', 'IMAGES/ILSVRC2012_val_00044587.JPEG',
'IMAGES/ILSVRC2012_val_00010816.JPEG', 'IMAGES/ILSVRC2012_val_00044685.JPEG', 'IMAGES/ILSVRC2012_val_00012843.JPEG',
'IMAGES/ILSVRC2012_val_00045679.JPEG', 'IMAGES/ILSVRC2012_val_00017261.JPEG', 'IMAGES/ILSVRC2012_val_00047447.JPEG',
'IMAGES/ILSVRC2012_val_00017951.JPEG', 'IMAGES/ILSVRC2012_val_00048007.JPEG', 'IMAGES/ILSVRC2012_val_00019795.JPEG',
'IMAGES/ILSVRC2012_val_00048827.JPEG', 'IMAGES/ILSVRC2012_val_00019940.JPEG']

labelnames = ['bustard', 'hamster', 'safety_pin', 'greenhouse', 'crane', 'valley', 'black_stork', 'rapeseed', 'dumbbell',
'steel_drum', 'folding_chair', 'crane', 'ruddy_turnstone', 'drilling_platform', 'limousine', 'organ',
'wall_clock', 'loupe', 'necklace', 'Angora', 'upright', 'liner', 'English_foxhound', 'lycaenid', 'lycaenid_2', 'steel_drum_2']  

print("images: ", len(images), "labelnames: ", len(labelnames))
dictImagesNames = dict({})
for i in range(len(images)):
    dictImagesNames[images[i]] = labelnames[i] 
picnames = ['pics/ILSVRC2012_val_00038256.JPEG', 'pics/ILSVRC2012_val_00049144.JPEG', 'pics/ILSVRC2012_val_000324.JPEG',
'pics/ILSVRC2012_val_00035883.JPEG']
picnames_eps3 = ['pics/ILSVRC2012_val_00009649.JPEG', 'pics/ILSVRC2012_val_00023805.JPEG']
picnames_eps5 = ['pics/ILSVRC2012_val_00033162.JPEG', 'pics/ILSVRC2012_val_00024234.JPEG']

logname = 'IMAGES/hist_group/group_results.txt'
with open(logname, 'w') as logfile:
     logwriter = csv.writer(logfile, delimiter=',')
     logwriter.writerow(['filename', 'name', 'True label', 'Pred label', 'norm2'])

#picnames = ['new_pics/ILSVRC2012_val_00002255.JPEG']
img_variable, label_idx = OpenImage(picnames[0])
images_cum = torch.zeros_like(img_variable, device='cpu', requires_grad=False)
adv_cum = torch.zeros_like(img_variable, device='cpu', requires_grad=False)
for filename in images:
#for filename in picnames:
#for filename in picnames_eps3:
#for filename in picnames_eps5:
   img_variable, label_idx = OpenImage(filename, dictImagesNames[filename])
   y_true = label_idx.item()
   for name, method in models:
      x_adversarial, x_grad, x_adv_pred, adv_pred_prob = method(img_variable)
      with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow([filename, name, labels[y_true], labels[x_adv_pred], torch.norm(x_grad, p=2)])
      #filename = filename + 'eps3'
      if (name == "FW") and (x_adv_pred == y_true):
         print("The file name is: ", filename, "lable: ", x_adv_pred)
      #plotting(img_variable, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, name)      
      #plotting2(img_variable.detach(), x_adversarial.detach(), x_grad.detach(), eps, name, filename)
      plotting_name(img_variable.detach(), x_adversarial.detach(), x_grad.detach(), eps, name, dictImagesNames[filename])
      #images_cum += img_variable.cpu()
      #adv_cum += x_adversarial.cpu()
      #plot_histogram(x_grad, name, filename)
      plot_histogram_name(x_grad, name, dictImagesNames[filename])
#images_cum = images_cum.div(len(picnames))
#adv_cum = adv_cum.div(len(picnames))
#perturb_var = torch.clamp(adv_cum - images_cum, min = 0)
#plot_histogram(perturb_var, name)
   #plotting3(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, name)
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex_FW()
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex_pgd()
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex()
#visualize(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)
#plotting(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)



 

