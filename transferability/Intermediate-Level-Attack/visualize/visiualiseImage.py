
import json
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
sys.path.append('..')
from FW_attack import FW_vanilla_batch

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', help='<Required> source models', required=True)
parser.add_argument('--file', default='ex4.jpg', help='<Required> image file')
args = parser.parse_args()

#url = "https://savan77.github.io/blog/images/ex4.jpg"  #tiger cat #i have uploaded 4 images to try- ex/ex2/ex3.jpg
#response = requests.get(url)
img = Image.open(args.file)


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
    plt.title('Origianl with Pred %d' % clean_prob)

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
       plt.title( 'FWnucl' + ' Perturbed')
    else:
       plt.title(name.upper() + ' Perturbed')

    ax1 = plt.subplot(gs1[1])
    imc = plt.imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    plt.axis('off')
    if name == 'FW':
       plt.title('FWnucl' + ' Perturbation')
    else:
       plt.title(name.upper() + ' Perturbation')
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    #imc.set_clim(vmin=0, vmax=0.15)

    #plt.tight_layout(pad=0.0)  
    #plt.tight_layout()
    cbar_ax = plt.gcf().add_axes([0.93, 0.22, 0.02, .55])
    plt.colorbar(imc, cax=cbar_ax)
    plt.tight_layout(h_pad=0) 
    figname = args.file.split('/')[0] + '/ image' + name + args.file.split('/')[1] + '.png' if '/' in args.file else 'image' + name + args.file + '.png' 

    plt.savefig(figname)
    #plt.savefig('image' + name + args.file + '.png')
    plt.show()

def plotting2(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob, name):
    
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
       grid[0].set_title( 'FWnucl' + ' Perturbed')
    else:
       grid[0].set_title(name.upper() + ' Perturbed')

    imc = grid[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    grid[1].axis('off')

    if name == 'FW':
       grid[1].set_title('FWnucl' + ' Perturbation')
    else:
       grid[1].set_title(name.upper() + ' Perturbation')
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    #imc.set_clim(vmin=0, vmax=0.15)
    plt.colorbar(imc, cax=grid.cbar_axes[0])
    #plt.tight_layout(h_pad=0) 
    figname = args.file.split('/')[0] + '/ image' + name + args.file.split('/')[1] + '.png' if '/' in args.file else 'image' + name + args.file + '.png' 

    plt.savefig(figname)
    #plt.savefig('image' + name + args.file + '.png')
    plt.show()

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
    #imc.set_clim(vmin=0, vmax=0.15)

    #plt.tight_layout(pad=0.0)  
    plt.tight_layout() 
    figname = args.file.split('/')[0] + '/ image' + name + args.file.split('/')[1] + '.png' if '/' in args.file else 'image' + name +    args.file + '.png' 

    plt.savefig(figname)
    #plt.savefig('image' + name + args.file + '.png')
    plt.show()


inceptionv3 = models.inception_v3(pretrained=True).cuda()#download and load pretrained inceptionv3 model
inceptionv3.eval();



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


image_tensor = preprocess(img) #preprocess an i
image_tensor = image_tensor.unsqueeze(0).cuda() # add batch dimension.  C X H X W ==> B X C X H X W


img_variable = Variable(image_tensor, requires_grad=True).cuda() #convert tensor into a variable



output = inceptionv3.forward(img_variable)
label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
print("True lable:", label_idx)



#labels_link = "https://savan77.github.io/blog/files/labels.json"    
#labels_json = requests.get(labels_link).json()
json_file = open("labels.json")
labels_json = json.load(json_file)
labels = {int(idx):label for idx, label in labels_json.items()}
print("label_idx", label_idx)
x_pred = labels[label_idx.data.item()]
print("True pred:", x_pred)



#get probability dist over classes
output_probs = F.softmax(output, dim=1)
x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]).item() * 100,4)
print(x_pred_prob)
eps = 0.02

def gen_adv_ex():

   #y_true = 282   #tiger cat  ##change this if you change input image
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target)
   #perform a backward pass in order to get gradients
   loss = torch.nn.CrossEntropyLoss()
   loss_cal = loss(output, target)
   loss_cal.backward(retain_graph=True)    #this will calculate gradient of each variable (with requires_grad=True) and can be accessed by "var.grad.data"


   
   x_grad = torch.sign(img_variable.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
   x_adversarial = img_variable.data + eps * x_grad          #find adv example using formula shown above
   x_grad = x_adversarial - img_variable
   x_grad = x_grad.detach()
   output_adv = inceptionv3.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print(x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, x_adv_pred, adv_pred_prob


def pgd_white(inputs, targets, rand, step_size, epsilon, num_steps, mod=None, **kwargs):
        
        x = inputs.detach()
        if rand:
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = mod(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, 0, 1)

        return mod(x), x

def calculate_prob(x_adversarial):
   output_adv = inceptionv3.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)

   
   return x_adv_pred, adv_pred_prob

def gen_adv_ex_pgd():
    eps = 0.02
    #y_true = 282   #tiger cat  ##change this if you change input image
    niters = 20
    target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
    _, x_adversarial = pgd_white(inputs=image_tensor, targets=target, rand=False, step_size=0.01, epsilon=eps, num_steps=niters, mod = inceptionv3)
    x_grad = x_adversarial - image_tensor
    x_adv_pred, adv_pred_prob = calculate_prob(x_adversarial)

    return x_adversarial, x_grad, x_adv_pred, adv_pred_prob

def gen_adv_ex_FW():

   #y_true = 336   #tiger cat  ##change this if you change input image
   niters = 20
   target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
   print(target)
   

   FW = FW_vanilla_batch(radius=1, normalizer=normalize, device='cuda')
   x_adversarial = FW.attack(inceptionv3, image_tensor, target, niters)
   x_grad = x_adversarial - image_tensor
   
   output_adv = inceptionv3.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)      #find probability (confidence) of a predicted class

   print(x_adv_pred)
   print(adv_pred_prob)
   return x_adversarial, x_grad, x_adv_pred, adv_pred_prob


model_configs = { 
    "fgsm": ("fgsm", gen_adv_ex),
    "pgd": ("pgd", gen_adv_ex_pgd),
    "FW": ("FW", gen_adv_ex_FW)
    }
models = list(map(lambda model_name: model_configs[model_name], args.models))
print("transfer_models: ", models)
#y_true = 89
y_true = label_idx.item()
for name, method in models:
   x_adversarial, x_grad, x_adv_pred, adv_pred_prob = method()
   #plotting(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, name)
   plotting2(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, name)
   #plotting3(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, name)
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex_FW()
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex_pgd()
#x_adversarial, x_grad, x_adv_pred, adv_pred_prob = gen_adv_ex()
#visualize(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)
#plotting(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)

