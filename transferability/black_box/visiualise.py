
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


#url = "https://savan77.github.io/blog/images/ex4.jpg"  #tiger cat #i have uploaded 4 images to try- ex/ex2/ex3.jpg
#response = requests.get(url)

'''
Usage:

python visiualise.py --models pgd  --file pics/ILSVRC2012_val_00049144.JPEG
'''



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
    figname = 'simba/hist/' + picname + name + '.png'     
    _ = plt.xlabel('Intensity Value', fontsize=18)
    _ = plt.ylabel('Count', fontsize=18)
    #_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.margins(0,0)
    plt.tight_layout(h_pad=0) 
    plt.savefig(figname, pad_inches = 0)
    #plt.show()
    plt.close()

def plotting_name(x, x_adv, x_grad, epsilon, name, picname, label_tgt):
    
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
       grid[0].set_title(label_tgt, fontsize=12)

    imc = grid[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    grid[1].axis('off')

 
    if name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)

    if name == 'FW':
       grid[1].set_title('FWnucl' + ' Perturbation', fontsize=20)
    elif name == 'pgdnucl':
       grid[1].set_title('PGDnucl' + ' Perturbation', fontsize=20)
    else:
       grid[1].set_title('Perturbation', fontsize=20)
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    #imc.set_clim(vmin=0, vmax=0.15)
    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    plt.margins(0,0)
    cbar.ax.tick_params(labelsize=14) 
    plt.tight_layout(h_pad=0) 
    figname =  'simba/' + picname + name + '.png' 
    print("figname: ", figname)
    #plt.show()
    plt.savefig(figname, bbox_inches='tight',pad_inches = 0)
    #plt.savefig('image' + name + args.file + '.png')
    #plt.show()
    plt.close()



def OpenImage(filename, picname='1'):
   img = Image.open(filename)
   image_tensor = preprocess(img) #preprocess an i
   img.save('IMAGES/' + picname + '.png' , "png")
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




def calculate_prob(x_adversarial):
   output_adv = model.forward(normalize(Variable(x_adversarial)))   #perform a forward pass on adv example
   lable_adv = (torch.max(output_adv.data, 1)[1][0]).item()
   x_adv_pred = labels[(torch.max(output_adv.data, 1)[1][0]).item()]    #classify the adv example
   op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
   adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]).item() * 100, 4)
   
   
   return lable_adv, adv_pred_prob, x_adv_pred



def get_labels(model_tgt, model_src, x):
    output_tgt = model_tgt(normalize(x).cuda()).cpu()
    label_idx_tgt = torch.max(output_tgt.data, 1)[1][0].item()
    
    output_src = model_src(normalize(x).cuda()).cpu()
    label_idx_src = torch.max(output_src.data, 1)[1][0].item()
    return labels[label_idx_tgt], labels[label_idx_src]

def get_probs(model, model_src, x, y):
    output = model(normalize(x).cuda()).cpu()
    probs = torch.nn.Softmax()(output)[:, y]
    
    output_src = model_src(normalize(x).cuda()).cpu()
    label_idx_src = torch.max(output_src.data, 1)[1][0].item()
    return torch.diag(probs.data), label_idx_src

# 20-line implementation of (untargeted) SimBA for single image input
def simba_single(x, y, num_iters=10000, epsilon=0.2):
    n_dims = x.view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    last_prob, _ = get_probs(model, model_src, x, y)
    for i in range(num_iters):
        diff = torch.zeros(n_dims).cuda()
        diff[perm[i]] = epsilon
        left_prob, label_left_src = get_probs(model, model_src, (x - diff.view(x.size())).clamp(0, 1), y)
        if left_prob < last_prob and label_left_src == y:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob, label_right_src = get_probs(model, model_src, (x + diff.view(x.size())).clamp(0, 1), y)
            if right_prob < last_prob and label_right_src == y:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x



#model = models.inception_v3(pretrained=True).cuda()#download and load pretrained inceptionv3 model
#model = models.resnet50(pretrained=True).cuda()
model = models.densenet121(pretrained=True).cuda()
model_src = models.resnet18(pretrained=True).cuda()
#model = models.googlenet(pretrained=True).cuda()
model.eval()
model_src.eval()



json_file = open("labels.json")
labels_json = json.load(json_file)
labels = {int(idx):label for idx, label in labels_json.items()}
#mean and std will remain same irresptive of the model you use
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

preprocess = transforms.Compose([
                transforms.Resize((299,299)),  
                transforms.ToTensor()
            ])

mu = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1).cuda()
stdv = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1).cuda()

unnormalize = lambda x: x*stdv + mu
normalize = lambda x: (x-mu)/stdv




#images = ['ILSVRC2012_val_00020582.JPEG', 'ILSVRC2012_val_00001478.JPEG']

#labelnames = ['bustard', 'hamster']  

images = ['ILSVRC2012_val_00001478.JPEG']

labelnames = ['hamster'] 

print("images: ", len(images), "labelnames: ", len(labelnames))
dictImagesNames = dict({})
for i in range(len(images)):
    dictImagesNames[images[i]] = labelnames[i] 
picnames = ['ILSVRC2012_val_00020582.JPEG', 'ILSVRC2012_val_00001478.JPEG']

#picnames = ['new_pics/ILSVRC2012_val_00002255.JPEG']
img_variable, label_idx = OpenImage(picnames[0])
images_cum = torch.zeros_like(img_variable, device='cpu', requires_grad=False)
adv_cum = torch.zeros_like(img_variable, device='cpu', requires_grad=False)
for filename in images:
  img_variable, label_idx = OpenImage(filename, dictImagesNames[filename])
  y_true = label_idx.item()
  for mod_src in ['resnet18', 'resnet50', 'googlenet', 'densenet121']:
   model_src = getattr(models, mod_src)(pretrained=True).cuda()
   model_src.eval()   
   for mod_tgt in ['resnet18', 'resnet50', 'googlenet', 'densenet121']:
      if mod_tgt == mod_src:
         continue
      # load model and dataset
      model = getattr(models, mod_tgt)(pretrained=True).cuda()
      model.eval()       
      prob_org, _ = get_probs(model, model_src, img_variable, y_true)
      x_adversarial = simba_single(img_variable, y_true)
      label_tgt, label_src = get_labels(model, model_src, x_adversarial)
      prob_adv, _ = get_probs(model, model_src, x_adversarial, y_true)
      print("tgt label: ", label_tgt, " src label: ", label_src )
      name = 'simba' + '_src_' + mod_src + '_tgt_' + mod_tgt
      x_grad = x_adversarial - img_variable
      plotting_name(img_variable.detach(), x_adversarial.detach(), x_grad.detach(), 0.2, name, dictImagesNames[filename], label_tgt + ' ' + str(round(prob_org.item(), 2))  + '-->' + str(round(prob_adv.item(), 2)))
      
      plot_histogram_name(x_grad, name, dictImagesNames[filename])




 

