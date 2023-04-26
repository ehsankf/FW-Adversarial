from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from models.net_mnist import *
from models.small_cnn import *

import time

from FW_vanilla_batch_git import FW_vanilla as FW_vanilla_batch
#from FW_vanilla_batch import FW_vanilla_batch

'''
Usage:
python mnist.py --mode FW
'''

epsilons = [0, .05, .1, .15, .2, .25, .3]
epsilons = [0, 1, 2, 3, 4, 5]
epsilons = [1, 3, 5]
#pretrained_model = "data/lenet_mnist_model.pth"
pretrained_model = "data/checkpoint.pt.best"
use_cuda=True


parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='<Required> attack models', required=True)
args = parser.parse_args()


'''
# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''
def load_model(model_path):

        #model = SmallCNN().to(device)
        model = Net().to(device)
        #model.load_state_dict()
        import dill
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), pickle_module=dill)
        print(checkpoint.keys())
        #print(checkpoint['model'].keys())
        
        
        state_dict_path = 'model'
        if not ('model' in checkpoint):
           state_dict_path = 'state_dict'
        if ('net' in checkpoint):
           checkpoint['model'] = checkpoint['net']
           del checkpoint['net']
        
        if 'model' in checkpoint:
              if hasattr(checkpoint['model'], 'state_dict'):
                 print("Hi ehsan*******************")
                 sd = checkpoint['model'].state_dict()
              else:
                 print("This is me")
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

        sd = {k.replace('module.attacker.model.', '').replace('module.model.','').replace('module.','').replace('model.',''):v for k,v in sd.items()}
        
        keys = model.state_dict().keys()
        new_state = {}
        for k in sd.keys():
           if k in keys:
              new_state[k] = sd[k]
           else:
              print(k)
        
        model.load_state_dict(new_state)
        '''
        else:
           model = checkpoint['model']
        checkpoint = None
        sd = None
        '''
        model.eval().to(device)
 
        return model


# MNIST Test dataset and dataloader declaration
testset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
sampler = torch.utils.data.sampler.RandomSampler( testset, replacement=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, sampler=sampler)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = load_model(pretrained_model)
# Initialize the network
model = Net().to(device)

# Load the pretrained model
#sd = torch.load(pretrained_model, map_location='cpu')['model']
#sd = {k.replace('module.attacker.model.', '').replace('module.model.','').replace('module.','').replace('model.',''):v for k,v in sd.items()}
#model.load_state_dict(sd)
model = load_model(pretrained_model)
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def fgsm_attack(model, data, epsilon, target):

    output = model(data)
    # Calculate the loss
    loss = F.nll_loss(output, target)
        
        
    #loss = loss1(output, target)
  
    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    # Return the perturbed image
    return perturbed_data

class NegLLLoss(torch.nn.Module):
    def __init__(self, min_problem=True):
        super(NegLLLoss, self).__init__()
        assert type(min_problem) == bool
        self.min_problem = min_problem

    def forward(self, x, target):
        if self.min_problem:
            return nn.NLLLoss()(x, target)
        else:
            return -nn.NLLLoss()(x, target)

def test( model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []
    inf = float('inf')

    def negloss(data, label):
            #return -1 * F.nll_loss(data, label)
            return -1 * F.cross_entropy(data, label)
    if args.mode == 'FW':
       '''
       FW = FW_vanilla_batch(model,
                 radius=epsilon, eps=1e-3,
                 T_max=20,
                 type_ball='lp',
                 p=1,
                 mask=False,
                 channel_subsampling=True,
                 size_groups=4,
                 group_subsampling=True,
                 device=device, sum_dir = None, loss_type = 'Log')
       '''
       loss = NegLLLoss(min_problem=False)
       FW = FW_vanilla_batch(model, loss,
                              type_ball='nuclear',
                              p=1,
                              mask=False,
                              radius=epsilon,
                              stepsize=0.1,
                              eps=1e-10,
                              T_max=1000,
                              device=device)

    # Loop over all examples in test set
    for idx, datam in enumerate(test_loader):
 
        data, target = datam
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        
        if args.mode == 'fgsm':
           # Call FGSM Attack
           perturbed_data = fgsm_attack(model, data, epsilon, target)
        elif args.mode == 'FW':
           perturbed_data = FW.attack(data, target)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        print("index: ", idx)
        if final_pred.item() == target.item():            
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                perturbm = (data-perturbed_data).squeeze()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, torch.norm(perturbm, 'fro'),
                             torch.norm(perturbm, inf), torch.norm(perturbm, 'nuc')) )
            elif (epsilon == 0):
                break
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 10:
                print("this is me")
                perturbm = (data-perturbed_data).squeeze()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, torch.norm(perturbm, 'fro'),
                             torch.norm(perturbm, inf), torch.norm(perturbm, 'nuc')) )
            else:
                break
         

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

'''
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
'''

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(11, 5))
gs1 = gridspec.GridSpec(len(epsilons),10)
gs1.update(wspace=0.085, hspace=0.05) # set the spacing between axes. 
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        #cnt += 1
        plt.subplot(gs1[cnt])
        #plt.subplot(len(epsilons),len(examples[0]),gs1[cnt])
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=20)
        orig,adv,ex, distfro, distinf, distnuc = examples[i][j]
        #plt.title("{} -> {}\n {:0.2f}, {:0.2f}, {:0.2f}".format(orig, adv, distfro, distinf, distnuc))
        plt.imshow(ex[0], cmap="gray")
        #plt.imshow(ex, cmap="gray")
        plt.text(2, 25, "{}".format(adv), size=15, ha='center', va='center', color='r')
        cnt += 1
plt.tight_layout(h_pad=0)
plt.savefig('mnist_lenet.png', bbox_inches='tight',pad_inches = 0)
plt.show()



