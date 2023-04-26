#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from nuclear_group_norm import _LP_group_nuclear, _var_weight


# helpers
def avg(l):
    return sum(l) / len(l)

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

def sow_images(images):
    """Sow batch of torch images (Bx3xWxH) into a grid PIL image (BWxHx3)

    Args:
        images: batch of torch images.

    Returns:
        The grid of images, as a numpy array in PIL format.
    """
    images = torchvision.utils.make_grid(
        images
    )  # sow our batch of images e.g. (4x3x32x32) into a grid e.g. (3x32x128)
    
    mean_arr, stddev_arr = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    # denormalize
    #for c in range(3):
    #    images[c, :, :] = images[c, :, :] * stddev_arr[c] + mean_arr[c]

    images = images.cpu().numpy()  # go from Tensor to numpy array
    # switch channel order back from
    # torch Tensor to PIL image: going from 3x32x128 - to 32x128x3
    images = np.transpose(images, (1, 2, 0))
    return images


def attack_log(adv_perturb):

            eval_x_adv_linf = torch.norm(adv_perturb, p = float("inf"), dim = 1).sum().item()/adv_perturb.size(0)
            eval_x_adv_l0 = torch.norm(adv_perturb, p=0, dim = 1).sum().item()/adv_perturb.size(0)
            eval_x_adv_l2 = torch.norm(adv_perturb, p=2, dim = 1).sum().item()/adv_perturb.size(0)
            eval_x_adv_lnuc = torch.norm(adv_perturb[:, 0, :, :], p='nuc', dim = (1,2)).sum().item()/adv_perturb.size(0)             

            print('eval_x_adv_linf', eval_x_adv_linf)
            print('eval_x_adv_l0', eval_x_adv_l0)
            print('eval_x_adv_l2', eval_x_adv_l2)
            print('eval_x_adv_lnuc', eval_x_adv_lnuc)

def visualise(images, perturb, x_adv):
    np_image = sow_images(images[:display].detach())
    np_delta = sow_images(perturb[:display].detach())
    np_recons = sow_images((x_adv.detach())[:display])

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(3, 1, 1)
    plt.axis("off")
    plt.imshow(np_recons)
    fig.add_subplot(3, 1, 2)
    plt.axis("off")
    plt.imshow(np_image)
    fig.add_subplot(3, 1, 3)
    plt.axis("off")
    plt.imshow(np_delta)
    plt.show()


#from models.linear_minimization import LP_batch  # LP for Linear Programming
def LP_nuclear_gpu(D: torch.Tensor,
                    radius: float,
                    channel_subsampling=False,
                    batch=1, device='cuda') -> torch.Tensor:
    '''
    Version of _LP_nuclear that uses a function that can be implemented in
    GPU.
    '''
    assert type(D) == torch.Tensor
    if D.ndim == 3:
        (C, H, W) = D.shape
        assert batch == 1
    else:
        (B, C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor').to(device)  # to have float32
    if channel_subsampling:
        if D.ndim == 3:
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, _, V = torch.svd(D[c, :, :])
            v_FW[0, :, :] = radius * torch.mm(U[:, 0].view(len(U[:, 0]), 1),
                                              V[:, 0].view(1, len(V[0, :])))
       
        else:
            c = np.random.randint(0, C, B)  # choose one channel per image.
            # TODO: make that without list completion
            D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, _, V = torch.svd(D_inter)
            #print("U shape", U.shape, "V shape", V.shape)
            U = U.to(device)
            V = V.to(device)
            # TODO: remove the for loop.
            
            for i in range(B):
                v_FW[i, c[i], :, :] = radius * torch.mm(U[i, :, 0].view(-1, 1),
                                                        V[i, :, 0].view(1, -1))
            '''
            for i in range(B):
                v_FW[i, 0, :, :] = radius * torch.mm(U[i, 0, :, 0].view(-1, 1),
                                                        V[i, 0, :, 0].view(1, -1))
                v_FW[i, 1, :, :] = radius * torch.mm(U[i, 1, :, 0].view(-1, 1),
                                                        V[i, 1, :, 0].view(1, -1))
                v_FW[i, 2, :, :] = radius * torch.mm(U[i, 2, :, 0].view(-1, 1),
                                                        V[i, 2, :, 0].view(1, -1))
            '''
            #print("radius is", radius)
            #print(V[:, 0, :, 0].view(B, 1, -1))
            #v_FW[:, 0, :, :] = radius * torch.matmul(U[:, 0, :, 0].view(B, -1, 1),
            #                                            V[:, 0, :, 0].view(B, 1, -1))
                
    else:
            #c = np.random.randint(0, C, B)  # choose one channel per image.
            # TODO: make that without list completion
            #D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, _, V = torch.svd(D[:, :, :, :])
            #U, _, V = torch.svd(D_inter)
            #print("U shape", U.shape, "V shape", V.shape)
            U = U.to(device)
            V = V.to(device)
            # TODO: remove the for loop.
            '''
            for i in range(B):
                v_FW[i, 1, :, :] = radius * torch.mm(U[i, :, 0].view(-1, 1),
                                                        V[i, :, 0].view(1, -1))
            '''
            for i in range(B):
                v_FW[i, 0, :, :] = radius * torch.mm(U[i, 0, :, 0].view(-1, 1),
                                                        V[i, 0, :, 0].view(1, -1))
                v_FW[i, 1, :, :] = radius * torch.mm(U[i, 1, :, 0].view(-1, 1),
                                                        V[i, 1, :, 0].view(1, -1))
                v_FW[i, 2, :, :] = radius * torch.mm(U[i, 2, :, 0].view(-1, 1),
                                                        V[i, 2, :, 0].view(1, -1))
    return(v_FW.cuda())

class FW_vanilla_batch():
    '''
    Implementing vanilla Frank-Wolfe on various balls.
    '''
    # implementing vannila FW on nuclear ball
    def __init__(self, radius=1, 
                 eps=1e-10,
                 step_size=1.0,
                 T_max=100,
                 type_ball='lp',
                 p=1,
                 mask=False,
                 channel_subsampling=False,
                 size_groups=4,
                 group_subsampling=True,
                 normalizer = None,
                 rand = False,
                 power=None,
                 device='cpu', sum_dir = None, group_norm=None, nbr_par=1):
        '''
        eps: criterion for convergence.
        radius: radius of the distortion ball.
        channel_subsampling: if true, then search for a vertex with respect to
            only one channel (in the case of an RGB image).
        '''
        assert type_ball in ['lp', 'group_lasso', 'nuclear']
        self.lossfunc = self.NCWLoss#self.NCWLoss #self.NCriterion
        self.radius = radius
        self.eps = eps
        self.T_max = T_max
        self.device = device
        self.step_size = step_size

        # attribute caracterizing the ball.
        self.type_ball = type_ball
        self.p = p
        self.mask = mask
        self.sum_dir = sum_dir
        self.normalizer = normalizer
        self.rand = rand

        # for specific versions of Frank-Wolfe
        self.channel_subsampling = channel_subsampling
        self.group_norm = group_norm
        self.nbr_par = nbr_par
        self.power = power

    def _LP_batch(self, D: torch.Tensor, Weight = None) -> torch.Tensor:
        
        #v_FW = LP_batch(D,
        #                self.radius,
        #                type_ball=self.type_ball,
        #                p=self.p,
        #                mask=self.mask,
        #                channel_subsampling=self.channel_subsampling)
        if self.group_norm == 'group':
            v_FW = _LP_group_nuclear(D,
                              radius=self.radius,
                              nbr_h=self.nbr_par, w=Weight)
        elif self.group_norm == 'Schatten':
            v_FW = _LP_Schatten_gpu(D,
                              radius=self.radius,
                              p=self.power)
        elif self.group_norm is None:
           v_FW = LP_nuclear_gpu(D,
                              radius=self.radius,
                              channel_subsampling = self.channel_subsampling,
                              batch=1, device=self.device)
        return(v_FW)

    def _ball_norm(self, M):
        # TODO: make a customized one!
        # compute the nuclear norm of a Matrix
        U, sing, VT = np.linalg.svd(M)
        assert np.all(sing >= 0)
        norm = np.sum(sing)
        return(norm)

    def NCriterion(self, x, target):
        return -nn.CrossEntropyLoss()(x, target)
        

    def NLLLoss(self, x, target):
        return -nn.NLLLoss()(x, target)

    def NCWLoss(self, logits, labels):
        # setup the target variable, we need it to be in one-hot form for the loss function
        confidence = 0.0 
        self.num_classes = 1000
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        # compute the probability of the label class versus the maximum other
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        # if non-targeted, optimize for making this class least likely.
        #loss = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = real - other + confidence
        loss = torch.sum(loss)

        return loss

    def _FW_gap_batch(self, grad, X, V_FW):
        # compute <-grad, V_FW - X> where X is the current iteration.
        assert grad.shape == X.shape
        # FW_gap = - torch.sum(grad*(V_FW-X)).item()
        FW_gap = -(grad*(V_FW-X)).view(1, -1).sum(1).item()
        return(FW_gap)

    def attack(self, model, images: torch.Tensor,
               labels: torch.Tensor, visualize=False, display=4) -> torch.Tensor:

        '''
        Method:
        -------
        solving argmin_{||X - X^ori|| <= self.radius} objective(x) with
        vanilla Frank-Wolfe algorithm (i.e. 1/k+1 for the step size).
        '''
        # initialization
        # initialization
        iter = 0
        H = torch.zeros(images.shape, dtype=torch.float32).cuda()
        print("this is meww")
        if self.rand:   
          print("this is me")
          (B, C, Height, Width) = images.shape     
          Ar = torch.rand(B, C, Height)
          Br = torch.rand(B, C, Height)
          for c in range(C): 
            H[:,c, :, :] =  self.radius * torch.matmul(Ar[:, c, :].view(B, -1, 1),
                                                       Br[:, c, :].view(B, 1, -1))
 
        H.requires_grad = True
        # TODO: should give the possibility to start from a vertex..
        # keep track
        self.l_gap = []
        print("images", images.device, "H", H.device)
        x_adv = images + H
        if self.sum_dir:
           summaryfile = open(self.sum_dir, 'a')  
           attack_log(self.model, images, x_adv, labels, 0, iter, summaryfile, self.normalizer)  
        else:
           summaryfile = None   
        if self.group_norm == 'group':
           Weight = _var_weight(images, self.nbr_par)
        else:
           Weight = None 
        # while iter < self.T_max or gap_FW < self.eps:
        while iter < self.T_max:
           # images.requires_grad = True       
            # outputs = model(images) 
            #print("T_max is: ", self.T_max)     
            with torch.enable_grad():
                outputs = model(self.normalizer(torch.clamp(images + H, 0, 1)))
                #outputs = model(torch.clamp(images + H, minpixelval, maxpixelval))
                loss = self.lossfunc(outputs, labels)
            grad = torch.autograd.grad(loss, [H], retain_graph=True)[0]
            '''
            outputs = model(torch.clamp(images + H, 0, 1))
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            '''
            # grad = images.grad
            #grad = H.grad
            # pdb.set_trace()
            V_FW = self._LP_batch(-grad, Weight)  # v_FW = argmax_v < v - x; -grad>
            #V_FW = V_FW.to(self.device)
            #gap_FW = self._FW_gap_batch(grad, H, V_FW)
            # assert gap_FW >= 0   # when subsampling cannot be checked?
            # update iterate
            step_size = self.step_size * 3./(iter + 3)
            assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            #self.l_gap.append(gap_FW)
            #H = H.detach()            
            x_adv = images.detach() + H.detach()
            x_adv = torch.clamp(x_adv, 0, 1)
            #print("nuc norm", torch.norm(H.detach()[:, 0, :, :], p='nuc', dim = (1,2)).mean().item())
            #x_adv = torch.clamp(x_adv, 0, 1)
            # perform visualization
            perturb = x_adv - images
            #attack_log(perturb)
            #print("max image pixel", torch.max(images.data), "min image pixel", torch.min(images.data))
            iter += 1
            if visualize is True and iter == self.T_max:
               visualise(images, perturb, x_adv)   
            # end visualization            
            if summaryfile:
               attack_log(model, images, x_adv, labels, iter, summaryfile, self.normalizer)   
        eval_x_adv_lnuc = torch.norm(H, p='nuc', dim = (2,3)).mean(dim=1).mean(dim=0)
        print("nuc norm: ", eval_x_adv_lnuc)
        x_adv = torch.clamp(images + H, 0, 1).detach()
        return x_adv

def wrap_attack_FW(attack, params, dataset='cifar10'):
    """
    A wrap for attack functions
    attack: an attack function
    params: kwargs for the attack func
    """
    FW = attack(**params)
    def wrap_f(model, images, labels, niters, use_Inc_model=False):
        # attack should process by batch
        return FW.attack(model, images, labels)

    return wrap_f



def _LP_nuclear_gpu_un(D: torch.Tensor,
                    radius: float,
                    type_subsampling=None,
                    batch=1) -> torch.Tensor:
    '''
    Doing LMO for nuclear norms with a function that can be implemented in
    GPU.
    '''
    # TODO: allow for non-channel subsampling.
    assert type(D) == torch.Tensor
    if D.ndim == 3:
        (C, H, W) = D.shape
        assert batch == 1
    else:
        (B, C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')  # to have float32
    if type_subsampling == 'channel':
        if D.ndim == 3:
            # TODO: remove that!! Should only be D.ndim == 4
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, _, V = torch.svd(D[c, :, :])
            v_FW[c, :, :] = radius * torch.mm(U[:, 0].view(len(U[:, 0]), 1),
                                              V[:, 0].view(1, len(V[0, :])))
        else:
            c = np.random.randint(0, C, B)  # choose one channel per image.
            # TODO: make that without list completion
            D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, _, V = torch.svd(D_inter)
            # TODO: remove the for loop.
            for i in range(B):
                v_FW[i, c[i], :, :] = radius * torch.mm(U[i, :, 0].view(len(U[i, :, 0]), 1),
                                                        V[i, :, 0].view(1, len(V[i, 0, :])))
    else:

            U, _, V = torch.svd(D[:, :, :, :])
            for i in range(B):
                v_FW[i, 0, :, :] = radius * torch.mm(U[i, 0, :, 0].view(-1, 1),
                                                        V[i, 0, :, 0].view(1, -1))
                v_FW[i, 1, :, :] = radius * torch.mm(U[i, 1, :, 0].view(-1, 1),
                                                        V[i, 1, :, 0].view(1, -1))
                v_FW[i, 2, :, :] = radius * torch.mm(U[i, 2, :, 0].view(-1, 1),
                                                        V[i, 2, :, 0].view(1, -1))
    return(v_FW)

def LP_batch_un(D: torch.Tensor,
             radius: float,
             type_ball: str,
             p=2,
             mask=True,
             type_subsampling=None,
             proba_subsampling=0.5) -> torch.Tensor:
    if type_subsampling is not None:
        assert type_subsampling in ['None', 'channel', 'group', 'channel_group']
    V_FW = _LP_nuclear_gpu_un(D, radius,
                               type_subsampling=type_subsampling)
    return(V_FW.cuda())

class FW_vanilla_batch_un():
    '''
    Implementing vanilla Frank-Wolfe on various balls.
    '''
    # implementing vannila FW on nuclear ball
    def __init__(self, loss,
                 radius=1, eps=1e-10,
                 T_max=100,
                 type_ball='lp',
                 p=1,
                 mask=False,
                 # channel_subsampling=True,
                 type_subsampling=None,
                 size_groups=4,
                 # group_subsampling=True,
                 proba_subsampling=0.5,
                 normalizer = None,
                 device='cpu', group_norm=None):
        '''
        eps: criterion for convergence.
        radius: radius of the distortion ball.
        channel_subsampling: if true, then search for a vertex with respect to
            only one channel (in the case of an RGB image).
        '''
        assert type_ball in ['lp', 'group_lasso', 'nuclear', 'schatten_p']
        self.lossfunc = loss
        self.radius = radius
        self.eps = eps
        self.T_max = T_max
        self.device = device

        # attribute caracterizing the ball.
        self.type_ball = type_ball
        self.p = p
        self.mask = mask

        # for specific versions of Frank-Wolfe
        self.type_subsampling = type_subsampling
        self.proba_subsampling = proba_subsampling
        self.normalizer = normalizer
        
        self.group_norm = group_norm

    def _LP_batch(self, D: torch.Tensor) -> torch.Tensor:
        
        #v_FW = LP_batch(D,
        #                self.radius,
        #                type_ball=self.type_ball,
        #                p=self.p,
        #                mask=self.mask,
        #                channel_subsampling=self.channel_subsampling)
        if self.group_norm is None:
            v_FW = _LP_group_nuclear(D,
                              radius=self.radius,
                              nbr_h=2)
        else:
            v_FW = LP_nuclear_gpu(D,
                              radius=self.radius,
                              channel_subsampling = self.channel_subsampling,
                              batch=1, device=self.device)
        
        return(v_FW)

    def _ball_norm(self, M):
        # TODO: make a customized one!
        # compute the nuclear norm of a Matrix
        U, sing, VT = np.linalg.svd(M)
        assert np.all(sing >= 0)
        norm = np.sum(sing)
        return(norm)

    def _FW_gap_batch(self, grad, X, V_FW):
        # compute <-grad, V_FW - X> where X is the current iteration.
        assert grad.shape == X.shape
        # FW_gap = - torch.sum(grad*(V_FW-X)).item()
        FW_gap = -(grad*(V_FW-X)).view(1, -1).sum(1).item()
        return(FW_gap)

    '''
    def attack(self, model, images: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
        
        #Method:
        #-------
        #solving argmin_{||X - X^ori|| <= self.radius} objective(x) with
        #vanilla Frank-Wolfe algorithm (i.e. 1/k+1 for the step size).
        
        # initialization
        iter = 0
        H = torch.zeros(images.shape, dtype=torch.float32).cuda()
        # TODO: should give the possibility to start from a vertex..
        # keep track
        self.l_gap = []
        x_adv = images + H
        model.eval()
        # while iter < self.T_max or gap_FW < self.eps:
        while iter < self.T_max:
            # images.requires_grad = True
            H.requires_grad = True
            # outputs = model(images)
            outputs = model(self.normalizer(images + H))
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            # grad = images.grad
            grad = H.grad
            # pdb.set_trace()
            V_FW = self._LP_batch(-grad)  # v_FW = argmax_v < v - x; -grad>
            gap_FW = self._FW_gap_batch(grad, H, V_FW)
            # assert gap_FW >= 0   # when subsampling cannot be checked?
            # update iterate
            step_size = 1/(iter + 3)
            assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            self.l_gap.append(gap_FW)
            H = H.detach()
            iter += 1
        x_adv = torch.clamp(images + H, 0, 1)
        return x_adv
    '''
    def attack(self, model, images: torch.Tensor,
               labels: torch.Tensor, visualize=False, display=4) -> torch.Tensor:

        '''
        Method:
        -------
        solving argmin_{||X - X^ori|| <= self.radius} objective(x) with
        vanilla Frank-Wolfe algorithm (i.e. 1/k+1 for the step size).
        '''
        # initialization
        print("This is me ")
        iter = 0
        H = torch.zeros(images.shape, dtype=torch.float32).cuda()
        #H.requires_grad_()
        # TODO: should give the possibility to start from a vertex..
        # keep track
        self.l_gap = []
        x_adv = images + H        
        #if self.sum_dir:
        #   summaryfile = open(self.sum_dir, 'a')  
        #   attack_log(model, images, x_adv, labels, iter, summaryfile, self.normalizer)  
        #else:
        summaryfile = None  
        (B, C, Height, Width) = H.shape
        self.model.eval()

        X_pgd = Variable(images.data, requires_grad=True)
        with torch.enable_grad():
            loss =  nn.CrossEntropyLoss()(self.model(X_pgd), labels)
        loss.backward()
        U, _, V = torch.svd(X_pgd.grad.data.sign())
        while iter < self.T_max:
            #print("T_max is: ", self.T_max)
            H.requires_grad = True
            # images.requires_grad = True       
            # outputs = model(images)      
            #with torch.enable_grad():
            #    outputs = model(self.normalizer(torch.clamp(images + H, 0, 1)))
            #    #outputs = model(torch.clamp(images + H, minpixelval, maxpixelval))
            #    loss = self.lossfunc(outputs, labels)
            #grad = torch.autograd.grad(loss, [H], retain_graph=True)[0]
            #H.requires_grad = True
            # outputs = model(images)
            outputs = model(self.normalizer(torch.clamp(images + H, 0, 1)))
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            '''
            outputs = model(torch.clamp(images + H, 0, 1))
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            '''
            # grad = images.grad
            grad = H.grad
            # pdb.set_trace()
            V_FW = self._LP_batch(-grad)  # v_FW = argmax_v < v - x; -grad>
            #V_FW = V_FW.to(self.device)
            #gap_FW = self._FW_gap_batch(grad, H, V_FW)
            # assert gap_FW >= 0   # when subsampling cannot be checked?
            # update iterate
            step_size = 3./(iter + 3)
            assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            #self.l_gap.append(gap_FW)
            #H = H.detach()            
            x_adv = images.detach() + H.detach()
            x_adv = torch.clamp(x_adv, 0, 1)
            #print("nuc norm", torch.norm(H.detach()[:, 0, :, :], p='nuc', dim = (1,2)).mean().item())
            #x_adv = torch.clamp(x_adv, 0, 1)
            # perform visualization
            perturb = x_adv - images
            #attack_log(perturb)
            #print("max image pixel", torch.max(images.data), "min image pixel", torch.min(images.data))
            iter += 1
            if visualize is True and iter == self.T_max:
               visualise(images, perturb, x_adv)   
            # end visualization            
            if summaryfile:
               attack_log(model, images, x_adv, labels, iter, summaryfile, self.normalizer)   
            H = H.detach()
        eval_x_adv_lnuc = torch.norm(H, p='nuc', dim = (2,3)).mean(dim=1).mean(dim=0)
        print("nuc norm: ", eval_x_adv_lnuc)
        x_adv = torch.clamp(images + H, 0, 1).detach()
        return x_adv
