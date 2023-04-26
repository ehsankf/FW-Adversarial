import numpy as np
import csv
import torch
import torch.nn as nn
from .nuclear_group_norm import _LP_group_nuclear
from .p_schatten import _LP_Schatten_gpu
 
            

#from models.linear_minimization import LP_batch  # LP for Linear Programming
def LP_nuclear_gpu(D: torch.Tensor,
                    radius: float,
                    channel_subsampling=True,
                    batch=1, device='cpu') -> torch.Tensor:
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
            U = U.to(device)
            V = V.to(device)
            for i in range(B):
                v_FW[i, c[i], :, :] = radius * torch.mm(U[i, :, 0].view(-1, 1),
                                                        V[i, :, 0].view(1, -1))
                
    else:
            U, _, V = torch.svd(D[:, :, :, :].cpu())
            U = U.to(device)
            V = V.to(device)
            for i in range(B):
                v_FW[i, 0, :, :] = radius * torch.mm(U[i, 0, :, 0].view(-1, 1),
                                                        V[i, 0, :, 0].view(1, -1))
                v_FW[i, 1, :, :] = radius * torch.mm(U[i, 1, :, 0].view(-1, 1),
                                                        V[i, 1, :, 0].view(1, -1))
                v_FW[i, 2, :, :] = radius * torch.mm(U[i, 2, :, 0].view(-1, 1),
                                                        V[i, 2, :, 0].view(1, -1))
                          
    return(v_FW)

class FW_vanilla_batch():
    '''
    Implementing vanilla Frank-Wolfe on various balls.
    '''
    # implementing vannila FW on nuclear ball
    def __init__(self, model,
                 radius=1, eps=1e-10,
                 step_size=1.0,
                 T_max=100,
                 type_ball='lp',
                 p=1,
                 mask=False,
                 channel_subsampling=True,
                 size_groups=4,
                 group_subsampling=True,
                 normalizer = None,
                 rand = True,
                 loss_type = 'Cross',
                 group_norm=None, nbr_par=1,
                 power=None,
                 device='cpu', sum_dir = None):
        '''
        eps: criterion for convergence.
        radius: radius of the distortion ball.
        channel_subsampling: if true, then search for a vertex with respect to
            only one channel (in the case of an RGB image).
        '''
        assert type_ball in ['lp', 'group_lasso', 'nuclear']
        self.model = model
        self.lossfunc = self.NCriterion
        self.radius = radius
        self.eps = eps
        self.T_max = T_max
        self.device = device
        self.step_size = step_size

        # attribute caracterizing the ball.
        self.type_ball = type_ball
        self.p = p
        self.mask = mask
        self.normalizer = normalizer
        self.rand = rand

        # for specific versions of Frank-Wolfe
        self.channel_subsampling = channel_subsampling
        #applying group norm
        self.nbr_par = nbr_par
        self.group_norm = group_norm
        self.power=power
        
    def _LP_batch(self, D: torch.Tensor) -> torch.Tensor:
        
        if self.group_norm == 'group':
           v_FW = _LP_group_nuclear(D,
                              radius=self.radius,
                              nbr_h=self.nbr_par)
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


    def NCriterion(self, x, target):
        return -nn.CrossEntropyLoss()(x, target)
        

    def NLLLoss(self, x, target):
        return -nn.NLLLoss()(x, target)

    def NCWLoss(self, logits, labels):
        # setup the target variable, we need it to be in one-hot form for the loss function
        confidence = 0.0
        self.num_classes = 10
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
        FW_gap = -(grad*(V_FW-X)).view(1, -1).sum(1).item()
        return(FW_gap)

    def attack(self, images: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
        '''
        Method:
        -------
        solving argmin_{||X - X^ori|| <= self.radius} objective(x) with
        vanilla Frank-Wolfe algorithm (i.e. 1/k+1 for the step size).
        '''
        # initialization        
        H = torch.zeros(images.shape, dtype=torch.float32).cuda()
        if self.rand:
          (B, C, Height, Width) = images.shape     
          Ar = torch.rand(B, C, Height) - 0.5
          Br = torch.rand(B, C, Height) - 0.5
          for c in range(C): 
             H[:,c, :, :] =  self.radius * torch.matmul(Ar[:, c, :].view(B, -1, 1),
                                                       Br[:, c, :].view(B, 1, -1))
        
        H.requires_grad = True
        iter = 0
        self.l_gap = []
        x_adv = images + H
        # while iter < self.T_max or gap_FW < self.eps:
        while iter < self.T_max:
            H.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(self.normalizer(torch.clamp(images + H, 0, 1)))
                loss = self.lossfunc(outputs, labels)
            grad = torch.autograd.grad(loss, [H], retain_graph=True)[0]
            V_FW = self._LP_batch(-grad)  # v_FW = argmax_v < v - x; -grad>
            step_size = self.step_size * 1./(iter + 3)
            assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            x_adv = images.detach() + H.detach()
            iter += 1
        x_adv = images.detach() + H.detach()
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv

