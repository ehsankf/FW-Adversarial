import numpy as np
import csv
import torch
import torch.nn as nn
#from .nuclear_group_norm import _LP_group_nuclear
from p_schatten import _LP_Schatten_gpu
from torch.autograd import Variable
import pdb

def adaptive_step_size(f,
                       d_t: torch.Tensor,
                       x_t: torch.Tensor,
                       grad_t: torch.Tensor,
                       g_t: float,
                       L: float,
                       gamma_max=1,
                       tau=1.2,
                       eta=0.9) -> (float, float):
    '''
    gamma_max = 1 corresponds to the FW case. Returns (gamma, M)..
    g_t : the Frank Wolfe gap
    d_t : the direction of decrease
    f(x_t) : should output the value of the objective function
    '''
    assert tau > 1 and eta <= 1
    # some pre-computed values
    f_x_t = f(x_t)
    inter = torch.sum(grad_t*d_t)
    norm_d_t_2 = torch.sum(d_t**2)

    def _Q_up(gamma: float, M: float) -> float:
        val = f_x_t + gamma*inter + gamma**2/2*norm_d_t_2
        return val
    M = L
    gamma = min(g_t/(L*norm_d_t_2.item()), gamma_max)
    while f(x_t + gamma*d_t) > _Q_up(gamma, M):
        M = tau*M
        print("g_t: ", g_t, "M: ", M, "norm_d_t_2: ", norm_d_t_2.item())
        gamma = min(g_t/(M*norm_d_t_2.item()), gamma_max)
    return(gamma, M, d_t)

def get_loss_and_preds(model, x, y):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y).double()
        _, preds = torch.max(logits, 1)
        return loss, preds
 
def attack_log(model, x, x_adv, y, gap_FW, step, summaryfile):
            loss_clean, preds_clean = get_loss_and_preds(model, x, y)
            clean_acc = 100.*preds_clean.eq(y).sum().item()/len(y)

            loss_adv, preds_adv = get_loss_and_preds(model, x_adv, y)
            preds_acc = 100.*preds_adv.eq(y).sum().item()/len(y)
            
            loss_incr = loss_adv - loss_clean

            eval_x_adv_linf = torch.norm(x - x_adv, p = float("inf"), dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_l0 = torch.norm(x - x_adv, p=0, dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_l2 = torch.norm(x - x_adv, p=2, dim = (1, 2, 3)).sum().item()/len(y)
            eval_x_adv_lnuc = torch.norm(x - x_adv, p='nuc', dim = (2,3)).sum().item()/len(y)
            
                       
            summarywriter = csv.writer(summaryfile, delimiter=',') 
            summarywriter.writerow([loss_clean.item(), loss_adv.item(), loss_incr.item(), clean_acc, preds_acc, 
                                    eval_x_adv_linf, eval_x_adv_l0, eval_x_adv_l2, 
                                    eval_x_adv_lnuc, gap_FW])
            '''
            swriter.add_scalar('loss_clean', loss_clean.data, step)
            swriter.add_scalar('loss_adv', loss_adv.data, step)
            swriter.add_scalar('loss_incr', loss_incr.data, step)
            swriter.add_scalar('clean_acc', clean_acc, step)
            swriter.add_scalar('preds_acc', preds_acc, step)
            swriter.add_scalar('eval_x_adv_linf', eval_x_adv_linf.mean(), step)
            swriter.add_scalar('eval_x_adv_l0', eval_x_adv_l0.mean(), step)
            swriter.add_scalar('eval_x_adv_l2', eval_x_adv_l2.mean(), step)
            swriter.add_scalar('eval_x_adv_lnuc', eval_x_adv_lnuc.mean(), step)
            '''
            

#from models.linear_minimization import LP_batch  # LP for Linear Programming
def LP_nuclear_gpu(D: torch.Tensor,
                    radius: float,
                    channel_subsampling=False,
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
    if not channel_subsampling:        
        if D.ndim == 3:
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, _, V = torch.svd(D[c, :, :])
            v_FW[0, :, :] = radius * torch.mm(U[:, 0].view(len(U[:, 0]), 1),
                                              V[:, 0].view(1, len(V[0, :])))
       
        else:
            #c = np.random.randint(0, C, B)  # choose one channel per image.
            # TODO: make that without list completion
            #D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            # U, _, V = torch.svd(D[:, c, :, :])
            U, _, V = torch.svd(D)
            #print("U shape", U.shape, "V shape", V.shape)
            U = U.to(device)
            V = V.to(device)
            # TODO: remove the for loop.
            #for i in range(B):
            #    v_FW[i, 0, :, :] = radius * torch.mm(U[i, :, 0].view(-1, 1),
            #                                            V[i, :, 0].view(1, -1))
            #print(V[:, 0, :, 0].view(B, 1, -1))
            v_FW[:, 0, :, :] = radius * torch.matmul(U[:, 0, :, 0].view(B, -1, 1),
                                                        V[:, 0, :, 0].view(B, 1, -1))
                
    else:
        raise ValueError('I have not treated this case yet...')
    return(v_FW)

class FW_vanilla_batch():
    '''
    Implementing vanilla Frank-Wolfe on various balls.
    '''
    # implementing vannila FW on nuclear ball
    def __init__(self, model,
                 radius=1, eps=1e-10,
                 T_max=100,
                 type_ball='lp',
                 p=1,
                 mask=False,
                 rand=False,
                 channel_subsampling=False,
                 size_groups=4,
                 group_subsampling=False,
                 device='cpu', sum_dir = None, step_size=1, loss_type='Cross', group_norm=None, nbr_par=1, power=None):
        '''
        eps: criterion for convergence.
        radius: radius of the distortion ball.
        channel_subsampling: if true, then search for a vertex with respect to
            only one channel (in the case of an RGB image).
        '''
        assert type_ball in ['lp', 'group_lasso', 'nuclear']
        self.model = model
        self.lossfunc = self.NCriterion if loss_type == 'Cross' else self.NCWLoss
        self.radius = radius
        self.eps = eps
        self.T_max = T_max
        self.device = device

        # attribute caracterizing the ball.
        self.type_ball = type_ball
        self.p = p
        self.mask = mask
        self.sum_dir = sum_dir

        # for specific versions of Frank-Wolfe
        self.channel_subsampling = channel_subsampling
        #applying group norm
        self.nbr_par = nbr_par
        self.group_norm = group_norm
        self.step_size = step_size
        self.rand = rand
        self.power=power

    def _LP_batch(self, D: torch.Tensor) -> torch.Tensor:
        
        #v_FW = LP_batch(D,
        #                self.radius,
        #                type_ball=self.type_ball,
        #                p=self.p,
        #                mask=self.mask,
        #                channel_subsampling=self.channel_subsampling)
        if self.group_norm == 'group':
            v_FW = _LP_group_nuclear(D,
                              radius=self.radius,
                              nbr_h=self.nbr_par)
        elif self.group_norm == 'Schatten':
            "this is mw***********"
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
        loss = -nn.CrossEntropyLoss()(x, target)
        return(loss)
        

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
        # FW_gap = - torch.sum(grad*(V_FW-X)).item()
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
        # TODO: should give the possibility to start from a vertex..
        # keep track
        self.l_gap = []
        x_adv = images + H
        if self.sum_dir is not None:
           summaryfile = open(self.sum_dir, 'a')  
           attack_log(self.model, images, x_adv, labels, 0, iter, summaryfile)    
        # while iter < self.T_max or gap_FW < self.eps:
        err = (self.model(x_adv).max(1)[1] != labels)
        loss_best = 200
        #self.L = 10
        #csighn = -1.0 
        #(B, C, Height, Width) = H.shape
        self.model.eval()
        #Ar = torch.rand(B, C, Height)
        #Br = torch.rand(B, C, Height)
        #X_pgd = Variable(images.data, requires_grad=True)
        #with torch.enable_grad():
        #    loss =  nn.CrossEntropyLoss()(self.model(X_pgd), labels)
        #loss.backward()
        #U, _, V = torch.svd(X_pgd.grad.data.sign())
        #H[:, 0, :, :] =  self.radius * torch.matmul(U[:, 0, :, 0].view(B, -1, 1),
        #                                              V[:, 0, :, 0].view(B, 1, -1))
        #H[:, 0, :, :] =  self.radius * torch.matmul(Ar[:, 0, :].view(B, -1, 1),
        #                                               Br[:, 0, :].view(B, 1, -1))
        
        #H = -0.30 * X_pgd.grad.data.sign()
        #scalar_nuc = 1. * self.radius / torch.norm(H, 'nuc', dim=(2,3)).mean().item()
        #H = scalar_nuc * H 
        while iter < self.T_max:
            '''
            if iter% 60 == 0:
              
              #H = (torch.rand_like(images) - 0.5) 
              for i in range(Height):
                 for j in range(Width):
                    H[:,:, i, j] = A_r[i, j]
                    #H[:,:, i, j] = csighn*0.3
                    #H = csighn*torch.zeros_like(images).fill_(0.3)
                 #scalar_nuc = 1. * self.radius / torch.norm(H, 'nuc', dim=(2,3)).mean().item()
                 #H = scalar_nuc * H
                 #csighn *= -1.0  
              
              scalar_nuc = 1. * self.radius / torch.norm(H, 'nuc', dim=(2,3)).mean().item()
              H = scalar_nuc * H            
              print("nuc norm: ", torch.norm(H, 'nuc', dim=(2,3)).mean().item(), "epsilon: ", self.radius)
            '''
            # images.requires_grad = True
            H.requires_grad = True
            # outputs = self.model(images)
            outputs = self.model(images + H)
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            if loss < loss_best:
               loss_best = loss
               x_adv_b = images + H
            # grad = images.grad
            grad = H.grad
            # pdb.set_trace()
            V_FW = self._LP_batch(-grad)  # v_FW = argmax_v < v - x; -grad>
            V_FW = V_FW.to(self.device)
            gap_FW = self._FW_gap_batch(grad, H, V_FW)
            # assert gap_FW >= 0   # when subsampling cannot be checked?
            # update iterate
            H = H.detach()
            if self.step_size == 'adaptive':               
               (step_size, M) = adaptive_step_size(lambda x: self.lossfunc(self.model(images + x), labels),
                                                    V_FW - H,  # descent direction
                                                    H,  # current iterate
                                                    grad,
                                                    gap_FW,
                                                    self.L,
                                                    gamma_max=1,
                                                    tau=1.2,
                                                    eta=0.99) 
               self.L = M         
            else:
               step_size = self.step_size*1.0/(iter + 3)
            #print("loss: ", loss.item(), "iter: ", iter, "step_size: ", step_size, "gap_FW: ", gap_FW, "DD: ", torch.sum(V_FW[~err]**2).item(), "H: ", torch.sum(H[~err]**2).item(), "loss_best: ", loss_best.item(), "label", labels.item())
            #assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            self.l_gap.append(gap_FW)  
            if self.sum_dir is not None:                
                attack_log(self.model, images, x_adv, labels, gap_FW, iter, summaryfile)               
            iter += 1
            #if self.group_norm is not None:
            #print("loss: ", loss, "iter: ", iter)
            #if loss < 0:
            x_adv = images + H
            #print(iter)
            err = (self.model(torch.clamp(x_adv, 0, 1)).max(1)[1] != labels)
            if len(~err) == 0:
               print("loss: ", loss, "iter: ", iter)
               break    
        x_adv = images + H
        return x_adv
