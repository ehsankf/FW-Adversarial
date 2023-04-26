import numpy as np
import csv
import torch
import torch.nn as nn
import pdb


def get_loss_and_preds(model, x, y):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y).double()
        _, preds = torch.max(logits, 1)
        return loss, preds
 
def attack_log(model, x, x_adv, y, step, summaryfile):
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
                                    eval_x_adv_lnuc])
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
                 channel_subsampling=True,
                 size_groups=4,
                 group_subsampling=True,
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

        # attribute caracterizing the ball.
        self.type_ball = type_ball
        self.p = p
        self.mask = mask
        self.sum_dir = sum_dir

        # for specific versions of Frank-Wolfe
        self.channel_subsampling = channel_subsampling

    def _LP_batch(self, D: torch.Tensor) -> torch.Tensor:
        
        #v_FW = LP_batch(D,
        #                self.radius,
        #                type_ball=self.type_ball,
        #                p=self.p,
        #                mask=self.mask,
        #                channel_subsampling=self.channel_subsampling)
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
        iter = 0
        H = torch.zeros(images.shape, dtype=torch.float32).to(self.device)
        # TODO: should give the possibility to start from a vertex..
        # keep track
        self.l_gap = []
        x_adv = images + H
        summaryfile = open(self.sum_dir, 'a')  
        attack_log(self.model, images, x_adv, labels, iter, summaryfile)    
        # while iter < self.T_max or gap_FW < self.eps:
        while iter < self.T_max:
            # images.requires_grad = True
            H.requires_grad = True
            # outputs = self.model(images)
            outputs = self.model(images + H)
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            # grad = images.grad
            grad = H.grad
            # pdb.set_trace()
            V_FW = self._LP_batch(-grad)  # v_FW = argmax_v < v - x; -grad>
            V_FW = V_FW.to(self.device)
            gap_FW = self._FW_gap_batch(grad, H, V_FW)
            # assert gap_FW >= 0   # when subsampling cannot be checked?
            # update iterate
            step_size = 1/(iter + 3)
            assert step_size > 0 and step_size <= 1
            H = H + step_size * (V_FW - H)
            self.l_gap.append(gap_FW)
            H = H.detach()            
            x_adv = images + H
            print(iter)
            iter += 1
            x_adv = images + H
            attack_log(self.model, images, x_adv, labels, iter, summaryfile)      
        x_adv = images + H
        return x_adv
