import torch
import pdb

from linear_minimization import LP  # LP for Linear Programming


class FW_vanilla():
    '''
    Implementing vanilla Frank-Wolfe on various balls.
    '''
    # implementing vannila FW on nuclear ball
    def __init__(self, model, loss,
                 radius=1, eps=1e-10,
                 T_max=100,
                 type_ball='lp',
                 p=1,
                 stepsize=1.0,
                 mask=False,
                 type_subsampling='channel',
                 proba_subsampling=0.5,
                 # channel_subsampling=True,
                 size_groups=4,
                 # group_subsampling=True,
                 device='cpu',
                 L=10,
                 adaptive_ls=False):
        '''
        eps: criterion for convschatten_
        radius: radius of the distortion ball.
        channel_subsampling: if true, then search for a vertex with respect to
            only one channel (in the case of an RGB image).
        '''
        assert type_ball in ['lp', 'group_lasso', 'nuclear', 'schatten_p']
        if type_subsampling is not None:
            assert type_subsampling in ['channel', 'group', 'channel_group']
        self.model = model
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

        # Either upper bound on Liptschitz constant or an hyper-parameter
        self.L = L
        self.adaptive_ls = adaptive_ls
        self.stepsize = stepsize 

    def _LP(self, D: torch.Tensor) -> torch.Tensor:
        v_FW = LP(D,
                  self.radius,
                  type_ball=self.type_ball,
                  p=self.p,
                  mask=self.mask,
                  type_subsampling=self.type_subsampling,
                  proba_subsampling=self.proba_subsampling)
        return(v_FW)

    def _ball_norm(self, M):
        # TODO
        return()

    def _FW_gap(self, grad, X, V_FW):
        # compute <-grad, V_FW - X> where X is the current iteration.
        assert grad.shape == X.shape
        FW_gap = - torch.sum(grad*(V_FW-X)).item()
        # assert FW_gap >= 0 (when computing the full)
        # TODO: when there is no subsampling it should be positive!!
        return(FW_gap)

    def attack(self, images: torch.tensor,
               labels: torch.tensor) -> torch.tensor:
        '''
        Method:
        -------
        solving argmin_{||X - X^ori|| <= self.radius} objective(x) with
        vannila Frank-Wolfe algorithm (i.e. 1/k+1 for the step size).
        '''
        # initialization
        iter = 0
        H = torch.zeros(images.shape, dtype=torch.float32).to(self.device)
        # TODO: try random restarts.
        # keep track
        self.l_gap = []
        x_adv = images + H
        while iter < self.T_max:
            H.requires_grad = True
            # outputs = self.model(images)
            outputs = self.model(torch.clamp(images + H, 0, 1))
            loss = self.lossfunc(outputs, labels)
            loss.backward(retain_graph=True)
            # grad = images.grad
            grad = H.grad
            V_FW = self._LP(-grad)  # v_FW = argmax_v < v - x; -grad>
            V_FW = V_FW.to(self.device)
            gap_FW = self._FW_gap(grad, H, V_FW)
            #print("gap_FW: ", gap_FW, "norm: ", torch.norm(grad, p=2))
            # update iterate
            if not self.adaptive_ls:
                step_size = self.stepsize/(iter + 3)
            else:
                step_size = min(gap_FW/(self.L*torch.sum((V_FW - H)**2)), 1)
            #assert step_size > 0 and step_size <= 1, pdb.set_trace()
            H = H + step_size * (V_FW - H)
            self.l_gap.append(gap_FW)
            H = H.detach()
            iter += 1
        x_adv = images + H
        return torch.clamp(x_adv, 0, 1)
