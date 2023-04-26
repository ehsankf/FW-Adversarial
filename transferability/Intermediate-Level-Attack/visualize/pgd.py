import torch
import torch.nn.functional as F
import numpy as np
import pdb


def projection(img_pertub, img_ori, eps, type_projection, p=2):
    (batch_size, C, H, W) = img_ori.shape
    assert C == 1 or C == 3, print('it should be either a RGB or grey scale image.')
    assert batch_size == 1, print('code not ready for several batches.')

    if type_projection == 'nuclear':
        return(projection_nuclear(img_pertub, img_ori, eps))
    elif type_projection == 'lp':
        return(projection_lp(img_pertub, img_ori, eps, p=p))

def project(x, inputs, epsilon):
        dx = x - inputs
        dx = dx.flatten(1)
        dx /= torch.norm(dx, p=2, dim=1, keepdim=True) + 1e-9
        dx *= epsilon
        return inputs + dx.view(x.shape)


def simplex_projection(s, radius):
    """Projection onto the unit simplex."""
    if np.sum(s) <=1 and np.alltrue(s >= 0):
        return s
    # Code taken from https://gist.github.com/daien/1272551
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    return radius * np.maximum(s-theta, 0)

def nuclear_projection(x, inputs, radius):
    dx = x - inputs
    batch_num, c, h, w = dx.size()
    output = torch.zeros_like(dx).cpu().numpy()

    for i in range(batch_num):
        for channel in range(c): 
           A = dx[i, channel].detach().cpu().numpy()
           """Projection onto nuclear norm ball."""
           U, s, V = np.linalg.svd(A, full_matrices=False)
           s = simplex_projection(s, radius)
           output[i, channel] =  U.dot(np.diag(s).dot(V))
    output = torch.from_numpy(output).float().cuda()
    return inputs + output

class PGD():
    """
    PGD attack in the paper 'Towards Deep Learning Models Resistant to
    Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)

    """
    def __init__(self, loss,
                 eps=0.3, alpha=2/255,
                 type_projection='linfty',
                 iters=40, rand='False', device='cpu',
                 p=2, normalizer=None):
        self.lossfunc = loss
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.rand = rand
        self.device = device
        # varying the type of projection
        self.type_projection = type_projection
        self.p = p  # in case of projection to the lp ball
        self.normalizer = normalizer

    

    def attack(self, model, images, labels):
        '''
        images: should be normalized images so not in the [0, 1] range.
        '''
        images = images.to(self.device)
        labels = labels.to(self.device)

        x = images.clone().detach()
        min_clip = 0
        max_clip = 1
        
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
        for i in range(self.iters):
             x.requires_grad_()
             with torch.enable_grad():
                logits = model(self.normalizer(x))
                loss = F.cross_entropy(logits, labels, size_average=False)
             grad = torch.autograd.grad(loss, [x])[0]
             if self.type_projection == 'linfty':
               # print(grad)
               x = x.detach() + self.alpha * torch.sign(grad.detach())
               x = torch.min(torch.max(x, images - self.eps), images + self.eps)
               x = torch.clamp(x, min_clip, max_clip)
             elif self.type_projection == 'nuc':
               # print(grad)
               grad = grad.flatten(1)
               #grad /= torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-9
               grad = grad.view(x.shape)
               x.data.add_(self.alpha * grad)
               x.data = nuclear_projection(x.data, images, self.eps)
               x = torch.clamp(x.detach(), min_clip, max_clip) 
             else:
               # print(grad)
               grad = grad.flatten(1)
               grad /= torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-9
               grad = grad.view(x.shape)
               x.data.add_(self.alpha * self.eps * grad)
               x.data = nuclear_projection(x.data, images, self.eps)
               x = torch.clamp(x.detach(), min_clip, max_clip) 

      
        return x
