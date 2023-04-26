from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FW_vanilla_batch import FW_vanilla_batch


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
        return x


class Net_binary(nn.Module):
    def __init__(self):
        super(Net_binary, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


'''
class AttackPGD(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_inf bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config, mode):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.mode = mode
        assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'

    def forward(self, inputs, targets):
        if not self.mode:
            return self.model(inputs), inputs

        x = inputs.detach()
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.model(x)
'''

def project_L0_box(inputs, k, lb, ub):
  ''' projection of the batch y to a batch x such that:
        - each image of the batch x has at most k pixels with non-zero channels
        - lb <= x <= ub '''
      
  x = inputs.detach()
  p1 = torch.sum(x**2, dim=-1)
  p2 = torch.min(torch.min(ub - x, x - lb), 0).values
  p2 = torch.sum(p2**2, dim=-1)
  p3 = torch.sort((p1-p2)).values
  p3 = p3.view(p2.shape[0],-1)[:,-k]
  x = x*(torch.le(lb, x) & torch.le(x, ub)) + lb*torch.gt(x, lb) + ub*torch.gt(ub, x)
  x *= ((p1 - p2) >= p3.reshape([-1, 1, 1])).unsqueeze(dim=-1)
  return x


def project(x, inputs, epsilon):
        dx = x - inputs
        dx = dx.flatten(1)
        dx /= torch.norm(dx, p=2, dim=1, keepdim=True) + 1e-9
        dx *= epsilon
        return inputs + dx.view(x.shape)


class AttackPGDL2(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_2 bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config, mode):
        super(AttackPGDL2, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.mode = mode
        assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'

    def forward(self, inputs, targets):
        if not self.mode:
            return self.model(inputs), inputs

        x = inputs.detach()
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            grad = grad.flatten(1)
            grad /= torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-9
            grad = grad.view(x.shape)
            x.data.add_(self.step_size * self.epsilon * grad)
            x.data = project(x.data, inputs, self.epsilon)
            x = torch.clamp(x.detach(), 0, 1)

        return self.model(x), x


class AttackPGDL0(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_2 bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config, mode):
        super(AttackPGDL0, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.mode = mode
        assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'

    def forward(self, inputs, targets):
        if not self.mode:
            return self.model(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            if i > 0:
               grad = torch.autograd.grad(loss, [x])[0]
               # print(grad)
               grad /= (1e-10 + torch.sum(torch.abs(grad), (1,2,3), keepdims=True))
               x.data.add_((torch.FloatTensor(grad.shape).uniform_(0, 1)-0.5)*1e-12 + self.step_size * grad + self.epsilon)
            dx = x.data - inputs.data
            x.data = inputs.data + project_L0_box(dx, 20, -inputs.detach(), 1-inputs.detach())
            x = torch.clamp(x.detach(), 0, 1)
            
        return self.model(x), x


class AttackFW(nn.Module):
    """Adversarial training with PGD.

    Adversarial examples are constructed using PGD under the L_2 bound.
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config, mode):
        super(AttackFW, self).__init__()
        self.model = model
        self.type_ball = config['type_ball']
        self.rand = config['random_start']
        self.p = config['p']
        self.mask = config['mask']
        self.device = config['device']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.subsampling = config['subsampling']
        self.size_groups = config['size_groups']
        self.mode = mode
        #assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'
        
        self.FW_van = FW_vanilla_batch(model, type_ball=self.type_ball,
                                 p=self.p,
                                 mask=self.mask,
                                 radius=self.epsilon,
                                 eps=1e-10,
                                 step_size=self.step_size,
                                 T_max=self.num_steps,
                                 channel_subsampling=self.subsampling,
                                 size_groups=self.size_groups,
                                 group_subsampling=self.subsampling,
                                 device = self.device)
        

    def forward(self, inputs, targets):
        if not self.mode:
            return self.model(inputs), inputs
       
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        x = self.FW_van.attack(self.model, inputs, targets)
        x = torch.clamp(x.detach(), 0, 1) 
        x = x.to(self.device) 
            
        return self.model(x), x
        



      
               


