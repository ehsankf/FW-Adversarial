from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net1(nn.Module):
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
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
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
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
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
            x.data.add_(self.epsilon * grad)
            x.data = project(x.data, inputs, self.epsilon)
            x = torch.clamp(x.detach(), 0, 1)

        return self.model(x)


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
            
        return self.model(x)

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class ModelNet(nn.Module):
    def __init__(self, i_c=1, n_c=10):
        super(ModelNet, self).__init__()

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)


        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, n_c)


    def forward(self, x_i, _eval=False):

        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
            
        x_o = self.conv1(x_i)
        x_o = torch.relu(x_o)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o)
        x_o = torch.relu(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        x_o = torch.relu(self.fc1(x_o))

        self.train()

        return self.fc2(x_o)


      


