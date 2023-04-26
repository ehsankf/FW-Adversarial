import argparse
import numpy as np
import torch
import pdb

def sampler(dataset, split_rate):
    indexs = [[] for _ in range(len(dataset.classes))]  # you can't use `[[]] * len(dataset.classes)`. Although there might be better ways but I don't know
    for idx, (_, class_idx) in enumerate(dataset):
        indexs[class_idx].append(idx)
    train_indices  = []
    for cl_idx in indexs:
        size = len(cl_idx)
        split = int(np.floor(split_rate * size))
        np.random.shuffle(cl_idx)
        train_indices.extend(cl_idx[:split])

    return train_indices

def eval_test(model, test_loader, device):
    """
    evaluate model
    """
    model.eval() 
    natural_err_total = 0
    total = 0
    
    for data, target in test_loader:
        X, y = data.to(device), target.to(device)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        total += target.size(0)
        #print('natural err: ', err_natural)
        natural_err_total += err_natural

    total_acc = 100.*(total - natural_err_total)/total
    return total_acc.item()

def compute_norm(x, x_adv):
    """
    compute the perturbation norm
    """ 
    x, x_adv = x.float(), x_adv.float()
    adv_perturb = (x_adv - x).view(x.size(0), -1)
    eval_x_adv_linf = torch.norm(adv_perturb, p = float("inf"), dim=1)
    eval_x_adv_l2 = torch.norm(adv_perturb, p=2, dim=1)         
    eval_x_adv_l0 = torch.sum(torch.max(torch.abs(x - x_adv) > 1e-10, dim=1)[0], dim=(1,2)).float()
    eval_x_adv_lnuc = torch.norm(x[:, 0, :, :] - x_adv[:, 0, :, :], p='nuc', dim=(1,2))   

    return eval_x_adv_linf.mean().item(), eval_x_adv_l2.mean().item(), \
           eval_x_adv_l0.mean().item(), eval_x_adv_lnuc.mean().item() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='WideResNet34',
                        choices=['WideResNet28', 'WideResNet34', 'PreActResNet18'])
    parser.add_argument('--checkpoint', type=str, default='test')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--train_model', type=str, default='AWP', choices=['AWP', 'LBGAT'],
                        help='specify the training model')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--preprocess', type=str, default='01',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--sample_rate', type=float, default=1.)
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)

    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', default=False, action='store_true')
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='./log.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--attack', type=str, default='auto_attack', choices=['auto_attack', 'FW_attack'],)
    
    args = parser.parse_args()

    epsilons_dict = {'auto_attack': [2./255, 4./255., 8./255.], 'FW_attack': [1., 3., 5.]}   
    adv_shape = {'AWP':10000, 'LBGAT': 5000}[args.train_model] 
    
    for attack in ['FW_attack', 'auto_attack']:
        for epsilon in epsilons_dict[attack]:
           print('{}/{}_{}_{}_{}_1_{}_eps_{:.5f}.pth'.format(args.save_dir, \
                 args.data, args.train_model, attack, args.version, adv_shape, epsilon))

           stat_dict = torch.load('{}/{}_{}_{}_{}_1_{}_eps_{:.5f}.pth'.format(args.save_dir, \
                 args.data, args.train_model, attack, args.version, adv_shape, epsilon))

           print(f'attack: {attack} epsilon: {epsilon} nat_acc: {stat_dict["nat_acc"]: .2f} adv_acc: {stat_dict["adv_acc"]: .2f} linf: {stat_dict["linf"]: .2f} l2: {stat_dict["l2"]: .2f}, l0: {stat_dict["l0"]: .2f}, lnuc: {stat_dict["lnuc"]: .2f}')
    

    
