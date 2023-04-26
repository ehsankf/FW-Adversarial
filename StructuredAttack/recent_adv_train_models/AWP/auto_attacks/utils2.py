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
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--adver_file', type=str, default='adver')

    args = parser.parse_args()
    stat_dict = torch.load('{}/{}'.format(args.save_dir, args.adver_file))

    print('nat_acc:', stat_dict['nat_acc'], 'adv_acc:', stat_dict['adv_acc'],
                 'linf:', stat_dict['linf'], 'l2:', stat_dict['l2'], 'l0:', stat_dict['l0'], 'lnuc:', stat_dict['lnuc'])

    
