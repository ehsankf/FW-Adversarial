import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
from SSIM import SSIM, calculate_frechet_distance, calculate_psnr

"""
python utils.py --adver_file adv_inputs/CIFAR100_LBGAT_FW_attack_standard_1_10000_eps_5.00000.pth --intensity --adver_file_helper adv_inputs/CIFAR100_LBGAT_auto_attack_standard_1_10000_eps_0.12549.pth  --epsnuc 5 --epsinf 32/255
"""
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
    parser.add_argument('--adver_file_helper', type=str, default=None)
    parser.add_argument('--intensity', action='store_true', default=False)
    parser.add_argument('--fig_dir', type=str, default='./intensity_values_figure')

    args = parser.parse_args()
    stat_dict = torch.load('{}/{}'.format(args.save_dir, args.adver_file))

    
    if args.intensity: 
        bar_width = 0.0001
        fig, ax1 = plt.subplots(figsize=(15, 15))
        legends = [r'FWnucl $\epsilon_{S1} = 5$', r'Auto-Attack $\epsilon = 16/255$']
        perturbation = np.abs(np.mean(stat_dict['adv_complete'] - stat_dict['x_test'], axis=0)).ravel()
        bin_nums = np.int((max(perturbation) - min(perturbation)) / bar_width)
        ax1.hist(256 * perturbation, bins=bin_nums, color = 'tab:blue', label = legends[0])
        ax1.set_xlabel('Intensity Value', fontsize=18)
        ax1.set_ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if args.adver_file_helper: 
            stat_dict = torch.load('{}/{}'.format(args.save_dir, args.adver_file_helper))
            perturbation = np.abs(np.mean(stat_dict['adv_complete'] - stat_dict['x_test'], axis=0)).ravel()
            bin_nums = np.int((max(perturbation) - min(perturbation)) / bar_width)
            ax1.hist(256 * perturbation, bins=bin_nums, color = 'tab:red', label = legends[1], alpha = 0.5)
        ax1.legend(fontsize=15)
        ax1.set_ylim(0, 2500)
        plt.savefig("{}/{}_auto_atc{}.png".format(args.fig_dir, args.adver_file.split('/')[-1], args.adver_file_helper.split('_')[-1]), bbox_inches='tight')
    else:
        metric = SSIM(data_range=1.0)
        metric.update([torch.from_numpy(stat_dict['adv_complete']), torch.from_numpy(stat_dict['x_test'])])
        S = metric.compute()
     
        FID = 0 # calculate_frechet_distance(stat_dict['x_test'], stat_dict['adv_complete'])
        psnr = calculate_psnr(stat_dict['x_test'], stat_dict['adv_complete'])
        print('nat_acc:', stat_dict['nat_acc'], 'adv_acc:', stat_dict['adv_acc'],
                 'linf:', stat_dict['linf'], 'l2:', stat_dict['l2'], 'l0:', stat_dict['l0'], 
                 'lnuc:', stat_dict['lnuc'], 'SSIM:', S, 'FID:', FID, 'PSNR:', psnr)


    
