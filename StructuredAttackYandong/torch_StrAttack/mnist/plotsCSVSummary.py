import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns 
import pdb
#import amssymb

def get_data_plot(name_func, path):
    for file in os.listdir(path) :
        if file.find('tfevents') > -1 :
            path = path + '/' + file         
            #example of path is 'example_results/events.out.tfevents.1496749144.L-E7-thalita'
            ea = event_accumulator.EventAccumulator(path, 
                                                size_guidance={ # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                event_accumulator.IMAGES: 4,
                                                event_accumulator.AUDIO: 4,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 1,
                                                 })
            ea.Reload() # loads events from file 
            for function in ea.Tags()['scalars'] :
                if function.find(name_func) > -1 : #to find an approximate name_func 
                    values=[] #empty list
                    steps=[] #empty list
                    for element in (ea.Scalars(function)) :
                        values.append(element.value) #it is a named_tuple, element['value'] is wrong 
                        steps.append(element.step)  
                        

                    return np.array(steps), np.array(values)

def get_csvdata_plot(index, path):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        head_list = next(csvreader)
        print("length: ", csvreader)
        if index < len(head_list):
          values=[] #empty list
          steps=[] #empty list
          for row in csvreader:
              values.append(float(row[index])) #it is a named_tuple, element['value'] is wrong 
              #steps.append(element.step)  
          return np.array(range(1, len(values)+1)), np.array(values)
          


                    
               
    
def plot_train_eval2(index, paths):

    
    sns.set_palette('Set2')
    if not isinstance(paths, list) :
        paths = [paths]
    plt.figure()
    for i, path in enumerate(paths):
        print(path)
        x_scalar, y_scalar = get_csvdata_plot(index, path)
        print(y_scalar)
        plt.plot(x_scalar[0:], y_scalar[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('# of steps', fontsize=15)
    plt.ylabel(y_names[index], fontsize=15)
    plt.title("Mnist")
    plt.tight_layout()
    dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[index])))
    #plt.savefig(y_names[index] + 'attack.png')
    
    plt.show()

def plot_train_eval(x, results, dir_name, index=0):

    
    sns.set_palette('Set2')
    plt.figure()
    for i, y in enumerate(results):
        plt.plot(x[0:], y[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel(r'$\nu$', fontsize=15)
    plt.ylabel(y_names[index], fontsize=15)
    plt.title("Mnist")
    plt.tight_layout()
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[0])))
    #plt.savefig(y_names[index] + 'attack.png')
    
    # plt.show()
    
    



'''
import seaborn as sns
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);
    plt.show()

line_plot([1,2,3], [1,2,3], 'training', 'test', title='BTC')
'''

def plot_scatter_norm(name_funcx, name_funcy, name_funcz, path):

    sns.set(style='darkgrid')
    sns.set_palette('Set2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    _, x_scalar = get_csvdata_plot(name_funcx, path)
    _, y_scalar = get_csvdata_plot(name_funcy, path)
    _, z_scalar = get_csvdata_plot(name_funcz, path)
        

    corr = np.corrcoef(x_scalar[:2000], y_scalar[:2000])[0][1]
    ax1.scatter(x_scalar[:2000], y_scalar[:2000], color='k', marker='o', alpha=0.5, s=100)
    ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax1.set_xlabel('nuc-norm')
    ax1.set_ylabel('linf-norm')

    corr = np.corrcoef(x_scalar[:2000], z_scalar[:2000])[0][1]
    ax2.scatter(x_scalar[:2000], z_scalar[:2000], color='k', marker='o', alpha=0.5, s=100)
    ax2.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax2.set_xlabel('nuc-norm')
    ax2.set_ylabel('l2-norm')
    plt.savefig(name_funcx+'norm_attack.png')
    plt.show()



scalar_names = ['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf', 'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc']

y_names = ['IS', 'Clean Loss', 'Adversaial Loss', 'Loss Increment', 'Clean Accuracy (%)', 'Accuracy (%)', r'$L_{\infty}$', r'$L_{0}$', r'$L_{2}$', 'Nuclear Norm']

colors = ['ro-', 'r+-', 'g+--', 'b+-', 'b+-.', 'c+-.']


#path = ['summary/sum_pgd/pgditer7eps0.3SmallCNNmnist.csv', 'summary/sum_pgd/pgditer20eps0.3SmallCNNmnist.csv',
#         'summary/sum_pgd/pgditer41eps0.3SmallCNNmnist.csv']

#path = ['summary/sum_pgd/pgditer7eps0.3Netmnist_Net.csv', 'summary/sum_pgd/pgditer20eps0.3Netmnist_Net.csv',
#         'summary/sum_pgd/pgditer41eps0.3Netmnist_Net.csv']

import torch

path = ['summary/sum_rand_FW/FWiter20eps1.0SmallCNNdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps1.0SmallCNNdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps3.0SmallCNNdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps5.0SmallCNNdata.csv_dict.pt', 'summary/sum_rand_StrAttack/StrAttackiter10eps-1.0SmallCNNdata.csv_dict.pt',
'summary/sum_rand_CS/CSiter10eps-1.0SmallCNNdata.csv_dict.pt']


# path = ['summary/sum_rand_FW/FWiter20eps1.0Netdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps1.0Netdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps3.0Netdata.csv_dict.pt', 'summary/sum_rand_FW_group/FW_groupiter20eps5.0Netdata.csv_dict.pt', 'summary/sum_rand_StrAttack/StrAttackiter10eps-1.0Netdata.csv_dict.pt', 'summary/sum_rand_CS/CSiter10eps-1.0Netdata.csv_dict.pt']

legend = ['FWnucl-1', 'FWnucl-group-1', 'FWnucl-group-3', 'FWnucl-group-5', 'StrAttack', 'CS']

results_sum = []
succ_num = []
for pt in path:
   pt_f = torch.load(pt)
   results_sum.append(pt_f['IS_values'])
   # ind = np.where(pt_f['Y'] != pt_f['Y_ADV'])
   # print(f"checkpoint: {pt} succ_tot: {1000 - len(ind[0])}")
   # nuc_norm = torch.norm(pt_f['X_ADV'][ind[0], 0, :, :] - pt_f['X'][ind[0], 0, :, :], p='nuc', dim = (1,2)).mean().item()
   succ_num.append(results_sum[-1][6])

# A = [ 0.0432, 0.027, 0.4307, 1.1456, 0.0021, 0.3993]
# print('&'.join([str(A[i]*1000./succ_num[i]) for i in range(len(A))]))
print('&'.join([f'{succ_num[i]:.2f}' for i in range(len(succ_num))]))
'''
scalar_names = ['loss_clean', 'loss_adv']

paths = ['pgd40NetPureMENETNet56']


for name in scalar_names:
    plot_train_eval (name, paths)

plot_scatter_norm('eval_x_adv_lnuc', 'eval_x_adv_linf', 'eval_x_adv_l2', 'pgd40Netmnistadvtensor85acctensor96')

'''
#path = 'summary/sum_FW10eps10.0NetmnistItr40advtensor88.csv'

#path = 'summary/sum_FW10eps10.0NetmnistItr40advtensor88.csv'

#row_index, steps = get_csvdata_plot('eval_x_adv_lnuc', path)

#print(row_index)
#print(steps)

plot_train_eval(np.arange(30, 100, 10), results_sum, dir_name = path[0].split('/')[0])
#for i in range(len(scalar_names)):
#    plot_train_eval(i, path)



