import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns 
#import amssymb

#https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html

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

def find(target, myList):
     for i in range(len(myList)):
         if myList[i] == target:
            return i
     return None


def get_csvdata_plot_str(str_index, path):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        head_list = next(csvreader)
        index = find(str_index, head_list)
        if index is not None:
          values=[] #empty list
          steps=[] #empty list
          for row in csvreader:
              print("index:", index)
              values.append(float(row[index])) #it is a named_tuple, element['value'] is wrong 
              #steps.append(element.step)  
          return np.array(range(1, len(values)+1)), np.array(values)

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
          


                    
               
    
def plot_train_eval(index, paths, filename):

    
    #sns.set_palette('Set2')
    plt.style.use('seaborn-whitegrid')
    if not isinstance(paths, list) :
        paths = [paths]
    plt.figure(figsize=(4, 4))
    for i, path in enumerate(paths):
        print(path)
        x_scalar, y_scalar = get_csvdata_plot(index, path)
        print(y_scalar)
        plt.plot(x_scalar[0:20], y_scalar[0:20], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('# of steps', fontsize=15)
    plt.ylabel(y_names[index], fontsize=15)
    #plt.title("Mnist")
    plt.tight_layout()
    dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    if not os.path.isdir(dir_name):
       os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name,
                          "{}.png".format(y_names[index])))
    #plt.savefig(y_names[index] + 'attack.png')
    #plt.grid() 
    plt.savefig(filename)
    plt.show()


def plot_acc_vs__eval(ind1_str, ind2_str, paths, filename):

    
    #sns.set_palette('Set2')
    plt.style.use('seaborn-whitegrid')
    if not isinstance(paths, list) :
        paths = [paths]
    plt.figure(figsize=(4, 4))
    for i, path in enumerate(paths):
        print(path)
        _, x_scalar = get_csvdata_plot_str(ind2_str, path)
        _, y_scalar = get_csvdata_plot_str(ind1_str, path)
        print(y_scalar)
        plt.plot(x_scalar[0:], y_scalar[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel(y_names_dict[ind2_str], fontsize=15)
    plt.ylabel(y_names_dict[ind1_str], fontsize=15)
    #plt.title("Mnist")
    plt.tight_layout()
    #dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    #if not os.path.isdir(dir_name):
    #   os.mkdir(dir_name)
    #plt.savefig(os.path.join(dir_name,
    #                      "{}.png".format(y_names_dict[ind2_str])))
    #plt.savefig(y_names[index] + 'attack.png')
    plt.savefig(filename)
    
    plt.show()
    
    



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

    _, x_scalar = get_csvdata_plot_str(name_funcx, path)
    _, y_scalar = get_csvdata_plot_str(name_funcy, path)
    _, z_scalar = get_csvdata_plot_str(name_funcz, path)
        

    corr = np.corrcoef(x_scalar[1:2000], y_scalar[1:2000])[0][1]
    #ax1.scatter(x_scalar[1:2000], y_scalar[1:2000], color='k', marker='o', alpha=0.5, s=100)
    ax1.scatter(x_scalar[1:100], y_scalar[1:100], color='k', marker='o', alpha=0.5, s=100)
    ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax1.set_xlabel(r'Nuclear Norm')
    ax1.set_ylabel(r'$L_{\infty}$ Norm')

    corr = np.corrcoef(x_scalar[1:2000], z_scalar[1:2000])[0][1]
    #ax2.scatter(x_scalar[1:2000], z_scalar[1:2000], color='k', marker='o', alpha=0.5, s=100)
    ax2.scatter(x_scalar[1:100], z_scalar[1:100], color='k', marker='o', alpha=0.5, s=100)
    ax2.set_title('r = {:.2f}'.format(corr), fontsize=18)
    ax2.set_xlabel(r'Nuclear Norm')
    ax2.set_ylabel(r'$L_{2}$ Norm')
    plt.savefig(name_funcx+'norm_attack.png')
    plt.show()



scalar_names = ['loss_clean', 'loss_adv', 'loss_incr', 'clean_acc', 'preds_acc', 'eval_x_adv_linf', 'eval_x_adv_l0', 'eval_x_adv_l2', 'eval_x_adv_lnuc']

y_names = ['Clean Loss', 'Adversaial Loss', 'Loss Increment', 'Clean Accuracy (%)', 'Accuracy (%)', r'$L_{\infty}$', r'$L_{0}$', r'$L_{2}$', 'Nuclear Norm']

y_names_dict = {'loss_clean':'Clean Loss', 'loss_adv':'Adversaial Loss', 'loss_incr':'Loss Increment', 'clean_acc':'Clean Accuracy (%)', 'preds_acc':'Accuracy (%)', 'eval_x_adv_linf':r'$L_{\infty}$', 'eval_x_adv_l0':r'$L_{0}$', 'eval_x_adv_l2':r'$L_{2}$', 'eval_x_adv_lnuc':'Nuclear Norm', 'epsilon':r'$\varepsilon_{S1}$', 'Test Acc':'Accuracy (%)'}

colors = ['ro-', 'g+--', 'b+-', 'b+-.', 'r.-']


#path = ['summary/sum_pgd/pgditer7eps0.3SmallCNNmnist.csv', 'summary/sum_pgd/pgditer20eps0.3SmallCNNmnist.csv',
#         'summary/sum_pgd/pgditer41eps0.3SmallCNNmnist.csv']

path_pgd = ['summary/sum_pgd/pgditer7eps0.3Netmnist_Net.csv', 'summary/sum_pgd/pgditer20eps0.3Netmnist_Net.csv',
         'summary/sum_pgd/pgditer41eps0.3Netmnist_Net.csv']
'''
path_pgd = 

pgditer7eps0.3Netmnist_Net.csv
pgditer20eps0.3Netmnist_Net.csv
pgditer41eps0.3Netmnist_Net.csv

pgditer7eps0.3SmallCNNmnist.csv
pgditer20eps0.3SmallCNNmnist.csv
pgditer41eps0.3SmallCNNmnist.csv
'''

'''
path_FW = 

FWiter10eps1Netmnist_Net.csv
FWiter20eps1Netmnist_Net.csv 
FWiter50eps1Netmnist_Net.csv
FWiter100eps1Netmnist_Net.csv


FWiter10eps3Netmnist_Net.csv
FWiter20eps3Netmnist_Net.csv
FWiter50eps3Netmnist_Net.csv
FWiter100eps3Netmnist_Net.csv

FWiter10eps5Netmnist_Net.csv
FWiter20eps5Netmnist_Net.csv
FWiter50eps5Netmnist_Net.csv
FWiter100eps5Netmnist_Net.csv

FWiter10eps1SmallCNNmnist.csv
FWiter20eps1SmallCNNmnist.csv  
FWiter50eps1SmallCNNmnist.csv
FWiter100eps1SmallCNNmnist.csv

FWiter10eps3SmallCNNmnist.csv
FWiter20eps3SmallCNNmnist.csv
FWiter50eps3SmallCNNmnist.csv
FWiter100eps3SmallCNNmnist.csv 

FWiter10eps5SmallCNNmnist.csv 
FWiter20eps5SmallCNNmnist.csv
FWiter50eps5SmallCNNmnist.csv  
FWiter100eps5SmallCNNmnist.csv  
'''       
   
    
 

#path = ['summary/sum_pgd/pgditer20eps0.3SmallCNNmnist.csv', 'summary/sum_FW/FWiter20eps3SmallCNNmnist.csv']
path = ['summary/sum_pgd_ResNet18/pgditer20eps0.03137254901960784ResNet18ResNet18.csv', 'summary/sum_FW_ResNet18/FWiter20eps1ResNet18ResNet18.csv']
#path_accs_smallcnn = ['summary/sum_pgd/pgditer20eps0.3SmallCNNmnist.csv', 'summary/sum_FW/FWiter20eps1SmallCNNmnist.csv',
#'summary/sum_FW/FWiter20eps3SmallCNNmnist.csv', 'summary/sum_FW/FWiter20eps5SmallCNNmnist.csv']

path_accs_Lenet = ['summary/sum_pgd_ResNet18/pgditer20eps0.03137254901960784ResNet18ResNet18.csv', 'summary/sum_FW_ResNet18/FWiter20eps1ResNet18ResNet18.csv',
'summary/sum_FW_ResNet18/FWiter20eps3ResNet18ResNet18.csv', 'summary/sum_FW_ResNet18/FWiter20eps5ResNet18ResNet18.csv']



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

#plot_scatter_norm('eval_x_adv_lnuc', 'eval_x_adv_linf', 'eval_x_adv_l2', 'summary/sum_FW/FWiter100eps5SmallCNNmnist.csv')

#norm with scattering plots
#path_scatter = 'summary/sum_FW/FWiter100eps5SmallCNNmnist.csv'
#path_scatter = 'summary/sum_pgd/pgditer41eps0.3SmallCNNmnist.csv'
#plot_scatter_norm('eval_x_adv_lnuc', 'eval_x_adv_linf', 'eval_x_adv_l2', path_scatter)

#accuracy versus L2
#plot_acc_vs__eval('preds_acc', 'eval_x_adv_l2', path, 'plots/conv_vs_l2_cifar.png')

#accuray versus nuc norm
#plot_acc_vs__eval('preds_acc', 'eval_x_adv_lnuc', path, 'plots/conv_vs_lnuc_cifar.png')

#accuray versus linf norm
#plot_acc_vs__eval('preds_acc', 'eval_x_adv_linf', path)

#accuracy versus radius for ResNet18
'''
path_clean = ['logger/log_FW_ResNet18/FWiter10ResNet18ResNet18ResNet18Cleanaccuracies.csv', 
       'logger/log_FW_ResNet18/FWiter20ResNet18ResNet18ResNet18Cleanaccuracies.csv',
       'logger/log_FW_ResNet18/FWiter50ResNet18ResNet18ResNet18Cleanaccuracies.csv',
       'logger/log_FW_ResNet18/FWiter100ResNet18ResNet18ResNet18Cleanaccuracies.csv']

path_adversarial = ['logger/log_FW_ResNet18/FWiter10ResNet18ResNet18ResNet18Adversarialaccuracies.csv', 
       'logger/log_FW_ResNet18/FWiter20ResNet18ResNet18ResNet18Adversarialaccuracies.csv',
       'logger/log_FW_ResNet18/FWiter50ResNet18ResNet18ResNet18Adversarialaccuracies.csv',
       'logger/log_FW_ResNet18/FWiter100ResNet18ResNet18ResNet18Adversarialaccuracies.csv']

legend = [r'FWnucl 10', r'FWnucl 20', r'FWnucl 50', r'FWnucl 100', r'FWnucl $\epsilon=5$']
plot_acc_vs__eval('Test Acc', 'epsilon', path_clean, 'plots/conv_vs_radius_clean.png')

legend = [r'FWnucl 10', r'FWnucl 20', r'FWnucl 50', r'FWnucl 100', r'FWnucl $\epsilon=5$']
plot_acc_vs__eval('Test Acc', 'epsilon', path_adversarial, 'plots/conv_vs_radius_adv.png')
'''
#accuracies versus the powers for p-Scahtten norm for ResNet18
path_clean = ['logger/log_p_Schatten_FW_ResNet18/FWiter50power1.5ResNet18ResNet18ResNet18Cleanaccuracies.csv', 
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power2ResNet18ResNet18ResNet18Cleanaccuracies.csv',
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power5ResNet18ResNet18ResNet18Cleanaccuracies.csv',
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power-1ResNet18ResNet18ResNet18Cleanaccuracies.csv']

path_adversarial = ['logger/log_p_Schatten_FW_ResNet18/FWiter50power1.5ResNet18ResNet18ResNet18Adversarialaccuracies.csv', 
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power2ResNet18ResNet18ResNet18Adversarialaccuracies.csv',
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power5ResNet18ResNet18ResNet18Adversarialaccuracies.csv',
       'logger/log_p_Schatten_FW_ResNet18/FWiter50power-1ResNet18ResNet18ResNet18Adversarialaccuracies.csv']

#legend = [r'p = 1.5', r'p = 2', r'p = 5', r'p = -1', r'FWnucl $\epsilon=5$']
#plot_acc_vs__eval('Test Acc', 'epsilon', path_clean, 'plots/conv_vs_radius_clean.png')

#legend = [r'p = 1.5', r'p = 2', r'p = 5', r'p = -1', r'FWnucl $\epsilon=5$']
#plot_acc_vs__eval('Test Acc', 'epsilon', path_adversarial, 'plots/conv_vs_radius_adv.png')

import torch



def plot_queries_vs_acc(data1, data2, filename, i=0):

    
    #sns.set_palette('Set2')
    plt.style.use('seaborn-whitegrid')
    print("data1", data2)
    plt.figure(figsize=(4, 4))
    x_scalar = data1.mean(dim=0)
    for j in range(len(x_scalar)-1):
      x_scalar[j+1] = x_scalar[j] + x_scalar[j+1]
    y_scalar = data2.mean(dim=0) 
    print("y_scalar", y_scalar, "x_scalar", x_scalar)
    plt.plot(x_scalar[0:], y_scalar[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('Queries', fontsize=15)
    plt.ylabel('Success Rate', fontsize=15)
    #plt.title("Mnist")
    plt.tight_layout()
    #dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    #if not os.path.isdir(dir_name):
    #   os.mkdir(dir_name)
    #plt.savefig(os.path.join(dir_name,
    #                      "{}.png".format(y_names_dict[ind2_str])))
    #plt.savefig(y_names[index] + 'attack.png')
    plt.savefig(filename)
    
    #plt.show()

def plot_queries_vs_acc_paths(paths, filename, i=0):

    
    #sns.set_palette('Set2')
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(4, 4))
    for i, path in enumerate(paths):
       results = torch.load(path)
       data1 = results['queries']
       data2 = results['succs']
       print("size data1: ", data1.shape)
       x_scalar = data1.mean(dim=0) 
       print("size x_scalar: ", x_scalar.shape)      
       for j in range(len(x_scalar)-1):
          x_scalar[j+1] = x_scalar[j] + x_scalar[j+1]
       y_scalar = data2.mean(dim=0) 
       plt.plot(x_scalar[0:], y_scalar[0:], colors[i], label=legend[i], alpha=0.5)
       
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('Queries', fontsize=15)
    plt.ylabel('Success Rate', fontsize=15)
    #plt.title("Mnist")
    plt.tight_layout()
    #dir_name = os.path.dirname(path) + '/' + path.split('/')[-1].split('.')[0]
    #if not os.path.isdir(dir_name):
    #   os.mkdir(dir_name)
    #plt.savefig(os.path.join(dir_name,
    #                      "{}.png".format(y_names_dict[ind2_str])))
    #plt.savefig(y_names[index] + 'attack.png')
    plt.savefig(filename)


#results = torch.load('pixel_resnet50_1000_10000_224_0.2000_rand.pth')
#results_agents = torch.load('pixel_resnet18_to_resnet50_10_10000_224_0.2000_rand.pth')
#keys = ['original', 'vecs', 'probs', 'succs', 'queries', 'l2_norms', 'linf_norms']
#print('keys: ', results.keys(), 'shape of queries:', results[keys[4]].shape)
#plot_queries_vs_acc(results['queries'], results['succs'], 'queries_vs_success.png')
#plot_queries_vs_acc(results_agents['queries'], results_agents['succs'], 'queries_vs_success_agents.png')
colors = ['r--', 'g--', 'b-', 'b-.', 'r.-']
legend = ['Hard Lable', r'Soft Lable', 'PGD 40']
paths_black = ['pixel_googlenet_to_resnet50_100_10000_224_0.2000_rand.pth', 'white_pixel_googlenet_to_resnet50_100_10000_224_0.2000_rand.pth']
#plot_queries_vs_acc_paths(['white_pixel_googlenet_to_resnet50_100_10000_224_0.2000_rand_hardlabel.pth', 'white_pixel_googlenet_to_resnet50_100_10000_224_0.2000_rand_soflabel.pth'], 'queries_vs_success_path.png', i=0)
legend = ['SIMBA', r'Revised SIMBA', 'PGD 40']
plot_queries_vs_acc_paths(paths_black, 'queries_vs_success_path_black.png', i=0)
print()
#Comparison of robust accuracy as we increase the number of
#attack steps for FW vs. PGD on MNIST. Each reported robust accuracy is
#an average of 6 trials. As the number of steps increases, FW outperforms
#PGD.
#accuracy versus the number of steps for cifar10 and mnist
#legend = ['PGD', r'FWnucl $\epsilon$=1', 'FWnucl $\epsilon$=3', 'FWnucl $\epsilon$=5']
#plot_train_eval(find('preds_acc', scalar_names), path_accs_Lenet, 'plots/acc_cifar.png')

'''
for i in range(len(scalar_names)):
    plot_train_eval(i, path)
'''


##T0###Do
#bar plot for change in pixel values

#https://github.com/svarthafnyra/CNN_Visualizations
#generate plots for 2d perturbations

#Averaged classification accuracy of MIT Madry-Labs adversarially
#trained MNIST model under a single run of different attacks: cross-entropy
#(left), CWinf loss (right).

#from: Sensitivity of the FW Attack to different choices of . . .
# schatten norm (different values of p) accuracy vs number of steps on the $linf$ adversarially trained model of [31] at the different thresholds for different $p$.
# p  versus accuracies for different epsilons

#ablation study
#no restart
# no restart, random start
#100 restart
#100 restart, random start



#Downloade checkpoints for robust modeland compare them. trades, madr, L0
#Models robustness on MNIST (left) and CIFAR-10(right):  impact on accuracy as we increase the maximumperturbation

