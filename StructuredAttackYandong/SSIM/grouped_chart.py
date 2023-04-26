# libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb

"""
python3 grouped_chart.py --attack_type FGSMNet
"""

parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')

parser.add_argument('--attack_type', help='specifying the type of SSIM plot.')

args = parser.parse_args()

barWidth = 0.95

def func_bars(arg_attack_type):

    Bars = dict()
    legends = dict()
    attack_type = 'FGSMNet'
 
    # set heights of bars
    bars1 = [95.1, 85.87, 85.28, 95.34, 90.08, 94.28, 95.1, 93.33, 92.12, 97.47, 95.99, 94.16, 96.52] # accuracy
    bars2 = [0.49, 1.20, 1.55, 0.48, 1.10, 5.72, 0.5, 1.0, 1.50, 1.60, 3.17, 4.65, 3.87] # l2 norm
    bars3 = [0.84, 2.29, 3.28, 0.81, 2.15, 3.19, 1.64, 3.80, 5.94, 6.62, 13.07, 19.12, 13.73] # nuclear norm
    bars4 = [0.9973, 0.9711, 0.9619, 0.9976, 0.9818, 0.9670, 0.8878, 0.6486, 0.5327, 0.5001, 0.3898, 0.3236, 0.7194] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$',  r'FWnucl-group $\epsilon_{S1} = 1$',
     r'FWnucl-group $\epsilon_{S1} = 3$', r'FWnucl-group $\epsilon_{S1} = 5$', r'$\rm{PGD}_2 \,\epsilon = 0.5$', r'$\rm{PGD}_2 \,\epsilon = 1.0$',
     r'$\rm{PGD}_2\, \epsilon = 1.5$', r'$\rm{PGD}\, \epsilon = 0.1$', r'$\rm{PGD}\, \epsilon = 0.2$', r'$\rm{PGD}\, \epsilon = 0.3$', 
     r'$\rm{FGSM} \,\epsilon = 0.3$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'FGSMSmallCNN'
 
    # set heights of bars
    bars1 = [98.04, 93.16, 89.65, 98.25, 96.16, 94.28, 98.14, 97.24, 96.12, 98.56, 97.61, 95.73, 97.95] # accuracy
    bars2 = [0.49, 1.32, 1.75, 0.46, 1.16, 1.54, 0.5, 1.0, 1.5, 1.84, 3.23, 4.68, 3.57] # l2 norm
    bars3 = [0.89, 2.43, 3.52, 0.83, 2.23, 3.19, 1.70, 3.72, 5.76, 7.48, 14.33, 18.98, 12.17] # nuclear norm
    bars4 = [0.9973, 0.9614, 0.9492, 0.9978, 0.9763, 0.9670, 0.9074, 0.6991, 0.5701, 0.4673, 0.3915, 0.3232, 0.8315] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$',  r'FWnucl-group $\epsilon_{S1} = 1$',
     r'FWnucl-group $\epsilon_{S1} = 3$', r'FWnucl-group $\epsilon_{S1} = 5$', r'$\rm{PGD}_2 \,\epsilon = 0.5$', r'$\rm{PGD}_2 \,\epsilon = 1.0$',
     r'$\rm{PGD}_2\, \epsilon = 1.5$', r'$\rm{PGD}\, \epsilon = 0.1$', r'$\rm{PGD}\, \epsilon = 0.2$', r'$\rm{PGD}\, \epsilon = 0.3$', 
     r'$\rm{FGSM} \,\epsilon = 0.3$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]


    attack_type = 'FGSMResNet18'
 
    # set heights of bars
    bars1 = [35.17, 0.67, 0.03, 65.58, 34.02, 18.97, 63.5, 41.07, 20.65, 74.98, 67.65, 49.95, 55.91] # accuracy
    bars2 = [1.58, 3.90, 5.55, 0.78, 1.77, 2.42, 1.0, 1.5, 0.42, 0.85, 1.67, 1.72, 1.68] # l2 norm
    bars3 = [1.01, 3.05, 5.00, 0.56, 1.56, 2.39, 1.15, 2.34, 3.56, 1.0, 1.97, 3.88, 4.04] # nuclear norm
    bars4 = [0.9731, 0.8835, 0.8039, 0.9930, 0.9748, 0.9588, 0.9956, 0.9824, 0.9612, 0.9948, 0.9811, 0.9373, 0.9327] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$',  r'FWnucl-group $\epsilon_{S1} = 1$',
     r'FWnucl-group $\epsilon_{S1} = 3$', r'FWnucl-group $\epsilon_{S1} = 5$', r'$\rm{PGD}_2 \,\epsilon = 0.5$', r'$\rm{PGD}_2 \,\epsilon = 1.0$',
     r'$\rm{PGD}_2\, \epsilon = 1.5$', r'$\rm{PGD}\, \epsilon = 2/255$', r'$\rm{PGD}\, \epsilon = 4/255$', r'$\rm{PGD}\, \epsilon = 8/255$', 
     r'$\rm{FGSM} \,\epsilon = 8/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'FGSMResNet50'
 
    # set heights of bars
    bars1 = [29.84, 0.8, 0.03, 66.64, 32.06, 18.49, 64.8, 35.76, 15.26, 80.94, 72.86, 52.91, 61.44] # accuracy
    bars2 = [1.49, 3.48, 4.92, 0.74, 1.65, 2.25, 0.5, 1.0, 1.5, 0.43, 0.84, 1.67, 1.72] # l2 norm
    bars3 = [1.0, 2.97, 4.86, 0.60, 1.62, 2.52, 1.23, 2.50, 3.78, 1.01, 1.99, 3.90, 4.10] # nuclear norm
    bars4 = [0.9751, 0.9005, 0.8279, 0.9935, 0.9771, 0.9624, 0.9954, 0.9817, 0.9597, 0.9948, 0.9812, 0.9387, 0.9316] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$',  r'FWnucl-group $\epsilon_{S1} = 1$',
     r'FWnucl-group $\epsilon_{S1} = 3$', r'FWnucl-group $\epsilon_{S1} = 5$', r'$\rm{PGD}_2 \,\epsilon = 0.5$', r'$\rm{PGD}_2 \,\epsilon = 1.0$',
     r'$\rm{PGD}_2\, \epsilon = 1.5$', r'$\rm{PGD}\, \epsilon = 2/255$', r'$\rm{PGD}\, \epsilon = 4/255$', r'$\rm{PGD}\, \epsilon = 8/255$', 
     r'$\rm{FGSM} \,\epsilon = 8/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'FGSMWideResNet'
 
    # set heights of bars
    bars1 = [32.54, 0.5, 0.02, 66.28, 33.46, 19.26, 64.65, 38.43, 17.15, 78.83, 71.24, 52.47, 59.06] # accuracy
    bars2 = [1.54, 3.67, 5.19, 0.75, 1.67, 2.30, 0.5, 1.0, 1.5, 0.007, 0.85, 1.67, 1.72] # l2 norm
    bars3 = [1.0, 3.01, 4.92, 0.58, 1.59, 2.48, 1.20, 2.44, 3.70, 1.02, 2.00, 3.92, 4.09] # nuclear norm
    bars4 = [0.9741, 0.8932, 0.8167, 0.9935, 0.9767, 0.9617, 0.9954, 0.9820, 0.9605, 0.9946, 0.9805, 0.9363, 0.9304] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$',  r'FWnucl-group $\epsilon_{S1} = 1$',
     r'FWnucl-group $\epsilon_{S1} = 3$', r'FWnucl-group $\epsilon_{S1} = 5$', r'$\rm{PGD}_2 \,\epsilon = 0.5$', r'$\rm{PGD}_2 \,\epsilon = 1.0$',
     r'$\rm{PGD}_2\, \epsilon = 1.5$', r'$\rm{PGD}\, \epsilon = 2/255$', r'$\rm{PGD}\, \epsilon = 4/255$', r'$\rm{PGD}\, \epsilon = 8/255$', 
     r'$\rm{FGSM} \,\epsilon = 8/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'CIFAR-10-AWP-AT'
 
    # set heights of bars
    bars1 = [36.46, 1.68, 0.14, 72.12, 53.90, 16.29, 0.32] # accuracy
    bars2 = [1.49, 3.54, 5.14, 0.11, 0.54, 2.32, 5.49] # l2 norm
    bars3 = [1.00, 2.93, 4.79, 0.26, 1.22, 5.21, 12.41] # nuclear norm
    bars4 = [0.9732, 0.8925, 0.8078, 0.9977, 0.9819, 0.8836, 0.6740] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$', r'Auto-Attack $\epsilon = 4/255$',  r'Auto-Attack $\epsilon = 8/255$', r'Auto-Attack $\epsilon = 16/255$', r'Auto-Attack $\epsilon = 32/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'CIFAR-10-LBGAT'
 
    # set heights of bars
    bars1 = [33.94, 2.01, 0.26, 73.31, 52.21, 13.81, 0.25] # accuracy
    bars2 = [1.43, 3.40, 5.00, 0.13, 0.61, 2.47, 5.58] # l2 norm
    bars3 = [1.00, 2.91, 4.73, 0.30, 1.40, 5.57, 12.51] # nuclear norm
    bars4 = [0.9739, 0.8982, 0.8174, 0.9972, 0.9781, 0.8754, 0.6766] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$', r'Auto-Attack $\epsilon = 4/255$',  r'Auto-Attack $\epsilon = 8/255$', r'Auto-Attack $\epsilon = 16/255$', r'Auto-Attack $\epsilon = 32/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'CIFAR-100-AWP-AT'
 
    # set heights of bars
    bars1 = [42.67, 13.57, 4.99, 43.58, 28.84, 9.26, 0.51] # accuracy
    bars2 = [0.93, 2.52, 3.84, 0.14, 0.53, 1.70, 3.84] # l2 norm
    bars3 = [1.00, 3.02, 5.09, 0.33, 1.20, 3.82, 8.77] # nuclear norm
    bars4 = [0.9886, 0.9473, 0.9061, 0.9973, 0.9822, 0.9122, 0.7627] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$', r'Auto-Attack $\epsilon = 4/255$',  r'Auto-Attack $\epsilon = 8/255$', r'Auto-Attack $\epsilon = 16/255$', r'Auto-Attack $\epsilon = 32/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    attack_type = 'CIFAR-100-LBGAT'
 
    # set heights of bars
    bars1 = [44.50, 12.23, 4.91, 46.78, 26.72, 5.83, 0.06 ] # accuracy
    bars2 = [0.90, 2.17, 3.04, 0.20, 0.73, 2.11, 4.31] # l2 norm
    bars3 = [0.99, 2.93, 4.78, 0.47, 1.70, 4.91, 10.44] # nuclear norm
    bars4 = [0.9888, 0.9545, 0.9229, 0.9955, 0.9722, 0.8784, 0.7081] # SSIM
    legends[attack_type] = [r'FWnucl $\epsilon_{S1} = 1$', r'FWnucl $\epsilon_{S1} = 3$', r'FWnucl $\epsilon_{S1} = 5$', r'Auto-Attack $\epsilon = 4/255$',  r'Auto-Attack $\epsilon = 8/255$', r'Auto-Attack $\epsilon = 16/255$', r'Auto-Attack $\epsilon = 32/255$']

    Bars[attack_type] = [bars1, bars2, bars3, bars4]

    return Bars[arg_attack_type], legends[arg_attack_type]



Bars, labels = func_bars(args.attack_type)

# Set position of bar on X axis
r1 = 5 * np.arange(len(Bars[0]))
locs = [r1]
for i in range(len(Bars)):
    locs.append([x + (i + 1) * barWidth for x in r1])


# legends
legends = ['Accuracy', r'$\ell_2$ norm', 'Nuclear norm', 'SSIM']

# colors 
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple'] 


fig, ax1 = plt.subplots(figsize=(15, 15))

# list of handels for the figure 
ha = []

# Make the plot
p = ax1.bar(locs[0], Bars[0], color=colors[0], width=barWidth, label=legends[0])
plt.yticks(fontsize=18)
ha.append(p) 

for i, v in enumerate(Bars[0]):
    ax1.text(locs[0][i] - barWidth / 2, v + 3.5, '{:.2f}'.format(v), fontsize=12, weight='bold', rotation=90)

ax2 = ax1.twinx()

for r, bar, c, la in zip(locs[1:], Bars[1:], colors[1:], legends[1:]):
    p = ax2.bar(r, bar, color=c, width=barWidth, label=la)
    ha.append(p) 


# Add xticks on the middle of the group bars
ax1.set_xlabel('Attacks', fontweight='bold', fontsize=18)
ax1.set_xticks([r + 3./2 * barWidth for r in locs[0]])
ax1.set_xticklabels(labels, rotation=60, fontsize=15)
# Set the font of yticks
plt.yticks(fontsize=18) 

# Create legend & Show graphic
ax2.legend(handles=ha, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(Bars), fontsize=15)

for j in range(1, len(locs)-1):
    for i, v in enumerate(Bars[j]):
        ax2.text(locs[j][i] - barWidth / 2, v + 0.75, '{:.2f}'.format(v), fontsize=12, weight='bold', rotation=90)
 
plt.tight_layout()
plt.savefig("groupAttack_{}.png".format(args.attack_type))















