import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
#matplotlib inline  
from matplotlib.font_manager import FontProperties

import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'testimagenet.csv', help='<Required> source files')
args = parser.parse_args()

df = pd.read_csv(args.file)
print("num batches: " + str(df['batch_index'].max()))

# Original model accuracies
print(df.groupby(['source_model', 'target_model','fool_method'])['original_acc'].mean().reset_index())

baseline = df.groupby(['source_model', 'target_model','fool_method'])['acc_after_attack'].mean().reset_index()

baselinegroup = df.groupby(['source_model', 'target_model','fool_method'])

print(baselinegroup.agg({'original_acc':'mean', 'fool_rate':'mean', 'acc_after_attack':'mean'}).reset_index())
#print(baseline)

