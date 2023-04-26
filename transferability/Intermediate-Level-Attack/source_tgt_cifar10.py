import argparse
from cifar10models import *
import numpy as np
import pandas as pd
from tqdm import tqdm

from FW_attack import FW_vanilla_batch, FW_vanilla_batch_un, wrap_attack_FW

'''
Usage:
python source_tgt_cifar10.py --source_models ResNet18 --transfer_models ResNet18 DenseNet121  --out_name=test1.csv  --attacks FW_vanilla_batch --num_batches=5 --batch_size=32

'''
data_preprocess = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

mean, stddev = data_preprocess
mu = torch.Tensor(data_preprocess[0]).unsqueeze(-1).unsqueeze(-1).cuda()
std = torch.Tensor(data_preprocess[1]).unsqueeze(-1).unsqueeze(-1).cuda()

unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

class NegLLLoss(torch.nn.Module):
    def __init__(self, min_problem=True):
        super(NegLLLoss, self).__init__()
        assert type(min_problem) == bool
        self.min_problem = min_problem

    def forward(self, x, target):
        if self.min_problem:
            return nn.NLLLoss()(x, target)
        else:
            return -nn.NLLLoss()(x, target)

def model_name(model):
    return model.__name__

model_configs = {
    model_name(ResNet18): (ResNet18, 'checkpoints/cifar10/resnet18_epoch_347_acc_94.77.pth'),
    model_name(DenseNet121): (DenseNet121,'checkpoints/cifar10/densenet121_epoch_315_acc_95.61.pth'),
    model_name(GoogLeNet): (GoogLeNet, 'checkpoints/cifar10/googlenet_epoch_227_acc_94.86.pth'),
    model_name(SENet18): (SENet18, 'checkpoints/cifar10/senet18_epoch_279_acc_94.59.pth')
}


def attack_name(attack):
    return attack.__name__  
         
attack_configs ={ 
    attack_name(FW_vanilla_batch): wrap_attack_FW(FW_vanilla_batch, {'normalizer': normalize, 'device': 'cuda', 'radius': 5.0, 'T_max':20}),
    attack_name(FW_vanilla_batch_un): wrap_attack_FW(FW_vanilla_batch_un, {'loss': NegLLLoss(False), 'normalizer': normalize, 'device': 'cuda', 'radius': 5.0, 'T_max':20, 'type_subsampling':'None'}),
   
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_models', nargs='+', help='<Required> source models', required=True)
    parser.add_argument('--transfer_models', nargs='+', help='<Required> transfer models', required=True)
    parser.add_argument('--attacks', nargs='+', help='<Required> base attacks', required=True)
    parser.add_argument('--num_batches', type=int, help='<Required> number of batches', required=True)
    parser.add_argument('--batch_size', type=int, help='<Required> batch size', required=True)
    parser.add_argument('--out_name', help='<Required> out file name', required=True)
    args = parser.parse_args()
    return args


def log(out_df, source_model, source_model_file, target_model, target_model_file, batch_index, fool_method,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':model_name(source_model), 
        'source_model_file': source_model_file,
        'target_model':model_name(target_model),
        'target_model_file': target_model_file,
        'batch_index':batch_index,       
        'fool_method':fool_method, 
        'fool_rate':fool_rate, 
        'acc_after_attack':acc_after_attack, 
        'original_acc':original_acc},ignore_index=True)



def get_data(batch_size, mean, stddev):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    #transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

def get_fool_adv_orig(model, adversarial_xs, originals, labels):
    total = adversarial_xs.size(0)
    correct_orig = 0
    correct_adv = 0
    fooled = 0

    advs, ims, lbls = adversarial_xs.cuda(), originals.cuda(), labels.cuda()
    outputs_adv = model(advs)
    outputs_orig = model(ims)
    _, predicted_adv = torch.max(outputs_adv.data, 1)
    _, predicted_orig = torch.max(outputs_orig.data, 1)

    correct_adv += (predicted_adv == lbls).sum()
    correct_orig += (predicted_orig == lbls).sum()
    fooled += (predicted_adv != predicted_orig).sum()
    return [100.0 * float(fooled.item())/total, 100.0 * float(correct_adv.item())/total, 100.0 * float(correct_orig.item())/total]


def test_adv_examples_across_models(transfer_models, adversarial_xs, originals, labels):
    accum = []
    for (network, weights_path) in transfer_models:
        net = network().cuda()
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(weights_path)
        accum.append(res)
    return accum

def get_orig(model, originals, labels):
    total = labels.size(0)
    correct_orig = 0
   

    ims, lbls = originals.cuda(), labels.cuda()
    outputs_orig = model(ims)
    _, predicted_orig = torch.max(outputs_orig.data, 1)

    correct_orig += (predicted_orig == lbls).sum()
    print(100.0 * float(correct_orig.item())/total)

def complete_loop(sample_num, batch_size, attacks, source_models, transfer_models, out_name):
    out_df = pd.DataFrame(columns=['source_model', 'source_model_file', 'target_model','target_model_file', 'batch_index','fool_method', 'fool_rate', 'acc_after_attack', 'original_acc'])


    trainloader, testloader = get_data(batch_size, *data_preprocess)
    for model_class, source_weight_path in source_models:
        model = model_class().cuda()
        model.load_state_dict(torch.load(source_weight_path))
        model.eval()
        dic = model._modules
        for attack_name, attack in attacks:
            print('using source model {0} attack {1}'.format(model_name(model_class), attack_name))
            iterator = tqdm(enumerate(testloader, 0))
            for batch_i, data in iterator:
                if batch_i == sample_num:
                    iterator.close()
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda() 

                #### baseline 
                adversarial_xs = attack(model, unnormalize(images), labels, niters= 50) 
                
                
               
                ### eval
                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels)
                for i, (target_fool_rate, target_acc_attack, target_acc_original, target_weight_path) in enumerate(transfer_list):    
                           
                    out_df = log(out_df,model_class, source_weight_path,transfer_models[i][0], 
                                 target_weight_path, batch_i, attack_name, 
                                 target_fool_rate, target_acc_attack, target_acc_original)


            #save csv
            out_df.to_csv(out_name, sep=',', encoding='utf-8')
            




if __name__ == "__main__":
    args = get_args()
    attacks = list(map(lambda attack_name: (attack_name, attack_configs[attack_name]), args.attacks))
    source_models = list(map(lambda model_name: model_configs[model_name], args.source_models))
    transfer_models = list(map(lambda model_name: model_configs[model_name], args.transfer_models))
   
    complete_loop(args.num_batches, args.batch_size, attacks, source_models, transfer_models, args.out_name);

    
    df = pd.read_csv(args.out_name)

    baselinegroup = df.groupby(['source_model', 'target_model','fool_method'])

    print(baselinegroup.agg({'original_acc':'mean', 'fool_rate':'mean', 'acc_after_attack':'mean'}).reset_index())
    '''
    trainloader, testloader = get_data(args.batch_size, *data_preprocess)
    for model_class, source_weight_path in source_models:
        model = model_class().cuda()
        model.load_state_dict(torch.load(source_weight_path))
        model.eval()
        total, correct = 0, 0
        for batch_idx, (images, labels) in tqdm(enumerate(testloader, 0)):
            images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            get_orig(model, images, labels)
            #print(batch_idx, 'Acc: %.3f%% error: %.3f%% (%d/%d)' %(100.*correct/total, 100.*(total - correct)/total, correct, total))
    '''















