import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('model_path', metavar='MODEL',
                    help='path to model')

args = parser.parse_args()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            #res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res

def load_model(model_path, model):
     checkpoint = torch.load(model_path)
     checkpoint = checkpoint['model']
     new_checkpoint = {}
     for key in checkpoint.keys():
         if key.startswith('module.attacker') or key.startswith('module.normalizer'):
            continue
         new_checkpoint[key.replace('module.model.', '')] = checkpoint[key]

     model.load_state_dict(new_checkpoint)

     return model

valdir = './ILSVRC2012_img_val'

criterion = nn.CrossEntropyLoss().cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False,
        num_workers=6, pin_memory=True)

model = models.resnet50(pretrained=False).cuda()

model = load_model(args.model_path, model)



model.eval()
import pdb
pdb.set_trace()
correct1 = 0
correct5 = 0
total = 0
with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                images = images.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print(acc1[0], acc5[0], target.size(0))
            correct1 += acc1[0]
            correct5 += acc5[0]  
            total += target.size(0)  
            print("Acc@1: {0:.3f} Acc@5: {1:.3f} ".format(correct1*100.0 / total, correct5*100.0 / total))

