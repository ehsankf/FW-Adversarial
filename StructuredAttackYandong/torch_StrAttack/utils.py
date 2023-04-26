import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import pdb


class CAM(nn.Module):
    def __init__(self, network):
        super(CAM, self).__init__()

        self.network = Net(network)
      
    def forward(self, x, topk=3):
        feature_map, output = self.network(x)
        prob, args = torch.sort(output, dim=1, descending=True)
        
        ## top k class probability
        topk_prob = prob.squeeze().tolist()[:topk]
        topk_arg = args.squeeze().tolist()[:topk]
        
        # generage class activation map
        b, c, h, w = feature_map.size()
        pdb.set_trace()
        feature_map = feature_map.view(b, c, h*w).transpose(1, 2)

        cam = torch.bmm(feature_map, self.network.fc_weight).transpose(1, 2)

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val
        
        ## top k class activation map
        topk_cam = cam.view(1, -1, h, w)[0, topk_arg]
        topk_cam = nn.functional.interpolate(topk_cam.unsqueeze(0), 
                                        (x.size(2), x.size(3)), mode='bilinear', align_corners=True).squeeze(0)
        topk_cam = torch.split(topk_cam, 1)

        return topk_prob, topk_arg, topk_cam

class Net(nn.Module):
    def __init__(self, net):
        super(Net, self).__init__()
        self.network = net 
        net_list = list(net.children())
        
        self.feature_extractor = nn.Sequential(*net_list[:-1])
        self.fc_layer = net_list[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.fc1.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.network(x), dim=1)
        return feature_map, output
