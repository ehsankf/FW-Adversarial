import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import utils
import math
import random
import torch.nn.functional as F
import argparse
import os
import pdb
import json

'''
python run_simba_white_fgsm_agents.py --num_iters 10000 --pixel_attack  --freq_dims 224 --num_runs 10 --data_root ../Intermediate-Level-Attack/imagenet  --batch_size 1 --model_src googlenet --epsilon 0.02


'''

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
parser.add_argument('--model_src', type=str, default='resnet18', help='type of base model to use')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')

#finite difference
parser.add_argument('--fd_eta', default=0.1, type=float,
                        help='finite difference eta')
args = parser.parse_args()

def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    z = torch.zeros(batch_size, 3, image_size, image_size)
    z[:, :, :size, :size] = x
    return z

def normalize(x):
    return utils.apply_normalization(x, 'imagenet')

def get_probs(model, x, y):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    probs = torch.index_select(torch.nn.Softmax()(output).data, 1, y)
    return torch.diag(probs)

def get_preds(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    _, preds = output.data.max(1)
    return preds

def get_logit(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    return output

def xent_loss(logit, label, target=None):
        return F.cross_entropy(logit, label, reduction='none')

def get_src_grad(model_src, image, label):
    # get sourcc model grad wrt image
    image.requires_grad = True
    adv_logit = model_src(image.cuda()).cpu()   
    loss = xent_loss(adv_logit, label).mean()
    image_grad = torch.autograd.grad(loss, [image])[0]
    return image_grad


def get_tgt_fdm_grad(model, image, label, direction):
    # finite difference
    q1 = direction
    q2 = - direction
    logit_q1 = get_logit(model, image + args.fd_eta * q1 / torch.norm(q1, dim=[1, 2, 3], p=2))
    logit_q2 = get_logit(model, image + args.fd_eta * q2 / torch.norm(q2, dim=[1, 2, 3], p=2))
    l1 = xent_loss(logit_q1, label)
    l2 = xent_loss(logit_q2, label)
    grad = (l1 - l2) / (args.fd_eta)
    grad = grad.view(-1, 1, 1, 1) * direction  
    return direction

def get_labels(model_tgt, model_src, x, gt_label):
    json_file = open("labels.json")
    labels_json = json.load(json_file)
    labels = {int(idx):label for idx, label in labels_json.items()}

    output_tgt = model_tgt(normalize(x).cuda()).cpu()
    label_idx_tgt = torch.max(output_tgt.data, 1)[1][0].item()
    
    output_src = model_src(normalize(x).cuda()).cpu()
    label_idx_src = torch.max(output_src.data, 1)[1][0].item()
    return labels[label_idx_tgt], labels[label_idx_src], labels[gt_label.item()]

def plot_histogram_name(x, name, picname):
    
    x = x.squeeze(0)
    ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #print(ndarr[:128, :128, 0])
    image = ndarr
    newvalues = [x for x in image.ravel() if x != 0]
    fig = plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 18})
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', )
    '''
    newvalues = image[:, :, 0].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'red', alpha = 0.5)
    newvalues = image[:, :, 1].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Green', alpha = 0.5)
    newvalues = image[:, :, 2].ravel()
    newvalues = [x for x in newvalues if x != 0]
    _ = plt.hist(newvalues, bins = 8, color = 'Blue', alpha = 0.5)
    '''
    figname = 'simba/hist/' + name + '.png'     
    _ = plt.xlabel('Intensity Value', fontsize=18)
    _ = plt.ylabel('Count', fontsize=18)
    #_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.margins(0,0)
    plt.tight_layout(h_pad=0) 
    plt.savefig(figname, pad_inches = 0)
    #plt.show()
    plt.close()

def plotting_name(x, x_adv, x_grad, epsilon, name, picname, label_tgt, model_title):
    
    x, x_adv, x_grad = x.cpu(), x_adv.cpu(), x_grad.cpu()
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    #x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    #x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    #fig=plt.figure(figsize=(6.5, 3))
    #fig=plt.figure(figsize=(8, 3))
    #fig=plt.figure(figsize=(12, 5))
    fig = plt.figure(figsize=(12, 5))
    grid = ImageGrid(fig, 111,
                nrows_ncols = (1,2),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )
    #plt.subplot(131)
    #plt.imshow(x)
    #plt.axis('off')
    #plt.title('Origianl with Pred %d' % clean_prob)
    grid[0].imshow(x_adv)
    grid[0].axis('off')    
    

    if name == 'FW':
       grid[0].set_title( 'FWnucl' + ' Perturbed', fontsize=20)
    elif name == 'pgdnucl':
       grid[0].set_title( 'PGDnucl' + ' Perturbed', fontsize=20)
    else:
       grid[0].set_title(label_tgt, fontsize=12)

    imc = grid[1].imshow(np.sum(np.abs(x_grad), axis=2), cmap='hot', interpolation='nearest')
    grid[1].axis('off')

 
    grid[1].set_title(model_title, fontsize=12)
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.00)
    #imc.set_clim(vmin=0, vmax=0.06)
    #imc.set_clim(vmin=0, vmax=0.15)
    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    plt.margins(0,0)
    cbar.ax.tick_params(labelsize=14) 
    plt.tight_layout(h_pad=0) 
    figname =  'simba/' + name + '.png' 
    print("figname: ", figname)
    #plt.show()
    plt.savefig(figname, bbox_inches='tight',pad_inches = 0)
    #plt.savefig('image' + name + args.file + '.png')
    #plt.show()
    plt.close()


def _pgd_whitebox1(model,
                  inputs,
                  y,
                  epsilon,
                  num_steps,
                  step_size, 
                  sum_dir = None):


        x = inputs.detach()
        if args.random:
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, 0, 1)
    
        return x

def fgsm(model_src, model, 
         X,
         y,
         epsilon=args.epsilon):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X.requires_grad = True
    with torch.enable_grad():
        loss = F.cross_entropy(model(normalize(X.cuda())).cpu(), y) - F.cross_entropy(model_src(normalize(X.cuda())).cpu(), y)
        #loss = - F.cross_entropy(model_src(X.cuda()).cpu(), y)
    loss.backward()
    # signed gradient
    eta = args.epsilon * X.grad.detach().sign()
    # Add perturbation to original example to obtain adversarial example
    return eta

# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>
def dct_attack_batch(model_src, model, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, order='rand', targeted=False, pixel_attack=False, log_every=1):
    batch_size = images_batch.size(0)
    image_size = images_batch.size(2)
    # sample a random ordering for coordinates independently per batch element
    if order == 'rand':
        indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
    elif order == 'diag':
        indices = utils.diagonal_order(image_size, 3)[:max_iters]
    elif order == 'strided':
        indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
    else:
        indices = utils.block_order(image_size, 3)[:max_iters]
    if order == 'rand':
        expand_dims = freq_dims
    else:
        expand_dims = image_size
    n_dims = 3 * expand_dims * expand_dims
    x = torch.zeros(batch_size, n_dims)
    # logging tensors
    probs = torch.zeros(batch_size, max_iters)
    succs = torch.zeros(batch_size, max_iters)
    queries = torch.zeros(batch_size, max_iters)
    l2_norms = torch.zeros(batch_size, max_iters)
    linf_norms = torch.zeros(batch_size, max_iters)
    prev_probs = get_probs(model, images_batch, labels_batch)
    preds = get_preds(model, images_batch)
    preds_src = get_preds(model_src, images_batch)
    if pixel_attack:
        trans = lambda z: z
    else:
        trans = lambda z: utils.block_idct(z, block_size=image_size)
    remaining_indices = torch.arange(0, batch_size).long()



    for k in range(max_iters):
        dim = indices[k]
        expanded = (images_batch[remaining_indices] + trans(expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
        perturbation = trans(expand_vector(x, expand_dims))
        l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
        linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
        preds_next = get_preds(model, expanded)
        preds[remaining_indices] = preds_next
        preds_next_src = get_preds(model_src, expanded)
        preds_src[remaining_indices] = preds_next_src
        if targeted:
            remaining = preds.ne(labels_batch) or preds_src.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch) or preds_src.ne(labels_batch)
        # check if all images are misclassified and stop early
        if remaining.sum() == 0:
            adv = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
            probs_k = get_probs(model, adv, labels_batch)
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(args.batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(args.batch_size, max_iters - k)
            break
        remaining_indices = torch.arange(0, batch_size)[remaining].long()
        if k > 0:
            succs[:, k-1] = ~ remaining
        diff = torch.zeros(remaining.sum(), n_dims)
        #diff = epsilon
        #diff[:, dim] = epsilon
        #diff =   epsilon * get_src_grad(model_src, expanded[remaining_indices], labels_batch[remaining_indices]).sign().view(-1, n_dims)
        diff = fgsm(model_src, model, expanded[remaining_indices], labels_batch[remaining_indices]).view(-1, n_dims)
        #diff =  diff.view(-1, n_dims)
        #diff /= torch.norm(diff, p=2, dim=1)
        #diff *= epsilon
        #diff = epsilon * get_tgt_fdm_grad(model, expanded[remaining_indices], labels_batch[remaining_indices], expand_vector(diff, expand_dims))
        diff =  diff.view(-1, n_dims)
        left_vec = x[remaining_indices] - diff
        right_vec = x[remaining_indices] + diff
        # trying negative direction
        adv = (images_batch[remaining_indices] + trans(expand_vector(left_vec, expand_dims))).clamp(0, 1)
        left_probs = get_probs(model, adv, labels_batch[remaining_indices])
        left_preds_src = get_preds(model_src, adv)
        queries_k = torch.zeros(batch_size)
        # increase query count for all images
        queries_k[remaining_indices] += 1
        if targeted:
            improved = left_probs.gt(prev_probs[remaining_indices]) and left_preds_src.eq(labels_batch[remaining_indices])
            #improved = left_preds_src.eq(labels_batch[remaining_indices])
        else:
            improved = left_probs.lt(prev_probs[remaining_indices]) and left_preds_src.eq(labels_batch[remaining_indices])
            #improved = left_preds_src.eq(labels_batch[remaining_indices])
        # only increase query count further by 1 for images that did not improve in adversarial loss
        if improved.sum() < remaining_indices.size(0):
            queries_k[remaining_indices[~improved]] += 1
        # try positive directions
        adv = (images_batch[remaining_indices] + trans(expand_vector(right_vec, expand_dims))).clamp(0, 1)
        right_probs = get_probs(model, adv, labels_batch[remaining_indices])
        right_preds_src = get_preds(model_src, adv)
        if targeted:
            right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs)) and right_preds_src.eq(labels_batch[remaining_indices])
            #right_improved = right_preds_src.eq(labels_batch[remaining_indices]) and ~improved
        else:
            right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs)) and right_preds_src.eq(labels_batch[remaining_indices])
            #right_improved = right_preds_src.eq(labels_batch[remaining_indices]) and ~improved
        probs_k = prev_probs.clone()
        # update x depending on which direction improved
        if improved.sum() > 0:
            left_indices = remaining_indices[improved]
            left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
            x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
            probs_k[left_indices] = left_probs[improved]
        if right_improved.sum() > 0:
            right_indices = remaining_indices[right_improved]
            right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
            x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
            probs_k[right_indices] = right_probs[right_improved]
        if right_improved.sum() == 0 and improved.sum() == 0:
            not_improved = ~improved
            not_indices = remaining_indices[not_improved]
            not_mask_remaining = not_improved.unsqueeze(1).repeat(1, n_dims)
            x[not_indices] = right_vec[not_mask_remaining].view(-1, n_dims)
            probs_k[not_indices] = right_probs[not_improved] 
        probs[:, k] = probs_k
        queries[:, k] = queries_k
        prev_probs = probs[:, k]
        print("left improved: ", improved.sum(), "right improved: ", right_improved.sum())
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f, labels_batch %d, src prediction %d' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean(), labels_batch.data, preds_src.data))
    expanded = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
    preds = get_preds(model, expanded)  
    preds_src = get_preds(model_src, expanded)  
    if targeted:
        remaining = preds.ne(labels_batch) or preds_src.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch) or preds_src.ne(labels_batch)
    succs[:, max_iters-1] = ~ remaining
    return expanded, probs, succs, queries, l2_norms, linf_norms


# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)

# load source models and dataset
model_src = getattr(models, args.model_src)(pretrained=True).cuda()
model_src.eval()

if args.model.startswith('inception'):
    image_size = 299
    testset = dset.ImageFolder(args.data_root + '/ILSVRC2012_img_val', utils.INCEPTION_TRANSFORM)
else:
    image_size = 224
    testset = dset.ImageFolder(args.data_root + '/ILSVRC2012_img_val', utils.IMAGENET_TRANSFORM)

# load sampled images or sample new ones
# this is to ensure all attacks are run on the same set of correctly classified images
batchfile = '%s/fgsm_white_agents_images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    images = torch.zeros(args.num_runs, 3, image_size, image_size)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    hard_label = labels + 1
    while preds.ne(labels).sum() > 0 or hard_label.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        print("idx", idx)
        for i in list(idx):
            print("i", i, labels[i], preds[idx])
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model_src, images[idx], 'imagenet', batch_size=args.batch_size)
    torch.save({'images': images, 'labels': labels}, batchfile)

if args.order == 'rand':
    n_dims = 3 * args.freq_dims * args.freq_dims
else:
    n_dims = 3 * image_size * image_size
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
print("Hello **********************************", N)

src_tgt_models = ['resnet18', 'resnet50', 'googlenet', 'densenet121']

for src_idx, mod_src_str in enumerate(src_tgt_models):
  model_src = getattr(models, mod_src_str)(pretrained=True).cuda()
  model_src.eval()   
  for tgt_idx, mod_tgt_str in enumerate(src_tgt_models):
   model = getattr(models, mod_tgt_str)(pretrained=True).cuda()
   model.eval()
   if mod_src_str == mod_tgt_str:
    continue
   for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    # replace true label with random target labels in case of targeted attack
    if args.targeted:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(1000 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
    tgt_label, src_label, batch_label_str = get_labels(model, model_src, images, labels_batch)
    print("tgt label: ", tgt_label, "src label: ", src_label, "batch label: ", batch_label_str)

    x, probs, succs, queries, l2_norms, linf_norms = dct_attack_batch(model_src,
        model, images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, order=args.order,
        targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
    print('Source model %s *************target model %s**' %(mod_src_str, mod_tgt_str))
    
    if i == 0:
        all_vecs = x
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
    else:
        all_vecs = torch.cat([all_vecs, x], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    if args.pixel_attack:
        prefix = 'pixel'
    else:
        prefix = 'dct'
    if args.targeted:
        prefix += '_targeted'  
      
    savefile = '%s/fgsm_white_%s_%s_to_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        args.result_dir, prefix, mod_src_str, mod_tgt_str, args.num_runs, args.num_iters, args.freq_dims, args.epsilon, args.order, args.save_suffix)
    torch.save({'original': images, 'vecs': all_vecs, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)
    print("labels_batch: ", labels_batch)
    tgt_label, src_label, batch_label_str = get_labels(model, model_src, x, labels_batch)
    name = '%s_%s_to_%s_simba_fgsm_%s_to_%s_eps_%s' %( str(i), mod_src_str, mod_tgt_str,  src_label, tgt_label, args.epsilon)
    print("tgt label: ", tgt_label, "src label: ", src_label, "batch label: ", batch_label_str)
    x_grad = x.detach() - images_batch.detach()
    plotting_name(images_batch.detach(), x.detach(), x_grad, args.epsilon, name, src_label, src_label + ' '+ '--> ' + tgt_label, mod_src_str + ' '+ '--> ' + mod_tgt_str)
      
    plot_histogram_name(x_grad.squeeze(0), name, str(i))

