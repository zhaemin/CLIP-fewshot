import os
import clip

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as Fu

from PIL import Image
from tqdm import tqdm
import numpy as np

def soft_target_cross_entropy(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=1)   # [B, C]
    loss = -torch.sum(target_probs * log_probs, dim=1)  # [B]
    return loss.mean()

def save_multi_crops(org_img, multi_crops, save_dir="multi_crop_output", prefix="img"):
    os.makedirs(save_dir, exist_ok=True)

    mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(3, 1, 1).cuda()
    std = torch.tensor([0.2686, 0.2613, 0.2758]).view(3, 1, 1).cuda()
    
    for i, crop in enumerate(multi_crops):
        img = crop.clone()
        if img.size(0) == 3:
            img = img * std + mean  # Unnormalize
        img = torch.clamp(img, 0, 1)
        img_pil = Fu.to_pil_image(img)

        save_path = os.path.join(save_dir, f"{prefix}_crop{i+1}.png")
        img_pil.save(save_path)

    org_img = org_img * std + mean  # Unnormalize
    org_img = torch.clamp(org_img, 0, 1)
    org_img_pil = Fu.to_pil_image(org_img)

    save_path = os.path.join(save_dir, f"{prefix}_org.png")
    org_img_pil.save(save_path)


def log(msg, log_file="log.txt"):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# code from https://github.com/clovaai/CutMix-PyTorch
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, target, beta=1.0, cutmix_prob=1.0):
    r = np.random.rand(1)
    input = images.clone().detach()
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam



# code from https://github.com/facebookresearch/mixup-cifar10
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc


def classwise_accuracy(output, target, num_classes, topk=1):
    pred = output.topk(topk, dim=1, largest=True, sorted=True)[1]  # (N, topk)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))  # (N, topk)

    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for i in range(target.size(0)):
        label = target[i].item()
        class_total[label] += 1
        if correct[i].any():
            class_correct[label] += 1

    class_acc = 100 * class_correct / (class_total + 1e-8)
    return class_acc.cpu().numpy()



def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target, multi_crops) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels

