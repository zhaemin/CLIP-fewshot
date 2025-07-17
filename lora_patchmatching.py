import torch
from torch import nn
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from statistics import mean



def sinkhorn_batch(out, epsilon=0.05, n_iters=3):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py
    Q = torch.exp(out / epsilon).transpose(1, 2) # Q is K-by-B for consistency with notations from our paper  => bs num_fine_weights N 
    B = Q.shape[2] # number of samples to assign -> N
    K = Q.shape[1] # how many prototypes -> 4

    # make the matrix sums to 1
    sum_Q = torch.sum(Q, dim=(1,2), keepdim=True)
    Q /= sum_Q

    for it in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q.transpose(1, 2)

def compute_entropy(logits):
    logits = logits.to(torch.float32)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

class PatchMatchingLoRA(nn.Module):
    def __init__(self, num_classes, text_dim, noise_scale, num_fine_weights):
        super(PatchMatchingLoRA, self).__init__()
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.num_fine_weights = num_fine_weights
        self.fine_weights = nn.Parameter(torch.randn(self.num_classes, num_fine_weights, self.text_dim, device='cuda') * noise_scale)

    def patch_matching_target(self, image_features, text_features, target):
        text_features_c = [torch.stack(list(text_features[t] + self.fine_weights[t][i] for i in range(self.num_fine_weights))) for t in target] # bs 4 dim
        text_features_c = torch.stack(text_features_c)
        
        B, N, D = image_features.shape
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            scores = torch.einsum('BND, BKD -> BNK', image_features, text_features_c)
        '''
        new_scores = scores.argmax(dim=-1)
        _, cnt_scores = torch.unique(new_scores[0], return_counts=True)
        print(cnt_scores)
        '''
        with torch.no_grad():
            fine_label = sinkhorn_batch(scores) # bs N num_fine_weights
        fine_target = fine_label.argmax(dim=-1) # bs N
        '''
        _, cnt = torch.unique(fine_target[0], return_counts=True)
        print(cnt)
        '''
        target_for_patch = target.unsqueeze(1).repeat(1, N).view(-1)
        
        fine_target = fine_target.view(-1) * self.num_classes + target_for_patch #bs*N
        return fine_target
    
    
    def evaluate_lora(self, args, clip_model, loader, dataset):
        clip_model.eval()
        with torch.no_grad():
            template = dataset.template[0] 
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

            text_queue = torch.zeros(self.num_classes * self.num_fine_weights, self.text_dim, dtype=torch.float16).cuda()
            for i in range(self.num_fine_weights):
                fine_weights_per_class = torch.stack([text_features[j] + self.fine_weights[j][i] for j in range(len(text_features))])
                text_queue[i * self.num_classes : (i+1) * self.num_classes] = fine_weights_per_class

        acc = 0.
        patch_acc = 0.
        mean_acc = 0.
        tot_samples = 0
        total_patch_entropy = 0.
        
        with torch.no_grad():
            for i, (images, target, multi_crops) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)

                cls_features = image_features[:, 0]
                patch_features = image_features[:, 1:]
                B, N, D = patch_features.shape
                patch_features = patch_features.reshape(-1, D) # bs*N D

                cls_cosine_similarity = cls_features @ text_features.t()
                
                '''
                patch_cosine_similarity = patch_features @ text_features.t()
                patch_class_probs = patch_cosine_similarity.view(B, N, self.num_classes)
                patch_prediction_target = F.one_hot(patch_class_probs.argmax(dim=-1), self.num_classes).float()
                patch_class_probs = patch_prediction_target.mean(dim=1) # B
                '''
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    patch_cosine_similarity = patch_features @ text_queue.t()

                patch_predictions_fine_target = patch_cosine_similarity.argmax(dim=-1) # B*N
                patch_predictions_real_target = patch_predictions_fine_target.view(B, N) % self.num_classes #B N
                one_hot = F.one_hot(patch_predictions_real_target, self.num_classes).float()
                patch_class_probs = one_hot.mean(dim=1) # B #classes
                 
                cls_probs = cls_cosine_similarity.softmax(dim=-1)
                patch_probs = patch_class_probs.softmax(dim=-1)
                mean_probs = (cls_probs + patch_probs) / 2
                
                mean_acc += cls_acc(mean_probs, target) * len(mean_probs)
                acc += cls_acc(cls_cosine_similarity, target) * len(cls_cosine_similarity)
                patch_acc += cls_acc(patch_class_probs, target) * len(patch_class_probs)

                tot_samples += len(cls_cosine_similarity)
        acc /= tot_samples
        patch_acc /= tot_samples
        mean_acc /= tot_samples
        total_patch_entropy /= len(loader)
        print(total_patch_entropy)
        return acc, patch_acc, mean_acc
    
    def load_lora_(self, args, clip_model):
        list_lora_layers = apply_lora(args, clip_model)
        clip_model = clip_model.cuda()
        load_lora(args, list_lora_layers)

    def run_lora(self, args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
        
        VALIDATION = False
        
        # Textual features
        print("\nGetting textual features as CLIP's classifier.")
        textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
        
        list_lora_layers = apply_lora(args, clip_model)
        clip_model = clip_model.cuda() 
        
        if args.eval_only:
            load_lora(args, list_lora_layers)
            acc_test = self.evaluate_lora(args, clip_model, test_loader, dataset)
            log(acc_test)
            return

        mark_only_lora_as_trainable(clip_model)
        total_iters = args.n_iters * args.shots
        
        optimizer = torch.optim.AdamW(get_lora_parameters(clip_model)+[self.fine_weights], weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
        
        best_acc_val, best_acc_test = 0., 0.
        best_epoch_val = 0
        
        # training LoRA
        scaler = torch.cuda.amp.GradScaler()
        count_iters = 0
        finish = False
        while count_iters < total_iters:
            clip_model.train()
            acc_train = 0
            tot_samples = 0
            loss_epoch = 0.
            if args.encoder == 'vision': 
                text_features = textual_features.t().half()

            for i, (images, target, multi_crops) in enumerate(tqdm(train_loader)):
                
                template = dataset.template[0]
                texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
                images, target = images.cuda(), target.cuda()

                B = images.size(0)
                
                if args.encoder == 'text' or args.encoder == 'both':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = clip.tokenize(texts).cuda()
                        class_embeddings = clip_model.encode_text(texts)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                    
                if args.encoder == 'vision' or args.encoder == 'both':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model.encode_image(images)
                            
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                cls_features = image_features[:, 0, :]
                cls_cosine_similarity = logit_scale * cls_features @ text_features.t()
                

                patch_features = image_features[:, 1:, :]
                B, N, D = patch_features.shape

                fine_target = self.patch_matching_target(patch_features, text_features, target) #bs*N 
                text_queue = torch.zeros(self.num_classes * self.num_fine_weights, self.text_dim, dtype=torch.float16).cuda()
                for i in range(self.num_fine_weights):
                    fine_weights_per_class = torch.stack([text_features[j] + self.fine_weights[j][i] for j in range(len(text_features))])
                    text_queue[i * self.num_classes : (i+1) * self.num_classes] = fine_weights_per_class
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    patch_cosine_similarity = logit_scale * patch_features.reshape(-1, D) @ text_queue.t()
            
                loss = F.cross_entropy(cls_cosine_similarity, target) + F.cross_entropy(patch_cosine_similarity, fine_target) / N
                
                acc_train += cls_acc(cls_cosine_similarity, target) * target.shape[0]
                loss_epoch += loss.item() * target.shape[0]
                tot_samples += target.shape[0]

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                scheduler.step()
                
                count_iters += 1
                
                if count_iters == total_iters:
                    break
        
            if count_iters < total_iters:
                acc_train /= tot_samples
                loss_epoch /= tot_samples
                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))
            
            # Eval
            if VALIDATION:
                clip_model.eval()
                acc_val, patch_acc_test = self.evaluate_lora(args, clip_model, val_loader, dataset)
                print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

        
        if args.save_path != None:
            save_lora(args, list_lora_layers)
            
        return
