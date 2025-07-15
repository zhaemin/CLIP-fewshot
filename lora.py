import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from statistics import mean



def cls_acc_aug_hit(logits, y_a, y_b):
    pred = logits.argmax(dim=1)
    correct = (pred == y_a) | (pred == y_b)
    return correct.float().mean().item()


def evaluate_aug_mix(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    org_acc = 0.
    aug_acc = 0.

    org_acc_top2 = 0.
    aug_acc_top2 = 0.

    aug_acc_hit = 0.

    tot_samples = 0
    with torch.no_grad():
        for i, (images, target, _) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            B = images.size(0)

            if args.aug_test == 'mixup':
                aug_images, target_a, target_b, lam = mixup_data(images, target)
                y_a = F.one_hot(target_a, len(dataset.classnames)).half()
                y_b = F.one_hot(target_b, len(dataset.classnames)).half()
                aug_onehot = lam * y_a + (1 - lam) * y_b
                images = torch.cat((images, aug_images), dim=0)
            else: # cutmix
                aug_images, target_a, target_b, lam = cutmix(images, target)
                y_a = F.one_hot(target_a, len(dataset.classnames)).half()
                y_b = F.one_hot(target_b, len(dataset.classnames)).half()
                aug_onehot = lam * y_a + (1 - lam) * y_b
                images = torch.cat((images, aug_images), dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = image_features @ text_features.t()

            org_acc += cls_acc(cosine_similarity[:B], target) * B
            org_acc_top2 += cls_acc(cosine_similarity[:B], target, topk=2) * B

            aug_acc += cls_acc(cosine_similarity[B:], aug_onehot.argmax(dim=1)) * B
            aug_acc_top2 += cls_acc(cosine_similarity[B:], aug_onehot.argmax(dim=1), topk=2) * B
            aug_acc_hit += cls_acc_aug_hit(cosine_similarity[B:], target_a, target_b) * 100

            tot_samples += B

    org_acc /= tot_samples
    aug_acc /= tot_samples

    org_acc_top2 /= tot_samples
    aug_acc_top2 /= tot_samples

    aug_acc_hit /= len(loader)

    return f'org acc: {round(org_acc,2)}, aug acc: {round(aug_acc,2)}, org_acc_top2: {round(org_acc_top2, 2)}, aug_acc_top2: {round(aug_acc_top2, 2)}, aug_acc_hit: {round(aug_acc_hit, 2)}'



def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc

def get_crop_features(multi_crops, clip_model):
    multi_crops = multi_crops.cuda() # bs 6 3 96 96
    multi_crops = multi_crops.reshape(-1, 3, 224, 224) # bs*6 3 96 96
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            crop_features = clip_model.encode_image(multi_crops, multi_crops=False)
    crop_features = crop_features/crop_features.norm(dim=-1, keepdim=True)
    #for j in range(2):
    #    save_multi_crops(images[j], multi_crops[j], prefix=f"sample_{j}")
    return crop_features


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        #load_lora(args, list_lora_layers)
        acc_test = evaluate_aug_mix(args, clip_model, test_loader, dataset)
        log(acc_test)
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    #total_iters = 1
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
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

        local_accs = []
        global_accs = []

        for i, (images, target, multi_crops) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()

            B = images.size(0)

            # augmentations (mixup / cutmix)
            if args.aug_test == 'mixup':
                mixup_images, y_a, y_b, lam = mixup_data(images, target)
                images = torch.cat((images, mixup_images), dim=0)
            elif args.aug_test == 'cutmix':
                cutmix_images, y_a, y_b, lam = cutmix(images, target)
                images = torch.cat((images, cutmix_images), dim=0)
            
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
            cosine_similarity = logit_scale * image_features @ text_features.t()
           
            if args.aug_test != 'none':
                aug_loss = lam * F.cross_entropy(cosine_similarity[B:], y_a) + (1 - lam) * F.cross_entropy(cosine_similarity[B:], y_b)
                loss = F.cross_entropy(cosine_similarity[:B], target) + aug_loss
            else:
                loss = F.cross_entropy(cosine_similarity[:B], target)
            
            if args.multi_crops:
                crop_features = get_crop_features(multi_crops, clip_model)
                crop_logits = logit_scale * crop_features @ text_features.t() # bs*6 num_classses

                crop_logits = crop_logits.reshape(B, 6, -1)
                for i in range(6):
                    local_accs.append(cls_acc(crop_logits[:, i, :], target))
                
                global_accs.append(cls_acc(cosine_similarity, target))
                
                B, C = cosine_similarity.shape
                org_logits = cosine_similarity
                #org_logits = org_logits.expand(B, 6, C)  
                #org_logits = org_logits.reshape(B * 6, C)
                crop_logits = crop_logits.mean(dim=1)

                #log_probs_crop = F.log_softmax(crop_logits, dim=1)
                probs_org = F.softmax(org_logits, dim=1)
                
                #l2g_loss = F.kl_div(log_probs_crop, probs_org, reduction='batchmean')
                #loss += l2g_loss
                loss += soft_target_cross_entropy(crop_logits, probs_org)
                loss += F.cross_entropy(crop_logits, target)
                
            
            acc_train += cls_acc(cosine_similarity[:B], target) * target.shape[0]
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
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    if args.multi_crops:
        print(local_accs)
        print(mean(local_accs))
        print(mean(global_accs))
    
    acc_test = evaluate_aug_mix(args, clip_model, test_loader, dataset)
    log(acc_test, log_file=f"log/log_{args.aug_test}_mc{args.multi_crops}_seed{args.seed}.txt")
    #log("**** Final test accuracy: {:.2f}. ****\n".format(acc_test), log_file=f"log.txt")
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
