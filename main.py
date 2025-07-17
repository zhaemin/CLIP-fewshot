import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader, ImageFolderWrapper

from utils import *
from run_utils import *
from lora import run_lora 
from lora_patchmatching import PatchMatchingLoRA


def main():
    # Load config file
    args = get_arguments()
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    if args.dataset == 'imagenet' or args.dataset == 'sun397':
        val = ImageFolderWrapper(data_source=dataset.val, input_size=224,transform=preprocess, is_train=False)
        test = ImageFolderWrapper(data_source=dataset.test, input_size=224,transform=preprocess, is_train=False)
        val_loader = torch.utils.data.DataLoader(val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        if args.dataset == 'imagenet' or args.dataset == 'sun397':
            train_x = ImageFolderWrapper(data_source=dataset.train_x, input_size=224,transform=train_tranform, is_train=True, multi_crops=args.multi_crops)
            train_loader = torch.utils.data.DataLoader(train_x, batch_size=args.batch_size // 2, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8, multi_crops=args.multi_crops)
    
    if args.patch_matching:
        model = PatchMatchingLoRA(len(dataset.classnames), text_dim=512, noise_scale=args.noise_scale, num_fine_weights=args.num_fine_weights)
        model.cuda()

        if args.eval_only:
            model.load_state_dict(torch.load(f'{args.save_path}/vitb16/{args.dataset}/{args.shots}shots/seed{args.seed}_patchmatching_{args.noise_scale}/model.pt', map_location='cuda'))
            model.load_lora_(args, clip_model)
        else:
            log(f'{args.dataset}_{args.shots}_nfw{args.num_fine_weights}_results', log_file=f"log/log_patchlevel_ns{args.noise_scale}.txt")
            model.run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

            backbone = args.backbone.replace('/', '').replace('-', '').lower()
            save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}_patchmatching_{args.noise_scale}'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/model.pt"
            torch.save(model.state_dict(), save_path)

        acc_test, patch_acc_test, mean_acc_test = model.evaluate_lora(args, clip_model, test_loader, dataset)
        log("**** Final test accuracy(cls): {:.2f}.  // (patch): {:.2f}. // (mean): {:.2f} ****\n".format(acc_test, patch_acc_test, mean_acc_test), log_file=f"log/log_patchlevel_ns{args.noise_scale}.txt")

        
    else:
        log(f'{args.dataset}_{args.shots}_results', log_file=f"log/log_{args.aug_test}_mc{args.multi_crops}_seed{args.seed}.txt")
        run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()