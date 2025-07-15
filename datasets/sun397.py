import os
import random
from collections import defaultdict

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class SUN397():

    dataset_dir = 'sun397'
    template = ['a photo of a {}.']

    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        if train_preprocess is None:
            train_preprocess = transforms.Compose([
                                                    transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                                ])
        
        if test_preprocess is None:
            test_preprocess = preprocess
        
        self.train_x = datasets.ImageFolder(os.path.join(os.path.join(self.dataset_dir, 'train')))
        self.val = datasets.ImageFolder(os.path.join(os.path.join(self.dataset_dir, 'train')))
        self.test = datasets.ImageFolder(os.path.join(os.path.join(self.dataset_dir, 'val')))
        
        self.classnames = self.train_x.classes

        num_shots_val = min(4, num_shots)

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train_x.imgs)):
            split_by_label_dict[self.train_x.targets[i]].append(self.train_x.imgs[i])
        imgs = []
        targets = []
        imgs_val = []
        targets_val = []
        for label, items in split_by_label_dict.items():
            samples = random.sample(items, num_shots + num_shots_val)
            imgs = imgs + samples[0:num_shots]
            imgs_val = imgs_val + samples[num_shots:num_shots+num_shots_val]
            targets = targets + [label for i in range(num_shots)]
            targets_val = targets_val + [label for i in range(num_shots_val)]
            
        self.train_x.imgs = imgs
        self.train_x.targets = targets
        self.train_x.samples = imgs
        
        self.val.imgs = imgs_val
        self.val.targets = targets_val
        self.val.samples = imgs_val

'''
class SUN397(DatasetBase):

    dataset_dir = 'sun397'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_SUN397.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:] # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split('/')[1:] # remove 1st letter
                names = names[::-1] # put words like indoor/outdoor at first
                classname = ' '.join(names)
                
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items
'''