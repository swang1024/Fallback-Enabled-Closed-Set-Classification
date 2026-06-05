import os
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import INaturalist

def train_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]) 

def test_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

'''
assume classes across domains are the same.
[0 1 ............................................................................ N - 1]
|---- common classes --||---- source private classes --||---- target private classes --|

|-------------------------------------------------|
|                DATASET PARTITION                |
|-------------------------------------------------|
|DATASET    |  class split(com/sou_pri/tar_pri)   |
|-------------------------------------------------|
|DATASET    |    PDA    |    OSDA    |   UniDA    |
|-------------------------------------------------|
|Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
|-------------------------------------------------|
|OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
|-------------------------------------------------|
|VisDA-C    |           |   6/0/6    |   6/3/3    |
|-------------------------------------------------|  
|DomainNet  |           |            | 150/50/145 |
|-------------------------------------------------|
'''

class SFUniDADataset(Dataset):
    
    def __init__(self, args, data_dir, data_list, d_type, preload_flg=True) -> None:
        super(SFUniDADataset, self).__init__()
        
        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg
        
        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num 
        
        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        
        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10 for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir 
        self.data_list = [item.strip().split() for item in data_list]

        self.src_labels = []
        for item in self.data_list:
            if int(item[1]) in self.source_classes:
                lbl = item[0].split('/')[0].replace('_', ' ')
                if lbl not in self.src_labels:
                    self.src_labels.append(lbl)
        # print(self.src_labels)

        self.tgt_labels = {}
        for item in self.data_list:
            if int(item[1]) in self.target_classes:
                lbl = item[0].split('/')[0].replace('_', ' ')
                if int(item[1]) not in self.tgt_labels:
                    self.tgt_labels[int(item[1])] = lbl
        # print("target", self.tgt_labels.values())
        
        # Filtering the data_list
        if self.d_type == "source":
            # self.data_dir = args.source_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
        else:
            # self.data_dir = args.target_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        self.pre_loading()
        
    def pre_loading(self):
        if self.dataset == "Office" and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, "images", item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        elif self.dataset == "OfficeHome" and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        img_f, img_label = self.data_list[img_idx]
        if self.dataset ==  "Office" and self.preload_flg:
            img = self.img_list[img_idx]
        elif self.dataset == "OfficeHome":
            if self.preload_flg:
                img = self.img_list[img_idx]
            else:
                img = os.path.join(self.data_dir, img_f)
        else:  
            img = os.path.join(self.data_dir, img_f) 
        return img, img_label
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        img, img_label = self.load_img(img_idx)
        
        if self.d_type == "source":
            img_label = int(img_label)
        else:
            # img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)
            img_label = int(img_label)
        
        # img_train = self.train_transform(img)
        # img_test = self.test_transform(img)
        img_train = img

        if img_label in self.shared_classes:
            private = False
        else:
            private = True
        return ''.join(img_train), img_label, img_idx, ''.join(self.tgt_labels[img_label]), private
        # return ''.join(img_train) 


class SFUniDADataset_BLIP(Dataset):
    
    def __init__(self, args, data_dir, data_list, d_type, preload_flg=True) -> None:
        super(SFUniDADataset_BLIP, self).__init__()
        
        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg
        
        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num 
        
        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        
        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10 for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir 
        self.data_list = [item.strip().split() for item in data_list]

        self.src_labels = []
        for item in self.data_list:
            if int(item[1]) in self.source_classes:
                lbl = item[0].split('/')[0].replace('_', ' ')
                if lbl not in self.src_labels:
                    self.src_labels.append(lbl)
        # print(self.src_labels)

        self.tgt_labels = {}
        for item in self.data_list:
            if int(item[1]) in self.target_classes:
                lbl = item[0].split('/')[0].replace('_', ' ')
                if int(item[1]) not in self.tgt_labels:
                    self.tgt_labels[int(item[1])] = lbl
        print(self.tgt_labels)
        
        # Filtering the data_list
        if self.d_type == "source":
            # self.data_dir = args.source_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
        else:
            # self.data_dir = args.target_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        self.pre_loading()
        
    def pre_loading(self):
        if self.dataset == "Office" and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, "images", item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        elif self.dataset == "OfficeHome" and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        img_f, img_label = self.data_list[img_idx]
        if self.dataset ==  "Office" and self.preload_flg:
            img = self.img_list[img_idx]
        elif self.dataset == "OfficeHome":
            if self.preload_flg:
                img = self.img_list[img_idx]
            else:
                img = os.path.join(self.data_dir, img_f)
        else:  
            img = os.path.join(self.data_dir, img_f) 
        return img, img_label
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        img, img_label = self.load_img(img_idx)
        
        if self.d_type == "source":
            img_label = int(img_label)
        else:
            # img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)
            img_label = int(img_label)
        
        # img_train = self.train_transform(img)
        # img_test = self.test_transform(img)
        img_train = img

        if img_label in self.shared_classes:
            private = False
        else:
            private = True
        return ''.join(img_train), img_label, img_idx, ''.join(self.tgt_labels[img_label]), private


class INaturalist_UniDA(INaturalist):
    
    def __init__(self, root, *, shared_classes=None, source_private_classes=None, target_private_classes=None, label_names_list=None, **kwargs):
        super().__init__(root=root, **kwargs)
        # self.shared_class_num = args.shared_class_num
        # self.source_private_class_num = args.source_private_class_num
        # self.target_private_class_num = args.target_private_class_num 

        # self.shared_classes = [i for i in range(args.shared_class_num)]
        # self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        # self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
        # source_classes stores the number label to all the source classes
        # target_classes stores the number label to all the target classes
        # data_list stores the path and label to all the images of source / target domain
        # src_labels stores all the name labels of source
        # tgt_labels store all the name labels of target

        self.shared_class_num = len(shared_classes)
        self.source_private_class_num = len(source_private_classes)
        self.target_private_class_num = len(target_private_classes)

        self.shared_classes = shared_classes
        self.source_private_classes = source_private_classes
        self.target_private_classes = target_private_classes

        self.source_classes = shared_classes + source_private_classes
        self.target_classes = shared_classes + target_private_classes

        self.src_labels = [label_names_list[i] for i in self.source_classes]
        self.tgt_labels = {}
        for i in self.target_classes:
            self.tgt_labels[i] = label_names_list[i]
        
        # Build a filtered list of indices to use for this dataset
        self._filtered_indices = []
        for idx in range(len(self.index)):
            cat_id, fname = self.index[idx]
            label = self.categories_map[cat_id][self.target_type[0]]
            if label not in self.source_private_classes:
                self._filtered_indices.append((cat_id, fname))

    def __len__(self):
        return len(self._filtered_indices)

    def __getitem__(self, index):
        # img, label = super().__getitem__(index)
        cat_id, fname = self._filtered_indices[index]
        image_path = os.path.join(self.root, self.all_categories[cat_id], fname)
        img = self.loader(image_path) if self.loader is not None else Image.open(image_path)
 
        target = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        img_label = int(target)
        
        img_train = image_path

        if img_label in self.shared_classes:
            private = False
        else:
            private = True
        
        return ''.join(img_train), img_label, index, ''.join(self.tgt_labels[img_label]), private
