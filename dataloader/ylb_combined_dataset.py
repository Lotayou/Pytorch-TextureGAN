import torch.utils.data as data

from PIL import Image
import glob
import os
import os.path as osp
import random

import torch

import torchvision.transforms as TVtrans


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(directory):
    classes = [d for d in os.listdir(directory) if osp.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(directory, opt, erode_seg=True):
    # opt: 'train' or 'val'
    img = glob.glob(osp.join(directory, opt + '_img/*/*.jpg'))
    img = sorted(img)
    skg = glob.glob(osp.join(directory, opt + '_skg/*/*.jpg'))
    skg = sorted(skg)
    seg = glob.glob(osp.join(directory, opt + '_seg/*/*.jpg'))
    seg = sorted(seg)
    txt = glob.glob(osp.join(directory, opt + '_txt/*/*.jpg'))
    #txt = glob.glob(osp.join(directory, opt + '_dtd_txt/*/*.jpg'))
    extended_txt = []
    #import pdb; pdb.set_trace()
    for i in range(len(skg)):
        extended_txt.append(txt[i%len(txt)])
    random.shuffle(extended_txt)
    

    if erode_seg:
        eroded_seg = glob.glob(osp.join(directory, 'eroded_' + opt + '_seg/*/*.jpg'))
        eroded_seg = sorted(eroded_seg)
        return list(zip(img, skg, seg , eroded_seg, extended_txt))
    else:
        return list(zip(img, skg, seg, extended_txt))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)


class GroundTruthImageFolder(data.Dataset):
    def __init__(self, phase, root, transform=None, target_transform=None,
                 loader=default_loader, erode_seg=True, max_dataset_size=-1):
     
        self.root = os.path.join(root, phase + '_images')
        self.imgs = [os.path.join(self.root, item) 
            for item in os.listdir(self.root)
            if is_image_file(item)
        ]
        self.imgs.sort()
        if 0 < max_dataset_size < len(self.imgs):
            self.imgs = self.imgs[:max_dataset_size]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.erode_seg = erode_seg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        combined = self.loader(self.imgs[index])
        w, h = combined.size
        left_bbox = [0, 0, h, h]
        mid_bbox = [h, 0, 2*h, h]
        right_bbox = [2*h, 0, 3*h, h]
        full = combined.crop(left_bbox)
        mask = combined.crop(mid_bbox)
        partial = combined.crop(right_bbox)
        if self.transform is not None:
            mask_lab, full_lab, partial_lab = self.transform([mask, full, partial])
            
        data_dict = {
            'mask': mask_lab[0:1],
            'partial_lab': partial_lab,
            'gt_lab': full_lab,
        }
        #for k,v in data_dict.items():
        #    print(k, v.shape, v.max(), v.min())
        return data_dict

    def __len__(self):
        return len(self.imgs)
        
        
class ExternalTextureFolder(data.Dataset):
    def __init__(self, phase, root, texture_root, transform=None, 
                 loader=default_loader, texture_size=512, max_dataset_size=-1):
        self.root = os.path.join(root, phase + '_images')
        self.imgs = [os.path.join(self.root, item) 
            for item in os.listdir(self.root)
            if is_image_file(item)
        ]
        self.imgs.sort()
        if 0 < max_dataset_size < len(self.imgs):
            self.imgs = self.imgs[:max_dataset_size]
            
        texture_types = os.listdir(texture_root)
        self.textures = []
        for type in texture_types:
            subfolder = os.path.join(texture_root, type)
            self.textures += [
                os.path.join(subfolder, item)
                for item in os.listdir(subfolder)
                if is_image_file(item)
            ]    
        self.textures.sort()
        self.texture_num = len(self.textures)
        print('%d external sample texture images loaded' % self.texture_num)
        
        self.transform = transform
        self.loader = loader
        self.texture_size = texture_size
        self.default_center_crop = TVtrans.CenterCrop(self.texture_size)
        self.default_scale = TVtrans.Resize(self.texture_size)
        
    def __getitem__(self, index):
        combined = self.loader(self.imgs[index])
        w, h = combined.size
        left_bbox = [0, 0, h, h]
        mid_bbox = [h, 0, 2*h, h]
        right_bbox = [2*h, 0, 3*h, h]
        full = combined.crop(left_bbox)
        mask = combined.crop(mid_bbox)
        partial = combined.crop(right_bbox)
        
        rand_id = random.randint(0, self.texture_num-1)
        texture = self.loader(self.textures[rand_id])
        wt, ht = combined.size
        # expect raw texture images have different size
        # normalize the size for batching and torchify
        
        if min(wt, ht) < self.texture_size:
            tmp_center_crop = TVtrans.CenterCrop(min(wt,ht) - 10)
            texture = self.default_resize(tmp_center_crop(texture))
        else:
            texture = self.default_center_crop(texture)
            
        if self.transform is not None:
            mask_lab, full_lab, partial_lab, texture_lab = \
                self.transform([mask, full, partial, texture])
        
        data_dict = {
            'mask': mask_lab[0:1],
            'partial_lab': partial_lab,
            'gt_lab': full_lab,
            'texture_lab': texture_lab,
        }
        #for k,v in data_dict.items():
        #    print(k, v.shape, v.max(), v.min())
        return data_dict
        
    def __len__(self):
        return len(self.imgs)
