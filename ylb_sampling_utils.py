import torch
import numpy as np
from utils import transforms as custom_transforms
from utils.visualize import vis_image, vis_patch
from skimage.io import imread, imsave
from skimage import color
import math
import random
import os

SMPL_TORSO_BBOX_256 = [0, 112, 160, 256]  # a 144*160 box within 256*256
SMPL_TORSO_BBOX_512 = [0, 224, 320, 512]  # a 288*320 box within 512*512

class RandomPatchSampler():
    def __init__(self, input_size=512, loss_patch_size=64, segbox_path='./SMPL_uv_512.png'):
        assert input_size in [256, 512], 'Only support 256 or 512 input'
        self.torso_bbox = SMPL_TORSO_BBOX_512 \
            if input_size == 512 else SMPL_TORSO_BBOX_256
        self.loss_patch_size = loss_patch_size
        self.input_patch_range = [40,60]    # fix it for now.
        self.uv_mask = imread(segbox_path)  # uv region where patches are sampled within
        self.swap_flag = False
        
    def get_rand_local_bbox(self):
        '''
            sample random size bbox within SMPL_TORSO_BBOX
        '''
        low, high = self.input_patch_range
        rand_size = random.randint(low, high)
        rand_top = random.randint(self.torso_bbox[1], self.torso_bbox[3] - rand_size - 1)
        rand_left = random.randint(self.torso_bbox[0], self.torso_bbox[2] - rand_size - 1)
        
        return rand_left, rand_top, rand_left + rand_size, rand_top + rand_size
        
    def get_loss_patch(self, input_tensor, bbox_type):
        '''
            sample [patch_size*patch_size] squares within SMPL_TORSO_BBOX ^ self.uv_mask.
            need to retain as much white as possible 
            
            For input/output: sample within torso box
            For texture: sample anywhere (unless it's replaced image)
        '''
        if bbox_type == 'torso':
            bbox = self.torso_bbox
        else:
            bbox = [0, 0, input_tensor.shape[3], input_tensor.shape[2]]
        
        # num_loss_patches=1
        b,c,h,w = input_tensor.shape
        patch_tensor = input_tensor[:,:,:self.loss_patch_size,:self.loss_patch_size].clone()
        
        for i in range(b):
            x = random.randint(bbox[1], bbox[3] - self.loss_patch_size)
            y = random.randint(bbox[0], bbox[2] - self.loss_patch_size)
            patch_tensor[i] = input_tensor[i,:,x:x+self.loss_patch_size, y:y+self.loss_patch_size]
            
        return patch_tensor
        
    def add_patch(self, input, texture):
        '''
            Input: 4*h*w, torso_bbox empty
            Output: torso_bbox patch filled
        '''
        l, t, r, b = self.get_rand_local_bbox()
        input[0, t:b, l:r] = 1.
        # by default, texture and input are the same size.
        input[1:4, t:b, l:r] = texture[:, t:b, l:r]
        return input
        
    def gen_input_and_labels(self, data, swap_prob=0.5):            
        '''
            Stage 2, use external texture from DTD dataset

                1. We apply external texture guidance only over torso region, which is
            conviently located inside a rectangular bounding box at the bottom left
            of the UV map. For other places, still apply global pixel and ab loss.
                2. We pick a random square cropping box inside the torso bbox and 
            recompensate the texture map into it, making a combined binary mask
            
            Note: Only using external (unseen) textures with 1 - swap_prob.
        '''
        self.partial = custom_transforms.normalize_lab(data['partial_lab'])
        self.gt = custom_transforms.normalize_lab(data['gt_lab'])
        self.mask = custom_transforms.normalize_seg(data['mask'])
        if random.random() < swap_prob:
            #print('*************************\n*** swap\n*************************')
            self.swap_flag = True
            texture = data['gt_lab']
        else:
            self.swap_flag = False
            texture = data['texture_lab']
        self.texture = custom_transforms.normalize_lab(texture)  # should be 256*256
        
        input = torch.cat((self.mask.unsqueeze_(1), self.partial), dim=1)  # b, 4, h, w
        input[:, :, self.torso_bbox[1]: self.torso_bbox[3], 
            self.torso_bbox[0]: self.torso_bbox[2]] = 0.  # reset torso region to black
        input[:, 1, self.torso_bbox[1]: self.torso_bbox[3], 
            self.torso_bbox[0]: self.torso_bbox[2]] = -1. 
        
        self.input = torch.stack([
            self.add_patch(bi, bt) for (bi, bt) in zip(input, self.texture)
        ],dim=0)
        
        out_dict = {
            'input': self.input,
            'gt': self.gt,
            'texture': self.texture,
            'flag': self.swap_flag
        }
        
        return out_dict
            
    def visualize_input(self, folder):
        bundle = self.input[:,1:4]
        bundle = custom_transforms.denormalize_lab(bundle)
        bundle = bundle[0].detach().cpu().numpy().transpose(1,2,0)
        bundle = (255. * color.lab2rgb(bundle)).astype(np.uint8)
        imsave(os.path.join(folder, 'input_stage_II.png'), bundle)  # float to uint8 is taken care of
    
    def visualize_texture(self, folder):
        bundle = self.texture
        bundle = custom_transforms.denormalize_lab(bundle)
        bundle = bundle[0].detach().cpu().numpy().transpose(1,2,0)
        bundle = (255. * color.lab2rgb(bundle)).astype(np.uint8)
        imsave(os.path.join(folder, 'texture_stage_II.png'), bundle)  # float to uint8 is taken care of
    

if __name__ == '__main__':
    # SMPL UV map torso bounding box
    from PIL import Image
    #im = Image.open('SMPL_uv_512.png').convert('RGB')
    #cropped = im.crop(SMPL_TORSO_BBOX_512)
    #cropped.save('cropped.png')
    
    from dataloader.ylb_combined_dataset import ExternalTextureFolder as StageIIDataset
    from torch.utils.data import DataLoader
    import argparser
    args = argparser.parse_arguments()
    transforms_list = []
    if args.use_flip:
        transforms_list += [custom_transforms.RandomHorizontalFlip(),]
        
    transforms_list += [
        custom_transforms.toLAB(),
        custom_transforms.toTensor()
    ]
    transforms = custom_transforms.Compose(transforms_list)
    
    data_path = '/backup2/Datasets/Partial_textures'
    texture_path = '/backup2/yanglingbo/data/Describable_Texture_Dataset/dtd/images'
    dataset = StageIIDataset(args.phase, data_path, texture_path, transforms)
    data_loader = DataLoader(num_workers=0, dataset=dataset, batch_size=1, shuffle=True)
    
    sampler = RandomPatchSampler()
    for data in data_loader:
        out_dict = sampler.gen_input_and_labels(data)
        sampler.visualize_texture('debug_folder')  # all in the right place
        break
    
    

'''
def rand_between(a, b):
    return a + torch.round(torch.rand(1) * (b - a))[0]

def gen_input(img, skg, ini_texture, ini_mask, xcenter=64, ycenter=64, size=40):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh

    w, h = img.size()[1:3]
    # print w,h
    xstart = max(int(xcenter - size / 2), 0)
    ystart = max(int(ycenter - size / 2), 0)
    xend = min(int(xcenter + size / 2), w)
    yend = min(int(ycenter + size / 2), h)

    input_texture = ini_texture  # torch.ones(img.size())*(1)
    input_sketch = skg[0:1, :, :]  # L channel from skg
    input_mask = ini_mask  # torch.ones(input_sketch.size())*(-1)

    input_mask[:, xstart:xend, ystart:yend] = 1

    input_texture[:, xstart:xend, ystart:yend] = img[:, xstart:xend, ystart:yend].clone()

    return torch.cat((input_sketch.cpu().float(), input_texture.float(), input_mask), 0)

def get_coor(index, size):
    index = int(index)
    #get original coordinate from flatten index for 3 dim size
    w,h = size
    
    return ((index%(w*h))/h, ((index%(w*h))%h))

def gen_input_rand(img, skg, seg, size_min=40, size_max=60, num_patch=1):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh
    
    bs, c, w, h = img.size()
    results = torch.Tensor(bs, 5, w, h)
    texture_info = []

    # text_info.append([xcenter,ycenter,crop_size])
    seg = seg / torch.max(seg) #make sure it's 0/1
    
    seg[:,0:int(math.ceil(size_min/2)),:] = 0
    seg[:,:,0:int(math.ceil(size_min/2))] = 0
    seg[:,:,int(math.floor(h-size_min/2)):h] = 0
    seg[:,int(math.floor(w-size_min/2)):w,:] = 0
    
    counter = 0
    for i in range(bs):
        counter = 0
        ini_texture = torch.ones(img[0].size()) * (1)
        ini_mask = torch.ones((1, w, h)) * (-1)
        temp_info = []
        
        for j in range(num_patch):
            crop_size = int(rand_between(size_min, size_max))
            
            seg_index_size = seg[i,:,:].view(-1).size()[0]
            seg_index = torch.arange(0,seg_index_size)
            seg_one = seg_index[seg[i,:,:].view(-1)==1]
            if len(seg_one) != 0:
                seg_select_index = int(rand_between(0,seg_one.view(-1).size()[0]-1))
                x,y = get_coor(seg_one[seg_select_index],seg[i,:,:].size())
            else:
                x,y = (w/2, h/2)
            
            temp_info.append([x, y, crop_size])
            res = gen_input(img[i], skg[i], ini_texture, ini_mask, x, y, crop_size)

            ini_texture = res[1:4, :, :]

        texture_info.append(temp_info)
        results[i, :, :, :] = res
    return results, texture_info

def gen_local_patch(patch_size, batch_size, eroded_seg, seg, img):
    # generate local loss patch from eroded segmentation
    
    bs, c, w, h = img.size()
    texture_patch = img[:, :, 0:patch_size, 0:patch_size].clone()

    if patch_size != -1:
        eroded_seg[:,0,0:int(math.ceil(patch_size/2)),:] = 0
        eroded_seg[:,0,:,0:int(math.ceil(patch_size/2))] = 0
        eroded_seg[:,0,:,int(math.floor(h-patch_size/2)):h] = 0
        eroded_seg[:,0,int(math.floor(w-patch_size/2)):w,:] = 0

    for i_bs in range(bs):
                
        i_bs = int(i_bs)
        seg_index_size = eroded_seg[i_bs,0,:,:].view(-1).size()[0]
        seg_index = torch.arange(0,seg_index_size).cuda()
        #import pdb; pdb.set_trace()
        #print bs, batch_size
        seg_one = seg_index[eroded_seg[i_bs,0,:,:].view(-1)==1]
        if len(seg_one) != 0:
            random_select = int(rand_between(0, len(seg_one)-1))
            #import pdb; pdb.set_trace()
            
            x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())
            #print x,y,i_bs
        else:
            x,y = (w/2, h/2)

        if patch_size == -1:
            xstart = 0
            ystart = 0
            xend = -1
            yend = -1

        else:
            xstart = int(x-patch_size/2)
            ystart = int(y-patch_size/2)
            xend = int(x+patch_size/2)
            yend = int(y+patch_size/2)

        k = 1
        while torch.sum(seg[i_bs,0,xstart:xend,ystart:yend]) < k*patch_size*patch_size:
                
            try:
                k = k*0.9
                if len(seg_one) != 0:
                    random_select = int(rand_between(0, len(seg_one)-1))
            
                    x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())
            
                else:
                    x,y = (w/2, h/2)
                xstart = (int)(x-patch_size/2)
                ystart = (int)(y-patch_size/2)
                xend = (int)(x+patch_size/2)
                yend = (int)(y+patch_size/2)
            except:
                break
                
            
        texture_patch[i_bs,:,:,:] = img[i_bs, :, xstart:xend, ystart:yend]
        
    return texture_patch
'''