import torch
from torch.autograd import Variable
import numpy as np
from utils import transforms as custom_transforms
from models import GramMatrix
from utils.visualize import vis_image, vis_patch
import time
#import cv2
import math
import random


from torch.nn import init
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print(('initialization method [%s]' % init_type))
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


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

def renormalize(img):
    """
    Renormalizes the input image to meet requirements for VGG-19 pretrained network
    """

    forward_norm = torch.ones(img.data.size()) * 0.5
    forward_norm = Variable(forward_norm.cuda())
    img = (img * forward_norm) + forward_norm  # add previous norm
    # return img
    mean = img.data.new(img.data.size())
    std = img.data.new(img.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    img -= Variable(mean)
    img = img / Variable(std)

    return img