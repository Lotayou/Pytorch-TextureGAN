from dataloader.ylb_combined_dataset import GroundTruthImageFolder as StageIDataset
from dataloader.ylb_combined_dataset import ExternalTextureFolder as StageIIDataset
import utils.transforms as custom_transforms
import argparser
import numpy as np

from skimage.io import imsave
from skimage import color

if __name__ == '__main__':
    # 20191021: Remove RandomSizedCrop
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
    dataset = StageIDataset(args.phase, data_path, transforms)
    print(len(dataset))
    item = dataset[2333]
    
    # mask visualization passed.
    mask = item['mask']
    mask = mask.numpy()[0]
    mask /= mask.max()
    mask = np.stack((mask,) * 3, axis=2)
    imsave('mask.png', mask)
    
    # texture passed.
    texture = item['gt_lab']
    texture = texture.permute(1,2,0).numpy()
    texture = color.lab2rgb(texture)
    
    imsave('texture.png' , texture)
    
    # partial texture
    texture = item['partial_lab']
    texture = texture.permute(1,2,0).numpy()
    texture = color.lab2rgb(texture)
    
    imsave('partial_texture.png' , texture)
    print('finished')