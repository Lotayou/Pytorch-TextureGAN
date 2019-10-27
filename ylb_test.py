import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

import time
import os
import math
import random
import torchvision.models as models
from skimage.io import imsave
from skimage import color

from utils import transforms as custom_transforms
from models import scribbler, discriminator, texturegan, define_G, \
    scribbler_dilate_128, FeatureExtractor, GANLoss, GramMatrix
from dataloader.ylb_combined_dataset import GroundTruthImageFolder, ExternalTextureFolder

from ylb_trainer_utils import *
from ylb_sampling_utils import *

from skimage.io import imsave

if __name__ == '__main__':
    from argparser import parse_arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.test()
