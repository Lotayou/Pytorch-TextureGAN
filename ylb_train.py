import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import os
import math
import random
import torchvision.models as models
from skimage.io import imsave
from skimage import color

import warnings

from utils import transforms as custom_transforms
from models import scribbler, discriminator, texturegan, define_G, \
    scribbler_dilate_128, FeatureExtractor, GANLoss, GramMatrix
from dataloader.ylb_combined_dataset import GroundTruthImageFolder, ExternalTextureFolder

from ylb_trainer_utils import *
from ylb_sampling_utils import *


################################################
#  Prepare dataset, models and loss functions  #
################################################

def get_transforms(args):
    # 20191021: Remove RandomSizedCrop
    transforms_list = []
    if args.use_flip:
        transforms_list.append(custom_transforms.RandomHorizontalFlip())
    if args.color_space == 'lab':
        transforms_list.append(custom_transforms.toLAB())
    elif args.color_space == 'rgb':
        transforms_list.append(custom_transforms.toRGB('RGB'))
    
    transforms_list.append(custom_transforms.toTensor())
    transforms = custom_transforms.Compose(transforms_list)
    return transforms
    
def get_loader(args):
    trans = get_transforms(args)
    if args.training_stage == 'I':
        dataset = GroundTruthImageFolder(args.phase, args.data_path, trans, 
            max_dataset_size=args.max_dataset_size)
    elif args.training_stage == 'II':
        print('Prepare stage II dataset')
        dataset = ExternalTextureFolder(args.phase, args.data_path, args.texture_path, trans,
            max_dataset_size=args.max_dataset_size)
    else:
        raise(NotImplementedError('Unrecognized stage [I/II]'))
    
    data_loader = DataLoader(num_workers=args.num_workers, dataset=dataset, 
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    return data_loader


def get_models(args):
    inc = args.input_nc
    ngf = args.ngf

    if args.model == 'scribbler':
        netG = scribbler.Scribbler(inc, 3, ngf)
    elif args.model == 'texturegan':
        netG = texturegan.TextureGAN(inc, 3, ngf)
    elif args.model == 'pix2pix':
        netG = define_G(inc, 3, ngf)
    elif args.model == 'scribbler_dilate_128':
        netG = scribbler_dilate_128.ScribblerDilate128(inc, 3, ngf)
    else:
        print((args.model + ' not support. Using Scribbler model'))
        netG = scribbler.Scribbler(inc, 3, ngf)

    if args.phase == 'train':
        sigmoid_flag = args.gan == 'dcgan'
        netD = discriminator.Discriminator(1, ngf, sigmoid_flag)
        netD_local = discriminator.LocalDiscriminator(2, ngf, sigmoid_flag)
    else:
        netD, netD_local = None, None

    if args.phase == 'test' or args.continue_train:
        load_network(netG, 'G', args.load_epoch, args)
        if args.phase == 'train':
            load_network(netD, 'D', args.load_epoch, args)
            load_network(netD_local, 'D_local', args.load_epoch, args)
    else:
        init_weights(netG, args.init_type)
        if args.phase == 'train':
            init_weights(netD, args.init_type)
            init_weights(netD_local, args.init_type)
            
    return netG, netD, netD_local

def load_network(model, network_label, epoch, args):
    ckpt_dir = './checkpoints/%s' % args.name
    filename = "{0}_net_{1}_{2}.pth".format(network_label, args.model, epoch)
    load_path = os.path.join(ckpt_dir, filename)
    model.load_state_dict(torch.load(load_path))
    
'''
20191021: Rewrite the original training procedure as a Trainer class
    No offense, but the original training code is so messy that I don't wanna change upon it!
'''
class Trainer():
    def name(self):
        return 'TextureGAN Trainer'
    
    def set_criterions(self, args):
        self.criterion_gan = GANLoss(
            use_lsgan=(args.gan == 'lsgan')
        )
        
        self.criterion_texturegan = GANLoss(use_lsgan=True) # just like mse

        # criterion_l1 = nn.L1Loss()
        self.criterion_pixel_l = nn.MSELoss()
        self.criterion_pixel_ab = nn.MSELoss()
        self.criterion_style = nn.MSELoss()
        self.criterion_feat = nn.MSELoss()
        
    def __init__(self, args):
        self.args = args
        self.stage_II_flag = args.training_stage == 'II'
        
        self.ckpt_dir = './checkpoints/%s' % args.name \
            if args.phase == 'train' \
            else './results/%s' % args.name
            
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.image_save_dir = self.ckpt_dir + '/images'
        if not os.path.isdir(self.image_save_dir):
            os.makedirs(self.image_save_dir)
                
        self.netG, self.netD, self.netD_local = get_models(args)
        self.data_loader = get_loader(args)
        cudaify_list = ['netG']   # modules to be loaded in GPU
        
        if self.stage_II_flag:
            self.patch_sampler = RandomPatchSampler()  # defined in ylb_sampling_utils.py
        
        if args.phase == 'train':
            self.set_criterions(args)  # dunt wanna write 6 assigns
            
            # VGG-19 perceptual and style feature extractor (move from main to here)
            feat_model = models.vgg19(pretrained=True).cuda()
            layers_map = {'relu4_2': '22', 'relu2_2': '8', 'relu3_2': '13','relu1_2': '4'}
            self.content_extractor = FeatureExtractor(feat_model.features, [layers_map[args.content_layers]])
            self.style_extractor = FeatureExtractor(feat_model.features,
                                             [layers_map[x.strip()] for x in args.style_layers.split(',')])
            self.gram = GramMatrix()
            
            self.opt_D = Adam(self.netD.parameters(), lr=args.learning_rate_D, betas=(0.5, 0.999))
            self.opt_G = Adam(self.netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            self.opt_D_local = Adam(self.netD_local.parameters(), lr=args.learning_rate_D_local, betas=(0.5, 0.999))
            
            self.opt_D.zero_grad()
            self.opt_G.zero_grad()
            self.opt_D_local.zero_grad()
            
            cudaify_list += [ 'netD', 'netD_local', 
                'criterion_feat', 'criterion_gan', 'criterion_pixel_ab',
                'criterion_pixel_l', 'criterion_style', 'criterion_texturegan'
            ]
            
        self.move_to_cuda(cudaify_list)
        
        arg_dict = vars(args)
        print('------------ Options -------------')
        for k, v in sorted(arg_dict.items()):
            print(('%s: %s' % (str(k), str(v))))
        print('-------------- End ----------------')
        with open(os.path.join(self.ckpt_dir, 'opt.txt'), 'w') as f:
            for k, v in sorted(arg_dict.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
        print('%s is set and ready to go' % self.name())
        if self.stage_II_flag:
            print('Begin stage II: External texture fine-tuning...')
        
    def move_to_cuda(self, names):
        # load all models and losses to cuda, set all networks to training mode
        for name in names:
            tmp = getattr(self, name)
            if name.startswith('net'):
                setattr(self, name, tmp.cuda().train())
            else:
                setattr(self, name, tmp.cuda())
        
    def set_input(self, data):
        '''
            Input: the full UV texture and a partial visibility matrix
            Output: something like TextureGAN input (except the binary sketch)
        '''
        if self.stage_II_flag:
            input_dict = self.patch_sampler.gen_input_and_labels(data)
            # additional
            self.input = input_dict['input'].float().cuda()
            self.gt = input_dict['gt'].float().cuda()
            self.texture = input_dict['texture'].float().cuda()
            self.swapped_flag = input_dict['flag']
        else:        
            # stage 1, sample random patches from the gt image
            partial = custom_transforms.normalize_lab(data['partial_lab'])
            gt = custom_transforms.normalize_lab(data['gt_lab'])
            mask = custom_transforms.normalize_seg(data['mask'])
            # mask out: n h w, unsqueeze in dim1
            input = torch.cat((mask.unsqueeze_(1), partial), dim=1)  # 4-channel
            # print(input.shape, gt.shape)
            # explicit type cast to float, matching module.weights
            self.input = input.float().cuda()
            self.gt = gt.float().cuda()
        
    def update_G(self):
        self.opt_G.zero_grad()
        self.outputG = self.netG(self.input)
        
        if self.args.loss_texture == 'original_image':
            self.target = self.gt
        else:
            self.target = self.texture  # still 0.5 change to be gt
        
        #### global color loss
        outputab = self.outputG[:,1:3]
        targetab = self.target[:, 1:3]
        err_pixel_ab = self.args.pixel_weight_ab * self.criterion_pixel_ab(outputab, targetab)
        
        #### global VGG feature loss
        # The original project only use the feature response of the first layer.
        self.outputl = self.outputG[:, 0:1]# to be used for discriminator update
        self.gtl = self.target[:, 0:1]   
        outputlll = torch.cat((self.outputl,) * 3, dim=1)
        gtlll = torch.cat((self.gtl,) * 3, dim=1)
        out_feat = self.content_extractor(renormalize(outputlll))[0]
        gt_feat = self.content_extractor(renormalize(gtlll))[0]
        
        err_feat = self.args.feature_weight * self.criterion_feat(out_feat, gt_feat.detach())
        
        #### global GAN loss, clear dis gradients.
        dis_feat = self.netD(self.outputl)
        err_gan = self.args.discriminator_weight * self.criterion_gan(dis_feat, True)  # fool the discriminator
        
        #### global Pixel L Loss
        err_pixel_l = self.args.global_pixel_weight_l * self.criterion_pixel_l(self.outputl, self.gtl.detach())
        
        #### style loss 
        err_style_total = self.get_style_loss(outputlll, gtlll)
        
        err_total = err_pixel_l + err_pixel_ab + err_gan + err_feat + err_style_total
        err_total.backward()
        
        self.opt_G.step()
        # return loss
        loss_dict = {
            'l': err_pixel_l.item(),
            'ab': err_pixel_ab.item(),
            'gan': err_gan.item(),
            'feat': err_feat.item(),
            'style': err_style_total.item(),
            'total': err_total.item(),
        }
        return loss_dict
        
    def get_style_loss(self, outputlll, gtlll):
        if self.stage_II_flag:
            err_style_total = 0
            for _ in range(self.args.num_local_texture_patch):
                gt_bbox_type = 'torso' if self.swapped_flag else 'all'
                gen_patch_lll = self.patch_sampler.get_loss_patch(outputlll, bbox_type='torso')
                gt_patch_lll = self.patch_sampler.get_loss_patch(gtlll, bbox_type=gt_bbox_type)
                
                gen_patch_l = self.patch_sampler.get_loss_patch(self.outputl, bbox_type='torso')
                gt_patch_l = self.patch_sampler.get_loss_patch(self.gtl, bbox_type=gt_bbox_type)
                
                #### Local Style Loss 
                output_style_feat = self.style_extractor(gen_patch_lll)
                target_style_feat = self.style_extractor(gt_patch_lll)
                err_style = 0
                for i in range(len(output_style_feat)):
                    gram_y = self.gram(output_style_feat[i])
                    gram_s = self.gram(target_style_feat[i])

                    err_style += self.criterion_style(gram_y, gram_s.detach())
                
                
                #### Local pixel L Loss (default weight == 0 delete)
                
                # err_pixel_l = self.args.local_pixel_weight_l \
                #    * self.criterion_pixel_l(gen_patch_l, gt_patch_l.detach())
                
                #### Local D Loss
                fake_real_pack = torch.cat((gen_patch_l, gt_patch_l), dim=1)
                out_dis_feat = self.netD_local(fake_real_pack)
                err_d_local = self.args.discriminator_local_weight * self.criterion_texturegan(out_dis_feat, True)
                                   
                err_style_total += err_style * self.args.style_weight + err_d_local
            
            return err_style_total
            
        else:
            # Stage I, no texture gan loss, use global gram-based style loss.
            err_style = 0
            output_style_feat = self.style_extractor(outputlll)
            gt_style_feat = self.style_extractor(gtlll)
            for i in range(len(output_style_feat)):
                out_gram = self.gram(output_style_feat[i])
                gt_gram = self.gram(gt_style_feat[i])
                err_style += self.criterion_style(out_gram, gt_gram.detach())
            
            return err_style * self.args.style_weight
        
        
    def update_D(self):
        self.opt_D.zero_grad()
        # calc loss and backward
        real_D = self.netD(self.gtl)
        fake_D = self.netD(self.outputl.detach())
        err_D_fake = self.criterion_gan(fake_D, False)
        
        # calculate accuracy, only update D if acc score is below threshold
        
        real_acc = torch.mean(real_D.clamp(0,1).round())
        fake_acc = torch.mean(1 - fake_D.clamp(0,1).round())
        acc = (real_acc + fake_acc) / 2
        if acc.item() < self.args.threshold_D_max:
            err_D_real = self.criterion_gan(real_D, True)
            err_D_fake = self.criterion_gan(fake_D, False)
            err_D_total = (err_D_real + err_D_fake) / 2
            err_D_total.backward()
            self.opt_D.step()
            err_dict = {
                'real': err_D_real.item(),
                'fake': err_D_fake.item(),
                'total': err_D_total.item(),
            }
        else:
            print('Discriminator too strong [%.6f > %.6f], stop updating' \
                % (acc, self.args.threshold_D_max))
            err_dict = None
        
        # return loss
        return err_dict
        
    def update_D_local(self):
        self.opt_D_local.zero_grad()
        # calc loss and backward
        real_patch_l1 = self.patch_sampler.get_loss_patch(self.gtl, bbox_type='all')
        real_patch_l2 = self.patch_sampler.get_loss_patch(self.gtl, bbox_type='all')
        real_pack = torch.cat((real_patch_l1, real_patch_l2), dim=1)
        real_feat_D = self.netD_local(real_pack)
        realreal_acc = torch.mean(real_feat_D.clamp(0,1).round())
        
        fake_patch_l = self.patch_sampler.get_loss_patch(self.outputl, bbox_type='torso')
        fake_pack = torch.cat((fake_patch_l, real_patch_l1), dim=1)
        fake_feat_D = self.netD_local(fake_pack.detach())
        fakefake_acc = torch.mean(1 - fake_feat_D.clamp(0,1).round())
        
        acc = (realreal_acc + fakefake_acc) / 2
        if acc.item() < self.args.threshold_D_max:
            err_D_local_real = self.criterion_texturegan(real_feat_D, True)
            err_D_local_fake = self.criterion_texturegan(fake_feat_D, False)
            err_D_local_total = (err_D_local_real + err_D_local_fake) / 2
            err_D_local_total.backward()
            self.opt_D_local.step()
            err_dict = {
                'real': err_D_local_real.item(),
                'fake': err_D_local_fake.item(),
                'total': err_D_local_total.item(),
            }
        else:
            print('Local Discriminator too strong [%.6f > %.6f], stop updating' \
                % (acc, self.args.threshold_D_max))
            err_dict = None
            
        return err_dict
        
    def train_one_epoch(self, epoch, is_stage_2):
        for i, data in enumerate(self.data_loader):
            self.set_input(data)
            self.G_loss_dict = self.update_G()
            self.D_loss_global_dict = self.update_D()
            if is_stage_2:
                self.D_loss_local_dict = self.update_D_local()
            
            if i % self.args.visualize_every == 0:
                self.logging(epoch, i)
                self.visualize(epoch, i)
         
    def train(self):
        cur_epoch = 0
        if self.stage_II_flag or self.args.continue_train:
            cur_epoch = self.args.load_epoch
            
        for i in range(cur_epoch + 1, args.num_epoch + 1):
            tic = time.time()
            self.train_one_epoch(i, self.stage_II_flag)
            print('Epoch %d finished within %f s' % (i, time.time() - tic))
            
            if i % self.args.save_every == 0:
                self.save_network(self.netG, 'G', i, self.args)
                self.save_network(self.netD, 'D', i, self.args)
                if self.stage_II_flag:
                    self.save_network(self.netD_local, 'D_local', i, self.args)
            
    def test_one_image(self):
        with torch.no_grad():
            self.outputG = self.netG(self.input)
            
    def test(self):
        for (i, data) in tqdm(enumerate(self.data_loader)):
            self.set_input(data)
            self.test_one_image()
            self.visualize(self.args.load_epoch, i)
        
    def logging(self, epoch, i):
        log = "Epoch: {0}  Iteration: {1}:".format(epoch, i)
        if self.D_loss_global_dict is not None:
            log += '\n\tD_global: '
            for k, v in self.D_loss_global_dict.items():
                log += '%s: %.6f  ' % (k, v)
        
        # TODO: stage II local loss dict
        if self.D_loss_local_dict is not None:
            log += '\n\tD_local: '
            for k, v in self.D_loss_local_dict.items():
                log += '%s: %.6f  ' % (k, v)
        log += '\n\tG: '
        for k, v in self.G_loss_dict.items():
            log += '%s: %.6f  ' % (k, v)
        print(log)
        
    def visualize(self, epoch, i):
        # For training i stands for iteration 
        # For testing i stands for case no.
        identifier = 'iter' if self.args.phase == 'train' else 'case'
        name = 'result_epoch_%03d_%s_%05d.png' % (epoch, identifier, i)
        
        # from left to right: gt, input, generated        
        bundle = torch.cat((self.target, self.input[:,1:4], self.outputG), dim=3)
        bundle = custom_transforms.denormalize_lab(bundle[0:1])
        bundle = bundle[0].detach().cpu().numpy().transpose(1,2,0)
        bundle = (255. *color.lab2rgb(bundle)).astype(np.uint8)
        imsave(os.path.join(self.image_save_dir, name), bundle)  # float to uint8 is taken care of
    
    def save_network(self, model, network_label, epoch, args):
        save_filename = "{0}_net_{1}_{2}.pth".format(network_label, args.model, epoch)
        # ckpt_dir will be at './result/$NAME' when phase == 'test'
        # but in testing mode no new models will be saved, so we are cool.
        save_path = os.path.join(self.ckpt_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)
        model.cuda()


###################
# Training Script #
###################

if __name__ == '__main__':
    from argparser import parse_arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    trainer = Trainer(args)
    if args.phase == 'train':
        trainer.train()
    else:
        trainer.test()

# obsolete code
#
# val_dataset = ImageFolder('val', args.data_path, transforms)
# indices = torch.randperm(len(val_dataset))
# val_display_size = args.batch_size
# val_display_sampler = SequentialSampler(indices[:val_display_size])
# val_loader = DataLoader(dataset=val_dataset, batch_size=val_display_size, sampler=val_display_sampler)
    