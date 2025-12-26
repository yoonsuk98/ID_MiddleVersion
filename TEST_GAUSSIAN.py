import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ir_task = 'denoise' 
net_name = 'proposed_middleversion' # same as model_zoo final dir name

# python TEST_GAUSSIAN.py --opt ./options/denoise/train_abl4_remove_mfb_reverse_9x9_denoise_15.json --test_set CBSD68 --noise 15
# python TEST_GAUSSIAN.py --opt ./options/denoise/train_abl4_remove_mfb_reverse_9x9_denoise_15.json --test_set Kodak24 --noise 15
# python TEST_GAUSSIAN.py --opt ./options/denoise/train_abl4_remove_mfb_reverse_9x9_denoise_15.json --test_set McMaster --noise 15
# python TEST_GAUSSIAN.py --opt ./options/denoise/train_abl4_remove_mfb_reverse_9x9_denoise_15.json --test_set urban100 --noise 15

def main():

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/QformerID_Gaussian.json', help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--test_set', default='Set5')
    parser.add_argument('--noise', default=75)
    noise = int(parser.parse_args().noise)
    

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    
    if ir_task == 'denoise':
        opt['path']['pretrained_netG'] = f'model_zoo/{ir_task}/{net_name}/noise_{parser.parse_args().noise}.pth'
        

    opt['datasets']['test']['dataroot_H']=f'testsets/{parser.parse_args().test_set}/'
    opt['datasets']['test']['sigma']=int(parser.parse_args().noise)
    opt['datasets']['test']['sigma_test']=int(parser.parse_args().noise)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))


    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)


    # ----------------------------------------
    # seed
    # ----------------------------------------

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'train':
            continue

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    #print('Number of parameters: {}'.format(model.info_params()))
    #if opt['rank'] == 0:
    #    logger.info(model.info_network())
    #   logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''


    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_org_noise_psnr = 0.0
    avg_org_noise_ssim = 0.0
    
    idx = 0
    total_t = 0.0
    
    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        # img_dir = os.path.join(opt['path']['images'], img_name)
        # util.mkdir(img_dir)
        
        ###################################################################################
        #quaternion
        # _, _, h_old, w_old = test_data['L'].size()
        # window_size = 8
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [2])], 2)[:, :, :h_old + h_pad, :]
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [3])], 3)[:, :, :, :w_old + w_pad]
        ###################################################################################

        if opt['netG']['net_type'] == 'art' or opt['netG']['net_type'] == 'cat':
            model.feed_data_pad_patch(test_data, patch_num=4)
            start_t = time.time()
            model.test_jpeg_patch(patch_num=4)
            end_t = time.time() - start_t
            total_t += end_t

            if model.ph > 0:
                model.L = model.L[:, :, :-model.ph, :]
                model.E = model.E[:, :, :-model.ph, :]
            if model.pw > 0:
                model.L = model.L[:, :, :, :-model.pw]
                model.E = model.E[:, :, :, :-model.pw]

        else:
            model.feed_data(test_data)
            start_t = time.time()
            model.test()
            end_t = time.time() - start_t
            total_t += end_t
        
        visuals = model.current_visuals()
        
        N_img = util.tensor2uint(visuals['L']) # 노이즈 이미지
        
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        # save_img_path = os.path.join('Output',opt['netG']['net_type'], 'Gaussian',f'{noise}',opt['datasets']['test']['dataroot_H'].split('/')[1],
        #                               '{:s}.png'.format(img_name))
        save_img_path = os.path.join('Output', f'{net_name}', 'Gaussian', f'{noise}',opt['datasets']['test']['dataroot_H'].split('/')[1],
                                     '{:s}.png'.format(img_name))
        save_noise_img_path = os.path.join('Output','Gaussian',f'{noise}',opt['datasets']['test']['dataroot_H'].split('/')[1],
                                      '{:s}.png'.format(img_name))
        directory = os.path.dirname(save_img_path)
        noise_directory = os.path.dirname(save_noise_img_path)
        ###################################################################################
        #quaternion
        # E_img = util.quatertensor2uint(visuals['E'][..., :h_old, :w_old])
        # H_img = util.quatertensor2uint(visuals['H'])
        
           

        # save_img_path = os.path.join('Output',opt['netG']['net_type'], 'Gaussian',f'{noise}',opt['datasets']['test']['dataroot_H'].split('/')[1],
        #                               '{:s}_QTSR.png'.format(img_name))
        # directory = os.path.dirname(save_img_path)
        
        ###################################################################################
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        util.imsave(E_img, save_img_path)
        
        if noise_directory != "" and not os.path.exists(noise_directory):
            os.makedirs(noise_directory)
        util.imsave(N_img, save_noise_img_path)
        
        current_org_noise_psnr = util.calculate_psnr(N_img, H_img, border=border)
        current_org_noise_ssim = util.calculate_ssim(N_img, H_img, border=0)
        
        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
        current_ssim = util.calculate_ssim(E_img, H_img, border=0)
        
        print('{:->4d}--> {:>10s} | {:<4.2f}dB/{:<4.4f}dB'.format(idx, img_name, current_psnr,current_ssim))
        
        avg_psnr += current_psnr
        avg_ssim += current_ssim
        
        avg_org_noise_psnr += current_org_noise_psnr
        avg_org_noise_ssim += current_org_noise_ssim

        # -----------------------
        # save estimated image E
        # -----------------------

    avg_t = total_t / idx
    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    
    avg_org_noise_psnr = avg_org_noise_psnr / idx
    avg_org_noise_ssim = avg_org_noise_ssim / idx
    
    print('ORG : Average PSNR/SSIM/Elapsed_time in cutting way: {:<.2f}dB/{:<.4f}dB/{:<.4f}ms'.format(avg_org_noise_psnr, avg_org_noise_ssim, avg_t))
    print('EST : Average PSNR/SSIM/Elapsed_time in cutting way: {:<.2f}dB/{:<.4f}dB/{:<.4f}ms'.format(avg_psnr, avg_ssim, avg_t))



if __name__ == '__main__':
    main()