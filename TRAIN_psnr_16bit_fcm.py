import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

from torchinfo import summary
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)
patch_num=8

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


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',pretrained_path=opt['path']['pretrained_netG'])
    
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['pretrained_netG'], net_type='G', pretrained_path=opt['path']['pretrained_netG'])
    # init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

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
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

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
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

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
    print('Number of pamrams: ',count_parameters(model.netG))
    logger.info(count_parameters(model.netG))
    logger.info(summary(model.netG))
    model.init_train()
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
        # logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    best_psnr=0.0
    psnr_history = []
    current_step_history =[]
    log_dir = opt['path']['log']
    
    for epoch in range(100000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            # if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            #     logger.info('Saving the model.')
            #     model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0

                for test_data in tqdm(test_loader):
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    # img_dir = os.path.join(opt['path']['images'], img_name)
                    img_dir = os.path.join(opt['path']['images'], str(current_step))
                    img_dir_best = os.path.join(opt['path']['images'], 'best_psnr')
                    
                    util.mkdir(img_dir)
                    util.mkdir(img_dir_best)

                    if opt['netG']['net_type']=='cat' or opt['netG']['net_type']=='QformerID':
                        model.feed_data_pad_patch(test_data,patch_num=patch_num)
                        model.test_jpeg_patch(patch_num=patch_num)
                        if model.ph > 0:
                            model.L = model.L[:, :, :-model.ph, :]
                            model.E = model.E[:, :, :-model.ph, :]
                        if model.pw > 0:
                            model.L = model.L[:, :, :, :-model.pw]
                            model.E = model.E[:, :, :, :-model.pw]
                    
                    
                    else:
                        model.feed_data_patch(test_data,patch_num=patch_num)
                        model.test_patch(patch_num=patch_num)

                    
                    if opt['netG']['net_type']=='QformerID':
                        visuals = model.current_visuals()
                        E_img = util.quatertensor2uint16_1ch(visuals['E'])
                        H_img = util.quatertensor2uint16_1ch(visuals['H'])
                    
                    else:
                        visuals = model.current_visuals()
                        E_img = util.tensor2uint16(visuals['E'])
                        # print(np.unique(E_img))
                        H_img = util.tensor2uint16(visuals['H'])
                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                    util.imsave(E_img, save_img_path)
                    # print(f'train:{np.unique(E_img)}')
                    # print(np.unique(H_img))
                    
                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr16(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim16(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB {:<4.4f}dB'.format(idx, image_name_ext, current_psnr, current_ssim))

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                
                # Append the average PSNR to history
                psnr_history.append(avg_psnr)

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}dB\n'.format(epoch, current_step, avg_psnr,avg_ssim))
                
                # 최고 psnr만 따로 저장
                if avg_psnr>best_psnr:
                        logger.info('Best score is updated, Saving the best model.\n')
                        model.save_best(current_step)
                        best_psnr=avg_psnr


                # -------------------------------
                # Save PSNR plot in log directory
                # -------------------------------
                # plt.figure()
                # plt.figure()
                # current_step_history.append(current_step)
                # plt.plot(current_step_history, psnr_history, label='Average PSNR')
                # plt.xlabel('iter')
                # plt.ylabel('PSNR (dB)')
                # plt.title('Flow of PSNR')
                # plt.legend()
                # plt.grid()
                
                # psnr_plot_path = os.path.join(log_dir, "PSNR_plot.png")
                # plt.savefig(psnr_plot_path)
                # plt.close()

if __name__ == '__main__':
    main()
