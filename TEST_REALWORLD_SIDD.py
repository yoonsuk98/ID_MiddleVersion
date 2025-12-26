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
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

from tqdm import tqdm

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


# python TEST_REALWORLD_SIDD.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_abl4_remove_mfb_reverse_realworld.json --model_name PROPOSED_MIDDLE

def main(json_path='weight/SIDD/SRnew/srnew.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--dist', default=False)
    parser.add_argument('--model_name', default='swinir')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    opt['path']['pretrained_netG'] = f'model_zoo/denoise_realworld/{parser.parse_args().model_name}.pth'

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
    # configure logger
    # ----------------------------------------

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)

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

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

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

    netG = model.netG
    netG.cuda()
    netG.eval()

    result_dir_mat = os.path.join('testsets/SIDD', parser.parse_args().model_name)
    os.makedirs(result_dir_mat, exist_ok=True)

    filepath = os.path.join('testsets/SIDD/ValidationNoisyBlocksSrgb.mat')
    filepath_gt = os.path.join('testsets/SIDD/ValidationGtBlocksSrgb.mat')

    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
    Inoisy /= 255.

    img_gt = sio.loadmat(filepath_gt)
    Igt = np.float32(np.array(img_gt['ValidationGtBlocksSrgb']))
    Igt /= 255.

    restored = np.zeros_like(Inoisy)
    with torch.no_grad():
        for i in tqdm(range(40)):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                restored_patch = netG(noisy_patch)
                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
                restored[i, k, :, :, :] = restored_patch

                org_noise_patch = Inoisy[i, k, :, :, :]
                gt_patch = Igt[i, k, :, :, :]

                save_file = os.path.join('Output_REALWORLD_DENOISE', opt['netG']['net_type'], 'SIDD',
                                         '%04d_%02d.png' % (i + 1, k + 1))

                save_org_noise = os.path.join('Output_REALWORLD_DENOISE', 'Noise', 'SIDD',
                                         '%04d_%02d.png' % (i + 1, k + 1))

                save_gt = os.path.join('Output_REALWORLD_DENOISE', 'GT', 'SIDD',
                                          '%04d_%02d.png' % (i + 1, k + 1))

                directory = os.path.dirname(save_file)
                if directory != "" and not os.path.exists(directory):
                    os.makedirs(directory)
                util.imsave(img_as_ubyte(restored_patch), save_file)

                directory_org_noise = os.path.dirname(save_org_noise)
                if directory_org_noise != "" and not os.path.exists(directory_org_noise):
                    os.makedirs(directory_org_noise)
                util.imsave(img_as_ubyte(org_noise_patch), save_org_noise)

                directory_gt = os.path.dirname(save_gt)
                if directory_gt != "" and not os.path.exists(directory_gt):
                    os.makedirs(directory_gt)
                util.imsave(img_as_ubyte(gt_patch), save_gt)

    sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored, })


if __name__ == '__main__':
    main()