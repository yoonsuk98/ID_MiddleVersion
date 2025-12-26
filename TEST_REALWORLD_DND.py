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


# python TEST_REALWORLD_DND.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_abl4_remove_mfb_reverse_realworld.json --model_name PROPOSED_MIDDLE

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
    # testset = parser.parse_args().test_set

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

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
        # logger.info(option.dict2str(opt))

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

    result_dir_mat = os.path.join('testsets/DND', parser.parse_args().model_name)
    os.makedirs(result_dir_mat, exist_ok=True)

    israw = False
    eval_version = "1.0"
    input_dir = "testsets/DND"
    # Load info
    infos = h5py.File(os.path.join(input_dir, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']

    # Process data
    with torch.no_grad():
        for i in tqdm(range(50)):
            Idenoised = np.zeros((20,), dtype=object)
            filename = '%04d.mat' % (i + 1)
            filepath = os.path.join(input_dir, 'ValidationNoisyBlocksSrgb', filename)
            img = h5py.File(filepath, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)

            # bounding box
            ref = bb[0][i]
            boxes = np.array(info[ref]).T

            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]

                org_noise_patch = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :]

                noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1], idx[2]:idx[3], :]).unsqueeze(0).permute(0, 3, 1,2).cuda()
                restored_patch = netG(noisy_patch)
                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                Idenoised[k] = restored_patch
                save_file = os.path.join('Output_REALWORLD_DENOISE', opt['netG']['net_type'], 'DND',
                                         '%04d_%02d.png' % (i + 1, k + 1))
                directory = os.path.dirname(save_file)
                if directory != "" and not os.path.exists(directory):
                    os.makedirs(directory)
                util.imsave(img_as_ubyte(restored_patch), save_file)

                save_org_noise = os.path.join('Output_REALWORLD_DENOISE', 'Noise', 'DND',
                                         '%04d_%02d.png' % (i + 1, k + 1))
                directory_org_noise = os.path.dirname(save_org_noise)
                if directory_org_noise != "" and not os.path.exists(directory_org_noise):
                    os.makedirs(directory_org_noise)
                util.imsave(img_as_ubyte(org_noise_patch), save_org_noise)

                torch.cuda.empty_cache()

            # save denoised data
            sio.savemat(os.path.join(result_dir_mat, filename),
                        {"Idenoised": Idenoised, "israw": israw, "eval_version": eval_version}, )
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()