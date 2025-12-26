import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os

class DatasetPostFilter(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for Postfiltering.

    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPostFilter, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 1
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        # ------------------------------------
        # 동일한 파일명 없을 시 처리
        # ------------------------------------

        file_in_H = set()  # 빈 집합 생성
        for file in self.paths_H:
            basename = os.path.basename(file)  # 파일 경로에서 파일 이름만 추출
            basename = basename.split('_')
            basename = basename[0]+'_'+basename[1] + '_' + basename[3]

            file_in_H.add(basename)  # 집합에 추가

        file_in_L = set()  # 빈 집합 생성
        for file in self.paths_L:
            basename = os.path.basename(file)  # 파일 경로에서 파일 이름만 추출
            basename = basename.split('_')
            basename = basename[0] + '_' + basename[1] + '_' + basename[3]

            file_in_L.add(basename)  # 집합에 추가

        qp = opt['dataroot_L'][-2:]

        for file in file_in_H - file_in_L:
            file = file.split('_')
            filename = file[0] + '_' + file[1] + '_' + '00' +'_'+file[2]
            remove_filename = opt['dataroot_H'] + '/' + filename
            os.remove(remove_filename)
            print(remove_filename)

        for file in file_in_L - file_in_H:
            file = file.split('_')
            filename = file[0] + '_' + file[1] + '_' + qp + '_' + file[2]
            remove_filename = opt['dataroot_L'] + '/' + filename
            os.remove(remove_filename)
            print(remove_filename)
        #######################################



        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        # print(f'first_H : {np.unique(img_H)}')
        # img_H = util.uint2single(img_H)
        img_H = util.uint162single(img_H)  ##
        # print(f'second_H : {np.unique(img_H)}')

        # ------------------------------------
        # modcrop
        # ------------------------------------
        # img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)
        # img_L = util.uint2single(img_L)
        img_L = util.uint162single(img_L) ##

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------

            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)



        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
