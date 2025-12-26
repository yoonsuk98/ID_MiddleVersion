from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, PerceptualLoss, CompoundLoss,SobelEdgeLoss, HighFrequencyLoss,PSNRLoss, FFTLoss, L_color, L_spa, L_exp, L_TV
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip

import torch.nn.functional as F
import math

from models.lr_scheduler2 import CosineAnnealingRestartCyclicLR
# from warmup_scheduler import GradualWarmupScheduler
############################################################
def pad_to_mod32(x):
    """ x: (B, C, H, W) 형태 텐서.
        32 배수(H,W) 크기로 만들기 위해 필요하면 아래/오른쪽에 반사-padding.
        returns: (x_padded, H_pad, W_pad)
    """
    B, C, H, W = x.shape
    mod = 32

    # 각 차원이 32의 배수가 되도록 패딩해야 하는 크기 계산
    H_pad = math.ceil(H / mod) * mod - H  # 32 배수에서 모자라는 부분
    W_pad = math.ceil(W / mod) * mod - W

    if H_pad > 0 or W_pad > 0:
        x_padded = F.pad(x, (0, W_pad, 0, H_pad), mode='reflect', value=0)
    else:
        x_padded = x

    return x_padded, H_pad, W_pad


def unpad_from_mod32(x, H_pad, W_pad):
    """ pad_to_mod32로 패딩했던 텐서를 다시 원래 크기로 잘라내는 함수 """
    if H_pad > 0:
        x = x[:, :, :-H_pad, :]
    if W_pad > 0:
        x = x[:, :, :, :-W_pad]
    return x
############################################################
def pad_to_mod2(x):
    B, C, H, W = x.shape
    # H와 W 각각에 대해 다음 2의 거듭제곱을 구함.
    new_H = 2 ** math.ceil(math.log2(H))
    new_W = 2 ** math.ceil(math.log2(W))
    H_pad = new_H - H
    W_pad = new_W - W
    
    if H_pad > 0 or W_pad > 0:
        x_padded = F.pad(x, (0, W_pad, 0, H_pad), mode='reflect', value=0)
    else:
        x_padded = x

    return x_padded, H_pad, W_pad

def unpad_from_mod2(x, H_pad, W_pad):
    """ pad_to_mod2로 패딩했던 텐서를 다시 원래 크기로 잘라내는 함수 """
    if H_pad > 0:
        x = x[:, :, :-H_pad, :]
    if W_pad > 0:
        x = x[:, :, :, :-W_pad]
    return x
############################################################
#8의배수로 패딩 후 언패딩
def pad_to_mod8(x):
    """
    입력 x (B, C, H, W)를 H, W가 8의 배수가 되도록 반사 패딩.
    Returns:
        x_padded: padded tensor
        H_pad: 추가된 높이
        W_pad: 추가된 너비
    """
    B, C, H, W = x.shape
    new_H = math.ceil(H / 8) * 8
    new_W = math.ceil(W / 8) * 8

    H_pad = new_H - H
    W_pad = new_W - W

    if H_pad > 0 or W_pad > 0:
        x_padded = F.pad(x, (0, W_pad, 0, H_pad), mode='reflect')
    else:
        x_padded = x

    return x_padded, H_pad, W_pad

def unpad_from_mod8(x, H_pad, W_pad):
    """
    pad_to_multiple8로 패딩했던 텐서를 원래 크기로 잘라냄
    """
    if H_pad > 0:
        x = x[:, :, :-H_pad, :]
    if W_pad > 0:
        x = x[:, :, :, :-W_pad]
    return x
############################################################

def pad_to_square(x, mod=64):
    """
    입력 x:(B,C,H,W) → square + win_size(mod) 배수
    """
    B, C, H, W = x.shape
    size = max(H, W)
    # win_size 배수로 올림
    size = math.ceil(size / mod) * mod

    pad_h = size - H
    pad_w = size - W

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w

def unpad_square(x, pad_h, pad_w):
    """
    pad_to_square 로 늘린 부분 제거
    """
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

############################################################

def pad_to_mod256(x):
    """
    입력 x: (B, C, H, W)
    → H, W를 각각 64의 배수로 반사 패딩
    Returns:
        x_padded, pad_h, pad_w
    """
    B, C, H, W = x.shape
    pad_h =  math.ceil(H / 256) * 256 - H
    pad_w =  math.ceil(W / 256) * 256 - W
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return x, pad_h, pad_w

def unpad_mod256(x, pad_h, pad_w):
    """
    pad_to_mod64 로 늘린 부분만 제거
    """
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

############################################################
class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # save best model
    # ----------------------------------------
    def save_best(self,iter_label):
        self.save_best_network(self.save_best_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_best_network(self.save_best_dir, self.netE, 'E', 'best')
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_best_dir, self.G_optimizer, 'optimizerG', 'best')


    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'psnrloss':
            self.G_lossfn = PSNRLoss().to(self.device)    
        elif G_lossfn_type == 'huber':
            self.G_lossfn = nn.Huber()
        elif G_lossfn_type == 'freq_gen':
            self.G_lossfn = nn.L1Loss().to(self.device)
            self.F_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'perceptual':
            self.G_lossfn == PerceptualLoss().to(self.device)
            self.G_lossfn.vgg = self.model_to_device(self.G_lossfn.vgg)
            self.G_lossfn.lossfn = self.G_lossfn.lossfn.to(self.device)
        elif G_lossfn_type == 'compound':
            self.G_lossfn = CompoundLoss().to(self.device)   
        
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ------------------------------------------------
    # 윤석추가 / Edge loss (SobelEdgeLoss) 사용 여부 확인
        # ------------------------------------------------
        self.use_edge_loss = self.opt_train.get('edge_loss', False)
        if self.use_edge_loss:
            self.edge_loss_fn = SobelEdgeLoss().to(self.device)
        else:
            self.edge_loss_fn = None

        self.use_hf_loss = self.opt_train.get('hf_loss', False)
        if self.use_hf_loss:
            self.hf_loss_fn = HighFrequencyLoss().to(self.device)
        else:
            self.hf_loss_fn = None
            
        self.use_chair_network_loss = self.opt_train.get('chair_network',False)
        if self.use_chair_network_loss:
            self.chair_fft_loss_fn = FFTLoss().to(self.device)
        else : 
            self.chair_fft_loss_fn = None
            
        self.use_lowlight_loss = self.opt_train.get('lowlight_loss',False)
        if self.use_lowlight_loss:
            self.lowlight_color_loss_fn = L_color().to(self.device)
            self.lowlight_spatial_loss_fn = L_spa().to(self.device)
            self.lowlight_exposure_loss_fn = L_exp(128,0.6).to(self.device)
            self.lowlight_tv_loss_fn = L_TV().to(self.device)
        else : 
            self.lowlight_color_loss_fn = None
            self.lowlight_spatial_loss_fn = None
            self.lowlight_exposure_loss_fn = None
            self.lowlight_tv_loss_fn = None
            
        self.use_lowlight_loss2 = self.opt_train.get('lowlight_loss2',False)
        if self.use_lowlight_loss2:
            self.l1_loss_fn = nn.L1Loss().to(self.device)
        else :
            self.l1_loss_fn = None

        self.use_dehaze_loss = self.opt_train.get('dehaze_proposed',False)
        if self.use_dehaze_loss:
            self.dehaze_loss_fn = FFTLoss().to(self.device)
        else : 
            self.dehaze_loss_fn = None
        
        self.use_dehaze_loss_rdsr33_2 = self.opt_train.get('dehaze_proposed_rdsr33_2',False)
        if self.use_dehaze_loss_rdsr33_2:
            self.dehaze_loss_fn_rdsr33_2 = FFTLoss().to(self.device)
        else : 
            self.dehaze_loss_fn_rdsr33_2 = None
        
    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'], capturable=True)
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'], capturable=True)
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.append(CosineAnnealingRestartCyclicLR(optimizer=self.G_optimizer,
                                                            periods=self.opt_train['G_scheduler_periods'],
                                                            restart_weights=self.opt_train['G_scheduler_restart_weights'],
                                                            eta_mins=self.opt_train['G_scheduler_eta_min']
                                                            ))
        
            
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealing':
            self.schedulers.append(lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_T_max'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        
        # elif self.opt_train['G_scheduler_type'] == 'GradualWarmupCosine':
        #     warmup_iters = 100000
        #     total_iters = self.opt_train['G_scheduler_T_max']
        #     eta_min = self.opt_train['G_scheduler_eta_min']
            
        #     # warmup 이후 적용할 cosine annealing scheduler 생성 (전체 에폭수에서 warmup 에폭 제외)
        #     cosine_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
        #                                                     T_max=total_iters - warmup_iters,
        #                                                     eta_min=eta_min)
        #     # GradualWarmupScheduler 생성
        #     self.schedulers.append(GradualWarmupScheduler(self.G_optimizer,
        #                                                 multiplier=1,
        #                                                 total_epoch=warmup_iters,
        #                                                 after_scheduler=cosine_scheduler))
        
            
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    def feed_data_freq(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)
            batch = (self.H).shape[0]
            fft_dim = (-2, -1)
            fft_H = torch.fft.rfftn(self.H, dim=fft_dim, norm='ortho')
            fft_H = torch.stack((fft_H.real, fft_H.imag), dim=-1)
            fft_H = fft_H.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            fft_H = fft_H.view((batch, -1,) + fft_H.size()[3:])
            self.HF=fft_H

    def feed_data_patch(self, data, need_H=True,patch_num=4):
        self.L = data['L'].to(self.device)
        # print(self.L.shape)
        
        self.res_h=(self.L).shape[2]
        self.res_w = self.L.shape[3]
        self.L_patchs=[]

        for i in range (0,patch_num):
            for j in range(0, patch_num):
                self.L_patchs.append(self.L[:,:,i*(self.res_h//patch_num):(i+1)*(self.res_h//patch_num),j*(self.res_w//patch_num):(j+1)*(self.res_w//patch_num)])
        # print(self.L_patchs[0].shape)
        if need_H:
            self.H = data['H'].to(self.device)
     
            
    def feed_data_pad_patch(self, data, need_H=True,patch_num=4):
        self.L = data['L'].to(self.device)
        _, _, h, w = self.L.shape
       
        # patch 크기 계산 (ceil)
        self.ph = math.ceil(h / 256) * 256 - h
        self.pw = math.ceil(w / 256) * 256 - w

        if self.ph > 0 or self.pw > 0:
            self.L = F.pad(self.L, (0, self.pw, 0, self.ph), mode='reflect')
        
        self.res_h = self.L.shape[2] 
        self.res_w = self.L.shape[3]
        
        self.L_patchs=[]

        for i in range (0,patch_num):
            for j in range(0, patch_num):
                self.L_patchs.append(self.L[:,:,i*(self.res_h//patch_num):(i+1)*(self.res_h//patch_num),j*(self.res_w//patch_num):(j+1)*(self.res_w//patch_num)])

        if need_H:
            self.H = data['H'].to(self.device)  

    def feed_data_adaptive_patch(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.res_h=(self.L).shape[2]
        self.res_w = self.L.shape[3]

        if self.res_h < 512 : self.patch_num_h=1
        elif self.res_h >= 512 and self.res_h < 1024: self.patch_num_h = 2
        elif self.res_h >= 1024 and self.res_h < 2048: self.patch_num_h = 4
        else : self.patch_num_h=8

        if self.res_w < 512 : self.patch_num_w=1
        elif self.res_w >= 512 and self.res_w < 1024: self.patch_num_w = 2
        elif self.res_w >= 1024 and self.res_w < 2048: self.patch_num_w = 4
        else : self.patch_num_w=8

        self.L_patchs=[]

        for i in range (0,self.patch_num_h):
            for j in range(0, self.patch_num_w):
                self.L_patchs.append(self.L[:,:,i*(self.res_h//self.patch_num_h):(i+1)*(self.res_h//self.patch_num_h),j*(self.res_w//self.patch_num_w):(j+1)*(self.res_w//self.patch_num_w)])

        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # print(self.L.shape)
        self.E = self.netG(self.L)
        
    def negG_forward_promptcir(self):
        patch_padded, H_pad, W_pad = pad_to_mod32(self.L)
        result_padded = self.netG(patch_padded)
        result_unpadded = unpad_from_mod32(result_padded, H_pad, W_pad)
        self.E = result_unpadded

    def negG_forward_uformer(self):
        patch_padded, H_pad, W_pad = pad_to_square(self.L)
        result_padded = self.netG(patch_padded)
        result_unpadded = unpad_square(result_padded, H_pad, W_pad)
        self.E = result_unpadded
    
    def negG_forward_restormer(self):
        patch_padded, H_pad, W_pad = pad_to_mod2(self.L)
        result_padded = self.netG(patch_padded)
        result_unpadded = unpad_from_mod2(result_padded, H_pad, W_pad)
        self.E = result_unpadded
    
    def negG_forward_drunet(self):
        patch_padded, H_pad, W_pad = pad_to_mod8(self.L)
        result_padded = self.netG(patch_padded)
        result_unpadded = unpad_from_mod8(result_padded, H_pad, W_pad)
        self.E = result_unpadded
    
    def negG_forward_fbcnn(self):
        patch_padded, H_pad, W_pad = pad_to_mod8(self.L)
        result_padded = self.netG(patch_padded)
        result_unpadded = unpad_from_mod8(result_padded, H_pad, W_pad)
        self.E = result_unpadded
    
    def netG_forward_vaunet(self):
        self.E, self.F, self.mu, self.log_var = self.netG(self.L)

    def netG_forward_adaptive_patch_vaunet(self):
        result_patchs=[]
        mu_sum = 0.0
        log_var_sum = 0.0
        for i, patch in enumerate(self.L_patchs):
            patch_out,_,mu_out,log_var_out=self.netG(patch)
            result_patchs.append(patch_out)
            mu_sum+=mu_out
            log_var_sum+=log_var_out

        merged_img=torch.tensor([]).to(self.device)
        for i in range(0,self.patch_num_h):
            merged_line=torch.tensor([]).to(self.device)
            for j in range(0,self.patch_num_w):
                merged_line=torch.cat((merged_line,result_patchs[self.patch_num_w*i+j]),axis=3)
            merged_img=torch.cat((merged_img,merged_line),axis=2)
        self.E = merged_img
        self.mu = mu_sum/len(result_patchs)
        self.log_var = log_var_sum/len(result_patchs)

##########################################################################
#prmptcir
    def netG_forward_test_promptcir_patch(self, patch_num=4):
        result_patchs = []
        for i, patch in enumerate(self.L_patchs):
            # 1) 32 배수가 아니라면 패딩
            patch_padded, H_pad, W_pad = pad_to_mod32(patch)
            # 2) 모델 통과
            result_padded = self.netG(patch_padded)
            # 3) 패딩 제거(원래 크기로 복원)
            result_unpadded = unpad_from_mod32(result_padded, H_pad, W_pad)
            # 4) 리스트에 저장
            result_patchs.append(result_unpadded)

        # ----------------------------------
        # 이제 기존처럼 패치 병합
        merged_img = torch.tensor([]).to(self.device)
        for i in range(0, patch_num):
            merged_line = torch.tensor([]).to(self.device)
            for j in range(0, patch_num):
                merged_line = torch.cat((merged_line, result_patchs[patch_num * i + j]), dim=3)
            merged_img = torch.cat((merged_img, merged_line), dim=2)

        self.E = merged_img
        
##########################################################################
#uformer
    def netG_forward_test_uformer_patch(self, patch_num=4):
        result_patchs = []
        for i, patch in enumerate(self.L_patchs):
            # 1) 32 배수가 아니라면 패딩
            patch_padded, H_pad, W_pad = pad_to_square(patch)
            # 2) 모델 통과
            result_padded = self.netG(patch_padded)
            # 3) 패딩 제거(원래 크기로 복원)
            result_unpadded = unpad_square(result_padded, H_pad, W_pad)
            # 4) 리스트에 저장
            result_patchs.append(result_unpadded)

        # ----------------------------------
        # 이제 기존처럼 패치 병합
        merged_img = torch.tensor([]).to(self.device)
        for i in range(0, patch_num):
            merged_line = torch.tensor([]).to(self.device)
            for j in range(0, patch_num):
                merged_line = torch.cat((merged_line, result_patchs[patch_num * i + j]), dim=3)
            merged_img = torch.cat((merged_img, merged_line), dim=2)

        self.E = merged_img
##########################################################################
#restormer
    def netG_forward_test_restormer_patch(self, patch_num=4):
        result_patchs = []
        for i, patch in enumerate(self.L_patchs):
            # 1) 32 배수가 아니라면 패딩
            patch_padded, H_pad, W_pad = pad_to_mod2(patch)
            # 2) 모델 통과
            result_padded = self.netG(patch_padded)
            # 3) 패딩 제거(원래 크기로 복원)
            result_unpadded = unpad_from_mod2(result_padded, H_pad, W_pad)
            # 4) 리스트에 저장
            result_patchs.append(result_unpadded)

        # ----------------------------------
        # 이제 기존처럼 패치 병합
        merged_img = torch.tensor([]).to(self.device)
        for i in range(0, patch_num):
            merged_line = torch.tensor([]).to(self.device)
            for j in range(0, patch_num):
                merged_line = torch.cat((merged_line, result_patchs[patch_num * i + j]), dim=3)
            merged_img = torch.cat((merged_img, merged_line), dim=2)

        self.E = merged_img
##########################################################################
#drunet, fbcnn
    def netG_forward_test_drunet_patch(self, patch_num=4):
        result_patchs = []
        for i, patch in enumerate(self.L_patchs):
            patch_padded, H_pad, W_pad = pad_to_mod8(patch)
            # 2) 모델 통과
            result_padded = self.netG(patch_padded)
            # 3) 패딩 제거(원래 크기로 복원)
            result_unpadded = unpad_from_mod8(result_padded, H_pad, W_pad)
            # 4) 리스트에 저장
            result_patchs.append(result_unpadded)

        # ----------------------------------
        # 이제 기존처럼 패치 병합
        merged_img = torch.tensor([]).to(self.device)
        for i in range(0, patch_num):
            merged_line = torch.tensor([]).to(self.device)
            for j in range(0, patch_num):
                merged_line = torch.cat((merged_line, result_patchs[patch_num * i + j]), dim=3)
            merged_img = torch.cat((merged_img, merged_line), dim=2)

        self.E = merged_img
        
##########################################################################

    def netG_forward_test_patch(self,patch_num=4):
        result_patchs=[]
        for i, patch in enumerate(self.L_patchs):
            result=self.netG(patch)
            result_patchs.append(result)

        merged_img=torch.tensor([]).to(self.device)
        for i in range(0,patch_num):
            merged_line=torch.tensor([]).to(self.device)
            for j in range(0,patch_num):
                merged_line=torch.cat((merged_line,result_patchs[patch_num*i+j]),axis=3)
            merged_img=torch.cat((merged_img,merged_line),axis=2)
        self.E = merged_img
        
        


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        
        #윤석 추가
        if hasattr(self, 'edge_loss_fn') and self.edge_loss_fn is not None:
            edge_loss = (1 - self.G_lossfn_weight) * self.edge_loss_fn(self.E, self.H)
            G_loss = (self.G_lossfn_weight * self.G_lossfn(self.E, self.H)) + edge_loss
            # print("this is default loss + edge_loss")
            
        elif hasattr(self, 'hf_loss_fn') and self.hf_loss_fn is not None:
            hf_loss = (1 - self.G_lossfn_weight) * self.hf_loss_fn(self.E, self.H)
            G_loss = (self.G_lossfn_weight * self.G_lossfn(self.E, self.H)) + hf_loss
            # print("this is default loss + hf_loss")
            
        elif hasattr(self, 'lowlight_loss_fn') and self.hf_loss_fn is not None:
            hf_loss = (1 - self.G_lossfn_weight) * self.hf_loss_fn(self.E, self.H)
            G_loss = (self.G_lossfn_weight * self.G_lossfn(self.E, self.H)) + hf_loss
            # print("this is default loss + hf_loss")
        
        elif all( hasattr(self, loss_attr) and getattr(self, loss_attr) is not None
                 for loss_attr in [
                     'lowlight_color_loss_fn',
                     'lowlight_spatial_loss_fn',
                     'lowlight_exposure_loss_fn',
                     'lowlight_tv_loss_fn']):
            enhanced_image_1, enhanced_image, A = self.E
            loss_tv = 200 * self.lowlight_tv_loss_fn(A)
            loss_spatial = torch.mean(self.lowlight_spatial_loss_fn(enhanced_image, self.L))
            loss_exp = 10 * torch.mean(self.lowlight_exposure_loss_fn(enhanced_image))
            loss_color = 5 * torch.mean(self.lowlight_color_loss_fn(enhanced_image))
            
            loss_org = (self.G_lossfn_weight * self.G_lossfn(enhanced_image, self.H))
            
            # G_loss = loss_tv + loss_exp + loss_color + loss_charbonnier
            G_loss = loss_tv + loss_spatial + loss_exp + loss_color 
            # G_loss = loss_org
        
        elif hasattr(self, 'l1_loss_fn') and self.l1_loss_fn is not None:
            G_loss = (self.G_lossfn_weight * self.G_lossfn(self.E[1], self.H)) + (self.G_lossfn_weight * self.l1_loss_fn(self.E[0], self.H))
        
        elif hasattr(self, 'chair_fft_loss_fn') and self.chair_fft_loss_fn is not None:
            H2 = F.interpolate(self.H, scale_factor=0.5, mode='bilinear')
            H4 = F.interpolate(self.H, scale_factor=0.25, mode='bilinear')
            
            l1_loss_1 = self.G_lossfn(self.E[0], H4)
            l1_loss_2 = self.G_lossfn(self.E[1], H2)
            l1_loss_3 = self.G_lossfn(self.E[2], self.H)
            sum_l1 = l1_loss_1 + l1_loss_2 + l1_loss_3
            
            fft_loss_1 = self.chair_fft_loss_fn(self.E[0], H4)
            fft_loss_2 = self.chair_fft_loss_fn(self.E[1], H2)
            fft_loss_3 = self.chair_fft_loss_fn(self.E[2], self.H)
            sum_fft = fft_loss_1 + fft_loss_2 + fft_loss_3 
            G_loss = sum_l1 + 0.1 * sum_fft
            # print("this is ChaIR loss(l1+fft)")
            
        elif hasattr(self, 'dehaze_loss_fn') and self.dehaze_loss_fn is not None:            
            l1_loss = self.G_lossfn(self.E, self.H)
            fft_loss = self.dehaze_loss_fn(self.E, self.H)
            
            G_loss = l1_loss + 0.1 * fft_loss
        
        elif hasattr(self, 'dehaze_loss_fn_rdsr33_2') and self.dehaze_loss_fn_rdsr33_2 is not None:            
            l1_loss1 = self.G_lossfn(self.E[0], self.H)
            l1_loss2 = self.G_lossfn(self.E[1], self.H)
            l1_loss3 = self.G_lossfn(self.E[2], self.H)
            l1_loss4 = self.G_lossfn(self.E[3], self.H)
            l1_sum = l1_loss1+l1_loss2+l1_loss3+l1_loss4
            
            fft_loss1 = self.dehaze_loss_fn_rdsr33_2(self.E[0], self.H)
            fft_loss2 = self.dehaze_loss_fn_rdsr33_2(self.E[1], self.H)
            fft_loss3 = self.dehaze_loss_fn_rdsr33_2(self.E[2], self.H)
            fft_loss4 = self.dehaze_loss_fn_rdsr33_2(self.E[3], self.H)
            fft_sum = fft_loss1+fft_loss2+fft_loss3+fft_loss4
            
            G_loss = l1_sum + 0.1 * fft_sum
            
        else :
            # if else 없이 기존에 있던 내용
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            # print("this is single default loss")
            
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters_vaunet(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward_vaunet()
        w_kl = 0.001
        w_freq = 0.565
        w_sr = 0.435


        loss_KL = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - (self.log_var.exp()), dim=1), dim=0)
        # G_loss = self.G_lossfn_weight * (self.G_lossfn(self.E, self.H)* w_sr +self.F_lossfn(self.F,self.HF) * w_freq + loss_KL * w_kl )
        G_loss = self.G_lossfn_weight * (self.G_lossfn(self.E, self.H) + self.F_lossfn(self.F, self.HF) + loss_KL)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        
        if self.opt['netG']['net_type'] =="promptcir":
            with torch.no_grad():
                self.negG_forward_promptcir()
        
        elif self.opt['netG']['net_type'] =="uformer":
            with torch.no_grad():
                # print(self.L.shape)
                self.negG_forward_uformer()
                # self.negG_forward_restormer()
        elif self.opt['netG']['net_type'] =="restormer":
            with torch.no_grad():
                self.negG_forward_restormer()
        
        elif self.opt['netG']['net_type'] =="drunet":
            with torch.no_grad():
                self.negG_forward_drunet()
        
        elif self.opt['netG']['net_type'] =="fbcnn":
            with torch.no_grad():
                self.negG_forward_fbcnn()
                
        else : 
            with torch.no_grad():
                self.netG_forward()    
        
        self.netG.train()

    def test_patch(self,patch_num=4):
        self.netG.eval()
        
        if self.opt['netG']['net_type'] =="promptcir":
            with torch.no_grad():
                # self.netG_forward()
                self.netG_forward_test_promptcir_patch(patch_num=patch_num)
        
        elif self.opt['netG']['net_type'] =="uformer":
            with torch.no_grad():
                # self.netG_forward()
               self.netG_forward_test_uformer_patch(patch_num=patch_num)
        
        elif self.opt['netG']['net_type'] =="restormer":
            with torch.no_grad():
                self.netG_forward_test_restormer_patch(patch_num=patch_num)
                
        elif self.opt['netG']['net_type'] =="drunet":
            with torch.no_grad():
                self.netG_forward_test_patch(patch_num=patch_num)
        
        elif self.opt['netG']['net_type'] =="fbcnn":
            with torch.no_grad():
                self.netG_forward_test_drunet_patch(patch_num=patch_num)
        
        else:
            with torch.no_grad():
                # self.netG_forward()
                self.netG_forward_test_patch(patch_num=patch_num)
        self.netG.train()
    
    def test_jpeg_patch(self,patch_num=4):
        self.netG.eval()
        with torch.no_grad():
            # self.netG_forward()
            self.netG_forward_test_patch(patch_num=patch_num)
        
        self.netG.train()
        
        
    def test_adaptive_patch(self):
        self.netG.eval()
        with torch.no_grad():
            # self.netG_forward()
            self.netG_forward_adaptive_patch_vaunet()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict
  
    
    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals_chair(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E[2].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict
    
    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals_rdsr33_2(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E[3].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict
    
    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals_lowlight(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E[1].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict
    
  


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
