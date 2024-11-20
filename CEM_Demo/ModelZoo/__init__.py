import os
import torch
from collections import OrderedDict
MODEL_DIR = './CEM_Demo/ModelZoo/models/'


NN_LIST = [
    'SRCNN',  #
    'FSRCNN',  #
    'ESPCN',  # https://github.com/yjn870/ESPCN-pytorch
    'DRRN',  # https://github.com/jt827859032/DRRN-pytorch
    'LapSRN',  # https://github.com/twtygqyy/pytorch-LapSRN
    'DDBPN',  #
    'SRResNet',  # https://github.com/twtygqyy/pytorch-SRResNet
    'EDSR',
    'MSRN',
    'RCAN',
    'RCAN_half_ca',
    'RCAN_no_ca',
    'RCAN_pooling4',
    'RCAN_pooling16',
    'RCAN_pooling2x2',
    'RCAN_pooling3x3',

    # 'RCANca',
    'CARN',
    'RRDBNet',
    'SRDenseNet',  # https://github.com/wxywhu/SRDenseNet-pytorch
    'SAN',  # https://github.com/daitao/SAN
    'RNAN',  #
    'CSNLN',
    'DRLN',
    'PAN',
    'IMDN',
    'RFDN',
    'SRDenseNet17',
    'SwinIR',
    'SwinIR-for-HAT',
    'SwinIR-Real-PSNR',
    'HAT',
    'HAT-L',
    'RealESRGAN',
    'RealESRNet',
    'MSRGAN',
    'XRestormer_SR',

    'DnCNN',
    'IRCNN',
    'FFDNet',
    'DRUNet',
    'MPRNet',
    'SCUNet',
    'SADNet',
    'MIRNetV2',
    'RIDNet',
    'SwinIR_DN',
    'Restormer_DN',
    'XRestormer_DN',
    'RCAN_DN',
    'SRResNet_DN',


    'HINet',
    'Restormer_DR',
    'XRestormer_DR',
    'MPRNet_DR',
    'SRResNet_DR',
    'RCAN_DR',
    'PReNet',
    'SwinIR_DR',

    'SRResNet_mix',
    'SwinIR_mix',
    'RCAN_mix',

    'SRResNet_MIO',
    'SwinIR_MIO',
]


MODEL_LIST = {
    'SRCNN': {
        'Base': 'SRCNN_15000.pth',
    },
    'FSRCNN': {
        'Base': 'FSRCNN-16-6_15000.pth',
    },
    'FSRCNN-16-6': {
        'Base': 'FSRCNN-16-6_15000.pth',
    },
    'FSRCNN-32-6': {
        'Base': 'FSRCNN-32-6_15000.pth',
    },
    'FSRCNN-48-6': {
        'Base': 'FSRCNN-48-6_15000.pth',
    },
    'FSRCNN-64-6': {
        'Base': 'FSRCNN-64-6_15000.pth',
    },
    'FSRCNN-80-6': {
        'Base': 'FSRCNN-80-6_15000.pth',
    },
    'FSRCNN-96-6': {
        'Base': 'FSRCNN-96-6_15000.pth',
    },
    'FSRCNN-112-6': {
        'Base': 'FSRCNN-112-6_15000.pth',
    },
    'FSRCNN-128-6': {
        'Base': 'FSRCNN-128-6_15000.pth',
    },
    'FSRCNN-48-2': {
        'Base': 'FSRCNN-48-2_15000.pth',
    },
    'FSRCNN-48-4': {
        'Base': 'FSRCNN-48-4_15000.pth',
    },
    'FSRCNN-48-8': {
        'Base': 'FSRCNN-48-8_15000.pth',
    },
    'FSRCNN-48-10': {
        'Base': 'FSRCNN-48-10_15000.pth',
    },
    'FSRCNN-48-12': {
        'Base': 'FSRCNN-48-12_15000.pth',
    },
    'FSRCNN-48-14': {
        'Base': 'FSRCNN-48-14_15000.pth',
    },
    'FSRCNN-48-16': {
        'Base': 'FSRCNN-48-16_15000.pth',
    },
    'LapSRN': {
        'Base': 'LapSRN_15000.pth',
    },
    'DDBPN': {
        'Base': 'DBPN_x4_new.pth',
    },
    'EDSR': {
        'Base': 'EDSR-64-16_15000.pth',
    },
    'EDSR-64-10': {
        'Base': 'EDSR-64-10_15000.pth',
    },
    'EDSR-64-12': {
        'Base': 'EDSR-64-12_15000.pth',
    },
    'EDSR-64-14': {
        'Base': 'EDSR-64-14_15000.pth',
    },
    'EDSR-64-16': {
        'Base': 'EDSR-64-16_15000.pth',
    },
    'EDSR-64-18': {
        'Base': 'EDSR-64-18_15000.pth',
    },
    'EDSR-64-20': {
        'Base': 'EDSR-64-20_15000.pth',
    },
    'EDSR-64-22': {
        'Base': 'EDSR-64-22_15000.pth',
    },
    'EDSR-64-24': {
        'Base': 'EDSR-64-24_15000.pth',
    },
    'EDSR-32-16': {
        'Base': 'EDSR-32-16_15000.pth',
    },
    'EDSR-96-16': {
        'Base': 'EDSR-96-16_15000.pth',
    },
    'EDSR-128-16': {
        'Base': 'EDSR-128-16_15000.pth',
    },
    'EDSR-160-16': {
        'Base': 'EDSR-160-16_15000.pth',
    },
    'EDSR-192-16': {
        'Base': 'EDSR-192-16_15000.pth',
    },
    'EDSR-224-16': {
        'Base': 'EDSR-224-16_15000.pth',
    },
    'EDSR-256-16': {
        'Base': 'EDSR-256-16_15000.pth',
    },
    'MSRN': {
        'Base': 'MSRN_x4.pt',
    },
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'RCAN_DN': {
        'Base': 'RCAN_DN.pth',
    },
    'RCAN_half_ca': {
        'Base': 'RCAN_half.pth',
    },
    'RCAN_no_ca': {
        'Base': 'RCAN_noCA.pth',
    },
    'RCAN_pooling4': {
        'Base': 'RCAN_pooling4.pth',
    },
    'RCAN_pooling16': {
        'Base': 'RCAN_pooling16.pth',
    },
    'RCAN_pooling2x2': {
        'Base': 'RCAN_pooling2x2.pth',
    },
    'RCAN_pooling3x3': {
        'Base': 'RCAN_pooling3x3.pth',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RealESRGAN_x4plus.pth',
    },
    'SRDenseNet': {
        'Base': 'SRDenseNet_15000.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'CSNLN': {
        'Base': 'CSNLN_x4.pt',
    },
    'DRLN': {
        'Base': 'DRLN_BIX4.pt',
    },
    'PAN': {
        'Base': 'PANx4_DF2K.pth',
    },
    'IMDN': {
        'Base': 'IMDN_x4_new.pth',
    },
    'RFDN': {
        'Base': 'RFDN_AIM.pth',
    },
    'SwinIR': {
        'Base': '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
    },
    'SwinIR-for-HAT': {
        'Base': 'SwinIR_SRx4_official.pth',
    },
    'SwinIR-Real-PSNR': {
        'Base': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth',
    },
    'HAT': {
        'Base': 'HAT_SRx4.pth',
    },
    'HAT-L': {
        'Base': 'HAT-L_SRx4_ImageNet-pretrain.pth',
    },
    'RealESRNet': {
            'Base': 'RealESRNet_x4plus.pth',
        },
    'SRResNet': {
            'Base': 'MSRResNet_net_g_1000000.pth',
        },
    'XRestormer_SR': {
        'Base': 'xrestormer_sr.pth',
    },
    # denoising
    'DnCNN': {
        'Base': 'dncnn_color_blind.pth',
    },
    'IRCNN': {
        'Base': 'ircnn_color.pth',
    },
    'FFDNet': {
        'Base': 'ffdnet_color.pth',
    },
    'DRUNet': {
        'Base': 'drunet_color.pth',
    },
    'MPRNet': {
        'Base': 'MPRNet_model.pth',
    },
    'SCUNet': {
        'Base': 'scunet_color_50.pth',
    },
    'SADNet': {
        'Base': 'sadnet_color_50.pth',
    },
    'RIDNet': {
        'Base': 'ridnet.pt',
    },
    'MIRNetV2': {
        'Base': 'MIRNetv2_real_denoising.pth',
    },
    'SwinIR_DN': {
        'Base': '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    },
    'Restormer_DN': {
        'Base': 'Restormer_gaussian_color_denoising_sigma50.pth',
    },
    'XRestormer_DN': {
        'Base': 'xrestormer_denoise_300k.pth',
    },
    'SRResNet_DN': {
        'Base': 'SRResNet_DN.pth',
    },
    
    # deraining
    'MPRNet_DR': {
        'Base': 'MPRNet_deraining.pth',
    },
    'HINet': {
        'Base': 'HINet-Rain13k.pth',
    },
    'Restormer_DR': {
        'Base': 'restormer_deraining.pth',
    },
    'XRestormer_DR': {
        'Base': 'xrestormer_derain.pth',
    },
    'RCAN_DR': {
        'Base': 'RCAN_DR.pth',
    },
    'PReNet': {
        'Base': 'PReNet6.pth',
    },
    'SRResNet_DR': {
        'Base': 'SRResNet_DR.pth',
    },
    'SwinIR_DR': {
        'Base': 'SwinIR_DR.pth',
    },
    #realSR
    'RealESRGAN': {
        'Base': 'RealESRGAN_x4plus.pth',
    },
    'MSRGAN': {
        'Base': 'MSRGAN_net_g_400000.pth',
    },

    'SRResNet_mix': {
        'Base': 'mix_SRResNet.pth',
    },
    'SwinIR_mix': {
        'Base': 'mix_SwinIR.pth',
    },
    'RCAN_mix': {
        'Base': 'mix_RCAN.pth',
    },

    'SRResNet_MIO': {
        'Base': 'SRResNet_M.pth',
    },
    'SwinIR_MIO': {
        'Base': 'SwinIR_M.pth',
    },
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))



def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:
        if model_name.startswith('ESPCN') and len(model_name.split('-')) == 3:
            from .NN.espcn import ESPCN
            _, width, depth = model_name.split('-')
            net = ESPCN(scale_factor=factor, num_channels=num_channels, width=int(width), depth=int(depth))

        elif model_name.startswith('EDSR') and len(model_name.split('-')) == 3:
            from .NN.edsr import EDSR
            _, width, depth = model_name.split('-')
            net = EDSR(num_channels=num_channels, factor=factor, width=int(width), depth=int(depth))

        elif model_name.startswith('FSRCNN') and len(model_name.split('-')) == 3:
            from .NN.fsrcnn import FSRCNN
            _, width, depth = model_name.split('-')
            net = FSRCNN(scale_factor=factor, num_channels=num_channels, s=int(width), m=int(depth))

        else:

            if model_name == 'SRCNN':
                from .NN.srcnn import SRCNN
                net = SRCNN(num_channels=num_channels, factors=factor)

            elif model_name == 'FSRCNN':
                from .NN.fsrcnn import FSRCNN
                net = FSRCNN(scale_factor=factor, num_channels=num_channels)

            elif model_name == 'ESPCN':
                from .NN.espcn import ESPCN
                net = ESPCN(scale_factor=factor, num_channels=num_channels)

            elif model_name == 'DRRN':
                from .NN.drrn import DRRN
                net = DRRN(num_channels=num_channels, factors=factor)

            elif model_name == 'LapSRN':
                from .NN.lapsrn import Net
                net = Net(num_channels=num_channels)

            elif model_name == 'DDBPN':
                from .NN.ddbpn import Net
                net = Net(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=4)

            elif model_name == 'EDSR':
                from .NN.edsr import EDSR
                net = EDSR(num_channels=num_channels, factor=factor)

            elif model_name == 'SRResNet_MIO':
                from .NN.mixsrresnet_arch import SRResNet_mix
                net = SRResNet_mix(upscale=1)
            elif model_name == 'SwinIR_MIO':
                from .NN.mixswinir_arch import SwinIR_mix
                net = SwinIR_mix(upscale=1,img_size=128)

            elif model_name == 'SRResNet_mix':
                from .NN.mixsrresnet_arch import SRResNet_mix
                net = SRResNet_mix(upscale=1)
            elif model_name == 'SwinIR_mix':
                from .NN.mixswinir_arch import SwinIR_mix
                net = SwinIR_mix(upscale=1)
            elif model_name == 'RCAN_mix':
                from .NN.mixrcan_arch import RCAN_mix
                net = RCAN_mix()
            elif model_name == 'MSRN':
                from .NN.msrn import MSRN
                net = MSRN(factor=factor, num_channels=num_channels)

            elif model_name == 'RCAN':
                from .NN.rcan import RCAN
                net = RCAN(factor=factor, num_channels=num_channels)

            elif model_name == 'RCAN_half_ca':
                from .NN.rcan_half_ca_arch import rcan
                net = rcan(num_in_ch=3,num_out_ch=3)

            elif model_name == 'RCAN_no_ca':
                from .NN.rcan_no_ca_arch import rcan
                net = rcan(num_in_ch=3,num_out_ch=3)

            elif model_name == 'RCAN_pooling4':
                from .NN.rcan_pooling4_arch import rcan
                net = rcan(num_in_ch=3,num_out_ch=3)

            elif model_name == 'RCAN_pooling16':
                from .NN.rcan_pooling16_arch import rcan
                net = rcan(num_in_ch=3,num_out_ch=3)

            elif model_name == 'RCANca':
                from .NN.rcanca import RCANca
                net = RCANca(factor=factor, num_channels=num_channels)

            elif model_name == 'RCAN_pooling2x2':
                from .NN.rcan_pooling2x2_arch import rcan_pooling2x2
                net = rcan_pooling2x2(num_in_ch=3,num_out_ch=3)

            elif model_name == 'RCAN_pooling3x3':
                from .NN.rcan_pooling3x3_arch import rcan_pooling3x3
                net = rcan_pooling3x3(num_in_ch=3,num_out_ch=3)

            elif model_name == 'CARN':
                from .CARN.carn import CARNet
                net = CARNet(factor=factor, num_channels=num_channels)

            elif model_name == 'RRDBNet':
                from .NN.rrdbnet import RRDBNet
                net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

            elif model_name == 'SRDenseNet':
                from .NN.srdense_2 import Net
                net = Net(num_channels=num_channels)

            elif model_name == 'SAN':
                from .NN.san import SAN
                net = SAN(factor=factor, num_channels=num_channels)

            elif model_name == 'RNAN':
                from .NN.rnan import RNAN
                net = RNAN(factor=factor, num_channels=num_channels)

            elif model_name == 'CSNLN':
                from .NN.csnln import CSNLN
                net = CSNLN(n_colors=num_channels)

            elif model_name == 'DRLN':
                from .NN.drln import DRLN
                net = DRLN()

            elif model_name == 'PAN':
                from .NN.pan import PAN
                net = PAN()

            elif model_name == 'IMDN':
                from .NN.imdn import IMDN
                net = IMDN()

            elif model_name == 'RFDN':
                from .NN.rfdn import RFDN
                net = RFDN()
            elif model_name == 'HAT':
                from .NN.hat_arch import HAT
                net = HAT()
            elif model_name == 'HAT-L':
                from .NN.hat_arch import HAT
                net = HAT(depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],num_heads= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
            elif model_name == 'XRestormer_SR':
                from .NN.xrestormer_arch import XRestormer
                net = XRestormer(scale=4)

            elif model_name == 'MSRGAN' or model_name == 'SRResNet':
                from .NN.msrresnet_arch import MSRResNet
                net = MSRResNet()

            elif model_name == 'SRDenseNet17':
                from .NN.srdense_17 import SRDenseNet
                net = SRDenseNet()

            elif model_name == 'SwinIR':
                from .NN.swinir_arch import SwinIR
                net = SwinIR(upscale=4,img_size=64,in_chans=3, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

            elif model_name == 'SwinIR-for-HAT':
                from .NN.swinir import SwinIR
                net = SwinIR()

            elif model_name == 'SwinIR-Real-PSNR':
                from .NN.swinir_arch import SwinIR
                net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

            elif model_name == 'RealESRGAN':
                from .NN.realesrgan import RRDBNet
                net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            elif model_name == 'RealESRNet':
                from .NN.realesrgan import RRDBNet
                net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    if model_name == 'RRDBNet' or model_name == 'RealESRGAN' or model_name == 'RealESRNet' or model_name == 'SRResNet' or model_name == 'MSRGAN':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name == 'RCAN_half_ca' or model_name == 'RCAN_pooling4' or model_name == 'RCAN_no_ca' or model_name == 'RCAN_pooling16':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name == 'RCAN_pooling2x2' or model_name == 'RCAN_pooling3x3':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name=='SRResNet_mix' or model_name=='SwinIR_mix' or model_name=='RCAN_mix':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name=='SRResNet_MIO' or model_name=='SwinIR_MIO':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name == 'HAT' or model_name == 'HAT-L' or model_name == 'SwinIR-for-HAT'or model_name == 'SwinIR-Real-PSNR' or model_name == 'XRestormer_SR':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)
        
    elif model_name == 'SwinIR':
        param_key_g = 'params'
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)
    return net

    # denoising model
def get_denoise_model(model_name, factor=1, num_channels=3):

    print(f'Getting Denoise Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:
        if model_name == 'DnCNN':
            from .NN.dncnn import DnCNN
            net = DnCNN()
        elif model_name == 'IRCNN':
            from .NN.IRCNN import IRCNN
            net = IRCNN()
        elif model_name == 'FFDNet':
            from .NN.ffdnet import FFDNet
            net = FFDNet(sigma = 50)
        elif model_name == 'DRUNet':
            from .NN.drunet import UNetRes
            net = UNetRes(in_nc=3+1, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose",noise_level=50)
        elif model_name == 'MPRNet':
            from .NN.mprnet import MPRNet
            net = MPRNet()
        elif model_name == 'SCUNet':
            from .NN.scunet import SCUNet
            net = SCUNet(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
        elif model_name == 'SADNet':
            from .NN.sadnet import SADNET
            net = SADNET()
        elif model_name == 'RCAN_DN':
            from .NN.rcan_denoise_arch import DeRCAN
            net = DeRCAN()
        elif model_name == 'SRResNet_DN':
            from .NN.SRResNet_DN_arch import SRResNet_DN
            net = SRResNet_DN(upscale=1)
        elif model_name == 'SRResNet_MIO':
            from .NN.mixsrresnet_arch import SRResNet_mix
            net = SRResNet_mix(upscale=1)
        elif model_name == 'SwinIR_MIO':
            from .NN.mixswinir_arch import SwinIR_mix
            net = SwinIR_mix(upscale=1,img_size=128)
        elif model_name == 'SRResNet_mix':
            from .NN.mixsrresnet_arch import SRResNet_mix
            net = SRResNet_mix(upscale=1)
        elif model_name == 'SwinIR_mix':
            from .NN.mixswinir_arch import SwinIR_mix
            net = SwinIR_mix(upscale=1)
        elif model_name == 'RCAN_mix':
            from .NN.mixrcan_arch import RCAN_mix
            net = RCAN_mix()
        elif model_name == 'RIDNet':
            from .NN.ridnet import RIDNET
            net = RIDNET()
        elif model_name == 'MIRNetV2':
            from .NN.mirnet_v2_arch import MIRNet_v2
            net = MIRNet_v2()
        elif model_name == 'SwinIR_DN':
            from .NN.swinir_arch import SwinIR
            net = SwinIR(upscale=1, img_size=128, in_chans=3, window_size=8,
                         img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                         mlp_ratio=2, upsampler='', resi_connection='1conv')
        elif model_name == 'Restormer_DN':
            from .NN.restormer_arch import Restormer
            net = Restormer(LayerNorm_type= 'BiasFree')
        elif model_name == 'XRestormer_DN':
            from .NN.xrestormer_arch import XRestormer
            net = XRestormer()
        else:
            raise NotImplementedError()
        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()

def load_denoise_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_denoise_model(model_name)

    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')

    if model_name == 'MPRNet':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        try:
            net.load_state_dict(state_dict["state_dict"])
        except:
            state_dict = state_dict["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
    elif model_name == 'MIRNetV2':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict['params'])

    elif model_name=='SRResNet_MIO' or model_name=='SwinIR_MIO':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name == 'SwinIR_DN' :
        param_key_g = 'params'
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
    elif model_name == 'Restormer_DN' or model_name == 'XRestormer_DN' or model_name == 'RCAN_DN' or model_name=='SRResNet_mix' or model_name=='SwinIR_mix' or model_name=='RCAN_mix' or model_name=='SRResNet_DN':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)
    else:
        # pt = torch.load(state_dict_path)
        # print(pt)
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)
    return net


    # derain model
def get_derain_model(model_name, factor=1, num_channels=3):

    print(f'Getting Derain Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:
        if model_name == 'HINet':
            from .NN.hinet_arch import HINet
            net = HINet()
        elif model_name == 'PReNet':
            from .NN.prenet import PReNet
            net = PReNet()
        elif model_name == 'MPRNet_DR':
            from .NN.MPRNet_DR import MPRNet
            net = MPRNet()
        elif model_name == 'Restormer_DR':
            from .NN.restormer_arch import Restormer
            net = Restormer()
        elif model_name == 'RCAN_DR':
            from .NN.rcan_derain_arch import DRRCAN
            net = DRRCAN(num_in_ch=3,num_out_ch=3)
        elif model_name == 'SRResNet_DR':
            from .NN.mixsrresnet_arch import SRResNet_mix
            net = SRResNet_mix(upscale=1)
        elif model_name == 'SwinIR_DR':
            from .NN.mixswinir_arch import SwinIR_mix
            net = SwinIR_mix(upscale=1)

        elif model_name == 'SRResNet_MIO':
            from .NN.mixsrresnet_arch import SRResNet_mix
            net = SRResNet_mix(upscale=1)
        elif model_name == 'SwinIR_MIO':
            from .NN.mixswinir_arch import SwinIR_mix
            net = SwinIR_mix(upscale=1,img_size=128)

        elif model_name == 'SRResNet_mix':
            from .NN.mixsrresnet_arch import SRResNet_mix
            net = SRResNet_mix(upscale=1)

        elif model_name == 'SwinIR_mix':
            from .NN.mixswinir_arch import SwinIR_mix
            net = SwinIR_mix(upscale=1)
        elif model_name == 'RCAN_mix':
            from .NN.mixrcan_arch import RCAN_mix
            net = RCAN_mix()
        elif model_name == 'XRestormer_DR':
            from .NN.xrestormer_arch import XRestormer
            net = XRestormer()
        else:
            raise NotImplementedError()
        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()

def load_derain_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_derain_model(model_name)

    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')

    if model_name == 'MPRNet_DR':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict["state_dict"])

    elif model_name=='SRResNet_MIO' or model_name=='SwinIR_MIO':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)

    elif model_name == 'Restormer_DR' or model_name == 'XRestormer_DR' or model_name == 'RCAN_DR' or model_name == 'HINet' or model_name=='SRResNet_mix' or model_name=='SRResNet_DR' or model_name=='SwinIR_mix' or model_name=='SwinIR_DR' or model_name=='RCAN_mix':
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(state_dict[keyname], strict=True)
    else:
        # pt = torch.load(state_dict_path)
        # print(pt)
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)
    return net

