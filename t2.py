import numpy as np
import torch

def flatten_rgb_to_bayer(im_rgb_3ch):
    '''
    3 channels to RGGB bayer
    '''
    if isinstance(im_rgb_3ch, np.ndarray):
        # 创建一个空的Bayer格式数组，大小是原图像的两倍
        im_bayer = np.zeros((im_rgb_3ch.shape[0] * 2, im_rgb_3ch.shape[1] * 2), dtype=im_rgb_3ch.dtype)
    elif isinstance(im_rgb_3ch, torch.Tensor):
        # 创建一个空的Bayer格式张量，大小是原图像的两倍
        im_bayer = torch.zeros((im_rgb_3ch.shape[1] * 2, im_rgb_3ch.shape[2] * 2), dtype=im_rgb_3ch.dtype)
    else:
        raise Exception("Input image must be either a numpy ndarray or a torch Tensor.")

    # RGGB Bayer排列
    im_bayer[0::2, 0::2] = im_rgb_3ch[0:, :, 0]  # R
    im_bayer[0::2, 1::2] = im_rgb_3ch[0:, :, 1]  # G
    im_bayer[1::2, 0::2] = im_rgb_3ch[0:, :, 1]  # G
    im_bayer[1::2, 1::2] = im_rgb_3ch[0:, :, 2]  # B

    return im_bayer
            im_bayer = torch.zeros((burst.shape[0],burst.shape[1],burst.shape[2],burst.shape[3] * 2, burst.shape[4] * 2), dtype=burst.dtype)
            
            im_bayer[0::2, 0::2] = burst[0:, :, 0]  # R
            im_bayer[0::2, 1::2] = burst[0:, :, 1]  # G
            im_bayer[1::2, 0::2] = burst[0:, :, 1]  # G
            im_bayer[1::2, 1::2] = burst[0:, :, 2]  # B
