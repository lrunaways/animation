import numpy as np
import cv2
import PIL


def make_transparent(im_alpha, str=1.2):
    im_alpha = np.array(im_alpha).copy()
    im_alpha[im_alpha.sum(axis=2) < 15] = 0
    x = im_alpha.mean(axis=2, keepdims=True)
    x = x**(1/np.log(x+2)+str)

    alpha_channel = np.clip(x, 0, 255)
    ksize = 3
    alpha_channel = cv2.blur(alpha_channel, (ksize, ksize), cv2.BORDER_DEFAULT)
    alpha_channel = alpha_channel[..., None]

    im_alpha = np.concatenate([im_alpha, alpha_channel.astype(np.uint8)], axis=-1)
    im_alpha = PIL.Image.fromarray(im_alpha, 'RGBA')
    return im_alpha