import numpy as np
import cv2
import PIL


def blur(image, kernel_size=5):
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.GaussianBlur(kernel_size))
    image = np.array(image)
    return image


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


def make_transparent_v2(image):
    image = np.array(image)
    alpha_norm = (image.max(axis=-1, keepdims=True) / 255)
    alpha_norm = alpha_norm/alpha_norm.max()
    alpha_norm_uint = (alpha_norm**1.2 * 255).astype(np.uint8)
    alpha_norm_uint[alpha_norm_uint <= 5] = 0
    fp_im = fp_im / (alpha_norm + 1e-10)
    fp_im = fp_im.astype(np.uint8)
    fp_im = np.concatenate([fp_im, alpha_norm_uint], axis=-1)

    fp_im_ = fp_im.copy()
    fp_im_[..., :-1] = 0
    fp_im_blurred = blur(fp_im_, 3)

    layered_image = Image.new("RGBA", image.shape[:-1])
    for i in range(4):
        layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im_blurred))
    for i in range(6):
        layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im_))
    layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im))
    return layered_image