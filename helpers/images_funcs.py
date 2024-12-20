import numpy as np
import cv2
import PIL
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


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
    im_alpha = Image.fromarray(im_alpha, 'RGBA')
    return im_alpha


def make_transparent_v2(image, mask=None, alpha_thr=5, cut_largest_contour=False, n_objects=1):
    def sigmoid(x, slope, bias):
        return np.exp(slope*x - bias) / (1 + np.exp(slope*x - bias))
    image = np.array(image)
    fp_im = image.astype(np.float32)

    alpha_norm = (image.max(axis=-1, keepdims=True) / 255)
    if mask is not None:
        alpha_norm = np.clip(alpha_norm + 0.5*mask[..., None]/255., 0, 1)
    alpha_norm = alpha_norm/alpha_norm.max()
    alpha_norm_ = (alpha_norm**1.2 * 255).astype(np.uint8)
    # alpha_norm_uint[alpha_norm_uint <= 5] = 0
    alpha_norm_ = alpha_norm_ * sigmoid(alpha_norm_, 0.35, 5)
    alpha_norm_uint = alpha_norm_.astype(np.uint8)
    if cut_largest_contour:
        contour_mask = np.zeros_like(alpha_norm_uint)

        contours, _ = cv2.findContours((alpha_norm_uint > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        max_areas_idxs = np.argsort(areas)[-n_objects:]
        largest_contours = [contours[idx] for idx in max_areas_idxs]

        for contour in largest_contours:
            contour_mask = cv2.fillPoly(contour_mask, [contour], 1)
        kernel = np.ones((5, 5), np.uint8)
        contour_mask_dilated = cv2.dilate(contour_mask, kernel, iterations=5)

        alpha_norm_uint[contour_mask_dilated < 0.5] = 0

    fp_im = fp_im / (alpha_norm + 1e-10)
    fp_im = fp_im.astype(np.uint8)
    fp_im = np.concatenate([fp_im, alpha_norm_uint], axis=-1)

    fp_im_ = fp_im.copy()
    fp_im_[..., :-1] = 0
    fp_im_blurred = blur(fp_im_, 3)

    layered_image = Image.new("RGBA", image.shape[:-1])
    # if power_alpha:
    #     image[..., -1] = (((image[..., -1] / 255) ** power_alpha) * 255).astype(np.uint8)
    for i in range(4):
        layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im_blurred))
    for i in range(6):
        layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im_))
    layered_image = Image.alpha_composite(layered_image, Image.fromarray(fp_im))
    return layered_image