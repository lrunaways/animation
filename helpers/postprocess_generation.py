import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter, ImageChops

BLUR_CONST = (0.1, 0.1)


# def alpha_premultiplied_blur(image, radius):
#     # Separate RGBA channels
#     r, g, b, a = image.split()
#     # Apply premultiplied alpha to RGB channels
#     r = ImageChops.multiply(r, a)
#     g = ImageChops.multiply(g, a)
#     b = ImageChops.multiply(b, a)
#     # Blur the premultiplied RGB channels
#     blurred_channels = [c.filter(ImageFilter.GaussianBlur(radius)) for c in (r, g, b, a)]
#
#     # Merge the channels back together
#     blurred_image = Image.merge('RGBA', blurred_channels)
#     return blurred_image


def shift_image(image, shift_x, shift_y):
    # Check if the image is RGB or RGBA, and create a new image accordingly
    new_image = Image.new(image.mode, image.size, (255, 255, 255) if image.mode == 'RGB' else (255, 255, 255, 0))

    # Paste the original image onto the new image with the specified shifts
    new_image.paste(image, (int(shift_x), int(shift_y)))

    return new_image


def adjust_exposure(image, exposure):
    image_adjusted = np.array(image).copy()
    image_adjusted = image_adjusted / 255.
    image_adjusted = image_adjusted * (2**exposure)
    image_adjusted = (np.clip(image_adjusted, 0, 1) * 255).astype(np.uint8)
    return image_adjusted


def adjust_gamma(image, gamma=1.0):
    # Ensure the gamma value is greater than 0
    if gamma < 0.1:
        gamma = 0.1

    # Apply gamma correction using the formula: output = input ^ (1/gamma)
    gamma_correction = 1.0 / gamma
    adjusted_image = np.power(image / 255.0, gamma_correction) * 255.0

    # Clip values to ensure they are in the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image


# def apply_gaussian_smoothing(image, kernel_size, sigma):
#     # Create a 1D Gaussian kernel for the x-direction
#     kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
#
#     # Create a 1D Gaussian kernel for the y-direction
#     kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
#
#     # Create a 2D Gaussian kernel by taking the outer product of the two 1D kernels
#     kernel_2d = np.outer(kernel_x, kernel_y)
#
#     # Apply the 2D Gaussian kernel to the image
#     smoothed_image = cv2.filter2D(image, -1, kernel_2d)
#
#     return smoothed_image


from PIL import Image, ImageFilter, ImageChops
#TODO: make brighter and saturation
def alpha_premultiplied_blur(image, radius):
    # Separate RGBA channels
    r, g, b, a = image.split()

    # Apply premultiplied alpha to RGB channels
    r = ImageChops.multiply(r, a)
    g = ImageChops.multiply(g, a)
    b = ImageChops.multiply(b, a)

    # Blur the premultiplied RGB channels
    blurred_image = Image.merge('RGBA', [c.filter(ImageFilter.GaussianBlur(radius)) for c in (r, g, b, a)])

    return blurred_image

def pad_to_square(image, pad_color=(0, 0, 0, 0)):
    """
    Pad a CV2 image to make it square.

    Parameters:
    - image: The input image (CV2 format).
    - pad_color: Tuple representing the color to pad with (default is black).

    Returns:
    - The squared image.
    """
    h, w, _ = image.shape

    # Find the maximum dimension
    max_dim = max(h, w)

    # Calculate padding
    pad_vert = max_dim - h
    pad_horiz = max_dim - w

    # Calculate padding on each side
    top_pad = pad_vert // 2
    bottom_pad = pad_vert - top_pad
    left_pad = pad_horiz // 2
    right_pad = pad_horiz - left_pad

    # Create padded image
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=pad_color)

    return padded_image


def increase_brightness_with_alpha(image, factor):
    # Split the image into BGR and alpha channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Convert the BGR image to the HSV color space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Increase the value (brightness) channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)

    # Convert the BGR image back to the HSV color space
    result_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Combine the adjusted BGR channels with the original alpha channel
    result_image = cv2.merge([result_bgr, alpha])

    return result_image


def process(image, image_gen_mask, mask_kernel, blur_kernel=31, shifts_str=1.0, return_pil=False, pad_to_squate=False,
            dark_adjust=0.15, bright_adjust=1.25, exposure_adjust=0.4, postprocess=True, cut_padding=0.1, n_objects_contours=1):
    #TODO: fix image size dependent kernel's sizes
    image[..., :-1] = adjust_exposure(image[..., :-1], exposure_adjust)

    # --------------- Process mask ---------------
    mask_resized = cv2.resize(image_gen_mask, (image.shape[1], image.shape[0]))
    if len(mask_resized.shape) > 2:
        mask_resized = mask_resized[..., 0]
    contours, _ = cv2.findContours((mask_resized > 16).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.concatenate(contours)
    # x_max = min(image.shape[0], all_contours[:, 0, 1].max() + int(image.shape[0]*cut_padding))
    # x_min = max(0, all_contours[:, 0, 1].min() - int(image.shape[0]*cut_padding))
    # y_max = min(image.shape[1], all_contours[:, 0, 0].max() + int(image.shape[1]*cut_padding))
    # y_min = max(0, all_contours[:, 0, 0].min() - int(image.shape[1]*cut_padding))

    hull_image = np.zeros_like(image, dtype=np.uint8)
    hull = cv2.convexHull(all_contours)
    hull_image = cv2.drawContours(hull_image, [hull], -1, (1, 1, 1, 1), thickness=cv2.FILLED)
    hull_mask = (hull_image[..., 0] > 0).astype(np.uint8)

    x_max = min(image.shape[0], all_contours[:, 0, 1].max() + int(image.shape[0]*cut_padding))
    x_min = max(0, all_contours[:, 0, 1].min() - int(image.shape[0]*cut_padding))
    y_max = min(image.shape[1], all_contours[:, 0, 0].max() + int(image.shape[1]*cut_padding))
    y_min = max(0, all_contours[:, 0, 0].min() - int(image.shape[1]*cut_padding))
    # --------------- blur image ---------------
    kernel = [int(image.shape[0]*BLUR_CONST[0]), int(image.shape[0]*BLUR_CONST[1])]
    kernel[0], kernel[1] = kernel[0] + (kernel[0]+1) % 2, \
                           kernel[1] + (kernel[1]+1) % 2

    kernel_ = np.ones((5, 5))
    hull_mask_dilated = cv2.dilate(hull_mask, kernel_, iterations=10)
    blurred_mask = cv2.GaussianBlur(hull_mask_dilated.astype(np.float32), kernel, 0.0)
    blurred_mask = np.sqrt(blurred_mask)

    image_masked = image.copy()
    image_masked[..., -1] = image[..., -1] * blurred_mask
    image_masked_pil = Image.fromarray(image_masked)
    if postprocess:
        # --------------- Blur effects ---------------
        # two dark blurred shadows + one semi-bright blurred halo
        layered_image = Image.new("RGBA", image.shape[:-1])

        dark_masked = adjust_gamma(image_masked, dark_adjust)
        dark_masked_pil = Image.fromarray(dark_masked)
        dark_masked_blurred_pil = alpha_premultiplied_blur(dark_masked_pil, blur_kernel)
        layered_image = Image.alpha_composite(layered_image, shift_image(dark_masked_blurred_pil, 0, 12*shifts_str))
        layered_image = Image.alpha_composite(layered_image, shift_image(dark_masked_blurred_pil, 0, -12*shifts_str))

        bright_masked = adjust_gamma(image_masked, bright_adjust)
        bright_masked = Image.fromarray(bright_masked)
        bright_masked_blurred_pil = alpha_premultiplied_blur(image_masked_pil, blur_kernel)
        converter = ImageEnhance.Color(bright_masked_blurred_pil)
        bright_masked_blurred_pil = converter.enhance(bright_adjust*1.5)
        bright_masked_blurred = np.array(bright_masked_blurred_pil)
        bright_masked_blurred = adjust_gamma(bright_masked_blurred, bright_adjust*1.25)
        bright_masked_blurred_corrected_pil = Image.fromarray(bright_masked_blurred)
        layered_image = Image.alpha_composite(layered_image, shift_image(bright_masked_blurred_corrected_pil, 0, -24*shifts_str))

        # -------------- Blend all together --------------
        layered_image = Image.alpha_composite(layered_image, image_masked_pil)

        bright_masked_blurred[..., -1] = bright_masked_blurred[..., -1] // 6
        bright_masked_blurred_pil = Image.fromarray(bright_masked_blurred)
        layered_image = Image.alpha_composite(layered_image, shift_image(bright_masked_blurred_pil, 0, 0))

        layered_image = np.array(layered_image)
    else:
        layered_image = image_masked
    layered_image = layered_image[x_min:x_max, y_min:y_max]
    layered_image[..., :-1] = adjust_exposure(layered_image[..., :-1], exposure_adjust/2)


    if pad_to_squate:
        layered_image = pad_to_square(layered_image)
    if return_pil:
        layered_image = Image.fromarray(layered_image)
    return layered_image

def pad_and_resize(image, padding):
    original_shape = image.shape[:-1]

    # Pad the image

    if padding < 0:
      padding = abs(padding)
      padded_image = image[padding:-padding, padding:-padding]
    else:
      padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                  cv2.BORDER_REPLICATE, value=[0, 0, 0])

    # Resize the padded image to the target shape
    resized_image = cv2.resize(padded_image, original_shape[::-1])

    return resized_image

def blur(image, kernel_size=5):
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.GaussianBlur(kernel_size))
    image = np.array(image)
    return image


if __name__ == '__main__':
    # im_path = r"C:\Users\nplak\Downloads\2_7.7_0.44_1385071967.png"
    # mask_path = r"C:\Users\nplak\Desktop\dash_new\templates\resume_button\resume_button768_v1.png"
    #
    # im_path = r"C:\Users\nplak\Desktop\dash_new\templates\logo\9.0_0.6_2069471825.png"
    # mask_path = r"C:\Users\nplak\Desktop\dash_new\templates\logo\swipe_defence768_v3.png"

    im_path = r"C:\Users\nplak\Downloads\e.png"
    mask_path = r"C:\Users\nplak\Desktop\dash_new\templates\exit_button\exit_button768_v14.png"

    image = cv2.imread(im_path, flags=cv2.IMREAD_UNCHANGED)
    image = image[..., [2, 1, 0, 3]].copy() # convert bgra to rgba

    mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
    mask = pad_and_resize(mask, -96)

    processed_image = process(image, mask, -1, blur_kernel=31, shifts_str=0.2)
    print('!')
