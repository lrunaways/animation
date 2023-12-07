import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
import tqdm

import tinify
tinify.key = "xQ8XQDdsjXbTdWjqZTWMTd33DpJLlVtY"

from diffuse_animation.helpers.postprocess_generation import process


FRAMES_PER_INTERP = 0
INTERPOLATE_TO_FIRST_FRAME = True

FRAMES_DIR = r"C:\Users\nplak\Desktop\dash_new\done_textures\new\play_button\1882965206_variations"
OUT_DIR = r"C:\Users\nplak\Desktop\dash_new\done_textures\assets\play_button"

# MASK_IMAGE = r"C:\Users\nplak\Desktop\dash_new\templates\exit_button\exit_button_768_v14.png"
MASK_IMAGE = r"C:\Users\nplak\Desktop\dash_new\templates\play_button\play_button_768_v3.png"


TINIFY = 'single'
# BLUR_KERNEL = 8
BLUR_KERNEL = 31
SHIFTS_STR = 0.2
PAD = -96

# RESIZE = [1224, 888]
RESIZE = [768, 768]
# RESIZE = None

def pad_and_resize(image, padding, border='reflect'):
    original_shape = image.shape[:-1]

    if border == 'reflect':
        border = cv2.BORDER_REPLICATE
    elif border == 'constant':
        border = cv2.BORDER_CONSTANT
    # Pad the image
    if padding < 0:
      padding = abs(padding)
      padded_image = image[padding:-padding, padding:-padding]
    else:
      padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                  border, value=[0, 0, 0, 0])

    # Resize the padded image to the target shape
    resized_image = cv2.resize(padded_image, original_shape[::-1])

    return resized_image

if __name__ == '__main__':
    assert TINIFY in ['single', 'atlas']

    if FRAMES_PER_INTERP > 0:
        FRAMES_PER_INTERP = FRAMES_PER_INTERP + 2
    else:
        FRAMES_PER_INTERP = 1

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    i_frame = 0
    filenames = os.listdir(FRAMES_DIR)
    try:
        filenames.sort(key=lambda x: int(x.split('_')[0]) * 1000 + int(x.split('_')[1][:-len('.png')]))
    except ValueError:
        print("RANDOM ORDER")
    if INTERPOLATE_TO_FIRST_FRAME:
        filenames.append(filenames[0])
    mask = cv2.imread(MASK_IMAGE, cv2.IMREAD_UNCHANGED)
    if PAD:
        mask = pad_and_resize(mask, PAD, border='reflect')

    all_images = []
    for i in tqdm.tqdm(range(len(filenames)-1)):
        im1 = cv2.imread(os.path.join(FRAMES_DIR, filenames[i]), cv2.IMREAD_UNCHANGED)
        im2 = cv2.imread(os.path.join(FRAMES_DIR, filenames[i+1]), cv2.IMREAD_UNCHANGED)

        im1 = pad_and_resize(im1, -PAD, border='constant')
        im2 = pad_and_resize(im2, -PAD, border='constant')

        processed_im1 = process(im1, mask, -1, blur_kernel=BLUR_KERNEL, shifts_str=SHIFTS_STR)
        processed_im2 = process(im2, mask, -1, blur_kernel=BLUR_KERNEL, shifts_str=SHIFTS_STR)
        for j in range(FRAMES_PER_INTERP):
            out_filename = str(i_frame) + '.png'
            alpha = j / FRAMES_PER_INTERP
            im = processed_im1*(1-alpha) + processed_im2*alpha
            im = im.astype(np.uint8)
            if RESIZE is not None:
                im_resized = cv2.resize(im, RESIZE, interpolation=cv2.INTER_AREA)
            else:
                im_resized = im
            out_img_path = os.path.join(OUT_DIR, out_filename)
            all_images.append(im_resized)
            if TINIFY == 'single':
                cv2.imwrite(out_img_path, im_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                source = tinify.from_file(out_img_path)
                source.to_file(out_img_path)
            i_frame += 1
    if TINIFY == 'atlas':
        concat_image = np.concatenate(all_images, axis=0)
        cv2.imwrite(out_img_path, concat_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        source = tinify.from_file(out_img_path)
        source.to_file(out_img_path)
    print('Success!')