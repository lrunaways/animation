import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
import tqdm

FRAMES_DIR = r"C:\Users\nplak\Downloads\cell_bonus_disappear"
FRAMES_PER_INTERP = 10
INTERPOLATE_TO_FIRST_FRAME = False

OUT_DIR = r"C:\Users\nplak\Desktop\dash2\cell_bonus_disappear"

if __name__ == '__main__':
    FRAMES_PER_INTERP = FRAMES_PER_INTERP + 2

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    i_frame = 0
    filenames = os.listdir(FRAMES_DIR)
    filenames.sort(key=lambda x: int(x.split('_')[0]) * 1000 + int(x.split('_')[1][:-len('.png')]))
    if INTERPOLATE_TO_FIRST_FRAME:
        filenames.append(filenames[0])
    for i in tqdm.tqdm(range(len(filenames)-1)):
        im1 = cv2.imread(os.path.join(FRAMES_DIR, filenames[i]), cv2.IMREAD_UNCHANGED)
        im2 = cv2.imread(os.path.join(FRAMES_DIR, filenames[i+1]), cv2.IMREAD_UNCHANGED)
        for j in range(FRAMES_PER_INTERP):
            out_filename = str(i_frame) + '.png'
            alpha = j / FRAMES_PER_INTERP
            im = im1*(1-alpha) + im2*alpha
            im = im.astype(np.uint8)
            cv2.imwrite(os.path.join(OUT_DIR, out_filename), im)
            i_frame += 1
    print('Success!')