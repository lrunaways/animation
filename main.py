import matplotlib.pyplot as plt

from animation.BaseAnimator import *
from animation.BaseImage import *

n_frames = 60
image_shape = 768, 768
fig_size = 64


if __name__ == '__main__':
    fig = RectangleAppear(n_frames, image_shape, fig_size)
    base_image = FractalImage(image_shape, fig_size)
    frame = fig.get_frame(3, base_image=base_image)
