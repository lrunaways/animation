import matplotlib.pyplot as plt

from animation.BaseAnimator import *
from animation.BaseImage import *

n_frames = 60
image_shape = 768, 768
fig_size = (24, 128)
TYPE = 'top'

if __name__ == '__main__':

    fig = TriangleGrow(n_frames, image_shape, fig_size, type=TYPE)
    base_image = FractalImage(image_shape, fig_size)
    frame = fig.get_frame(3, base_image=base_image)
    print(8)
