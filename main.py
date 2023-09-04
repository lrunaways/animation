import matplotlib.pyplot as plt

from animation.BaseAnimator import Circle

n_frames = 60
image_shape = 768, 768
fig_size = 64


if __name__ == '__main__':
    fig = Circle(n_frames, image_shape, fig_size)

    frame = fig.get_frame(3)
