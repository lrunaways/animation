import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

from . import perlin2d
from . import BaseImage


class BaseAnimator:
    """
    Отрисовывает 2д бинарную маску формы для покадровой анимации

    """
    def __init__(self, n_frames, image_shape, fig_size, seed=28, base_image=None, type='central'):
        self.n_frames = n_frames
        self.image_shape = image_shape
        self.center_coord = image_shape[0] // 2, image_shape[1] // 2
        self.fill_color = 1
        self.fig_size = np.array(fig_size)
        self.base_image = base_image
        self.type = type
        self.seed = seed

    def draw_func(self, **kwargs):
        pass

    def get_i_frame_params(self, i_frame):
        pass

    def get_frame(self, i_frame, base_image, blur=5):
        self.base_image = base_image.generate_base_image()
        self.noise_max = base_image.noise_max
        self.noise_min = base_image.noise_min
        self.image = Image.fromarray(np.zeros(self.image_shape))
        self.draw_obj = ImageDraw.Draw(self.image)

        draw_params = self.get_i_frame_params(i_frame)
        self.draw_func(**draw_params)
        frame = np.array(self.image)
        if blur:
            frame = cv2.GaussianBlur(frame, (blur, blur), 0)
            frame = frame / frame.max()

            self.base_image = cv2.GaussianBlur(self.base_image, (blur, blur), 0)
            self.base_image = self.base_image / self.base_image.max()

        return frame * self.base_image


class CircleGrow(BaseAnimator):
    def get_i_frame_params(self, i_frame):
        size_mul = (i_frame / self.n_frames)
        current_obj_size = self.fig_size*size_mul

        params = {
            'xy': [
                (self.center_coord[0]-current_obj_size[1], self.center_coord[1]-current_obj_size[1]),
                (self.center_coord[0]+current_obj_size[0], self.center_coord[1]+current_obj_size[0])
            ],
            'fill': self.fill_color
        }
        return params

    def draw_func(self, **kwargs):
        self.draw_obj.ellipse(xy=kwargs['xy'], fill=kwargs['fill'])
        return 1


class RectangleGrow(BaseAnimator):
    def get_i_frame_params(self, i_frame):
        assert self.type in ['central', 'bottom']
        size_mul = (i_frame / self.n_frames)
        current_obj_size = self.fig_size * size_mul

        if self.type == 'central':
            params = {
                'xy': [
                    (self.center_coord[0]-current_obj_size[1], self.center_coord[1]-current_obj_size[1]),
                    (self.center_coord[0]+current_obj_size[0], self.center_coord[1]+current_obj_size[0])
                ],
                'fill': self.fill_color,
            }
        elif self.type == 'bottom':
            params = {
                'xy': [
                    (self.center_coord[0]-self.fig_size[1], self.center_coord[1]-current_obj_size[1]),
                    (self.center_coord[0]+self.fig_size[0], self.center_coord[1]+current_obj_size[0])
                ],
                'fill': self.fill_color,
            }

        return params

    def draw_func(self, **kwargs):
        self.draw_obj.rectangle(xy=kwargs['xy'], fill=kwargs['fill'])
        return 1

class RectangleAppear(BaseAnimator):
    def get_i_frame_params(self, i_frame):
        current_obj_size = self.fig_size

        params = {
            'xy': [
                (self.center_coord[0]-current_obj_size[1], self.center_coord[1]-current_obj_size[1]),
                (self.center_coord[0]+current_obj_size[0], self.center_coord[1]+current_obj_size[0])
            ],
            'fill': self.fill_color,
            'i_frame': i_frame
        }
        return params

    def draw_func(self, **kwargs):
        self.draw_obj.rectangle(xy=kwargs['xy'], fill=kwargs['fill'])

        thr_step = (self.noise_max - self.noise_min) / self.n_frames
        thr = self.noise_max - thr_step * kwargs['i_frame']

        self.base_image[self.base_image < thr] = 0
        return 1

class TriangleGrow(RectangleGrow):
    def get_i_frame_params(self, i_frame):
        assert self.type in ['bottom', 'top']
        size_mul = (i_frame / self.n_frames)
        current_obj_size = self.fig_size * size_mul

        # if self.type == 'central':
        #     params = {
        #         'xy': [
        #             (self.center_coord[0]-current_obj_size[1], self.center_coord[1]-current_obj_size[1]),
        #             (self.center_coord[0]+current_obj_size[0], self.center_coord[1]+current_obj_size[0])
        #         ],
        #         'fill': self.fill_color,
        #     }
        if self.type == 'top':
            tip_point = self.center_coord[1] + current_obj_size[1]
            params = {
                'xy': [
                    (self.center_coord[0]-self.fig_size[0], self.center_coord[1]),
                    (self.center_coord[0]+self.fig_size[0], self.center_coord[1]),
                    (self.center_coord[0], tip_point)
                ],
                'fill': self.fill_color,
            }
        elif self.type == 'bottom':
            tip_point = self.center_coord[1] - current_obj_size[1]
            params = {
                'xy': [
                    (self.center_coord[0]-self.fig_size[0], self.center_coord[1]),
                    (self.center_coord[0]+self.fig_size[0], self.center_coord[1]),
                    (self.center_coord[0], tip_point)
                ],
                'fill': self.fill_color,
            }

        return params

    def draw_func(self, **kwargs):
        self.draw_obj.polygon(xy=kwargs['xy'], fill=kwargs['fill'])
        return 1

class FullImage(BaseAnimator):
    def __init__(self, image_shape, base_image=None):
        if base_image is not None:
            self.base_image = np.ones_like(base_image)
        else:
            self.base_image = np.ones(image_shape)

    def get_frame(self, i_frame, base_image, blur=None):
        return self.base_image

class FromMask(BaseAnimator):
    def __init__(self, mask_image):
        self.mask_image = mask_image if len(mask_image) == 2 else mask_image[..., 0]

    def get_frame(self, i_frame, base_image, blur=None):
        return self.mask_image