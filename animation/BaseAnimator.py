import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

import helpers.perlin2d


class BaseAnimator:
    """
    Отрисовывает 2д бинарную маску формы для покадровой анимации

    """
    def __init__(self, n_frames, image_shape, fig_size, seed=28):
        self.n_frames = n_frames
        self.image_shape = image_shape
        self.center_coord = image_shape[0] // 2, image_shape[1] // 2
        self.fill_color = 1
        self.fig_size = fig_size

        self.seed = seed


    def generate_base_image(self, noise_muls = [4, 1.5, 4]):
        image = np.zeros(self.image_shape)

        size_framed = (2 * self.fig_size, 2 * self.fig_size)
        noise = helpers.perlin2d.generate_fractal_noise_2d(size_framed, (8, 8), octaves=4, lacunarity=2, persistence=0.2) * noise_muls[0]
        noise += helpers.perlin2d.generate_fractal_noise_2d(size_framed, (16, 16), octaves=4, lacunarity=2, persistence=0.2) * noise_muls[1]
        noise += helpers.perlin2d.generate_fractal_noise_2d(size_framed, (4, 4), octaves=3, lacunarity=2, persistence=0.2) * noise_muls[2]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        image[
            self.center_coord[0] - self.fig_size: self.center_coord[0] + self.fig_size,
            self.center_coord[1] - self.fig_size: self.center_coord[1] + self.fig_size,
        ] = noise

        self.noise_max = noise.max() + 1e-6
        self.noise_min = noise.min() - 1e-6

        return image

    def draw_func(self, **kwargs):
        pass

    def get_i_frame_params(self, i_frame):
        pass

    def get_frame(self, i_frame, blur=5, noise_muls=[4, 1.5, 4]):
        self.image = Image.fromarray(np.zeros(self.image_shape))
        self.draw_obj = ImageDraw.Draw(self.image)

        np.random.seed(self.seed)
        self.init_image = self.generate_base_image(noise_muls=noise_muls)

        draw_params = self.get_i_frame_params(i_frame)
        self.draw_func(**draw_params)
        frame = np.array(self.image)
        if blur:
            frame = cv2.GaussianBlur(frame, (blur, blur), 0)
            frame = frame / frame.max()

            self.init_image = cv2.GaussianBlur(self.init_image, (blur, blur), 0)
            self.init_image = self.init_image / self.init_image.max()

        return frame * self.init_image


class CircleGrow(BaseAnimator):
    def get_i_frame_params(self, i_frame):
        size_mul = (i_frame / self.n_frames)
        current_obj_size = self.fig_size*size_mul

        params = {
            'xy': [
                (self.center_coord[0]-current_obj_size, self.center_coord[1]-current_obj_size),
                (self.center_coord[0]+current_obj_size, self.center_coord[1]+current_obj_size)
            ],
            'fill': self.fill_color
        }
        return params

    def draw_func(self, **kwargs):
        self.draw_obj.ellipse(xy=kwargs['xy'], fill=kwargs['fill'])
        return 1


class RectangleGrow(BaseAnimator):
    def get_i_frame_params(self, i_frame):
        size_mul = (i_frame / self.n_frames)
        current_obj_size = self.fig_size * size_mul

        params = {
            'xy': [
                (self.center_coord[0]-current_obj_size, self.center_coord[1]-current_obj_size),
                (self.center_coord[0]+current_obj_size, self.center_coord[1]+current_obj_size)
            ],
            'fill': self.fill_color
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
                (self.center_coord[0]-current_obj_size, self.center_coord[1]-current_obj_size),
                (self.center_coord[0]+current_obj_size, self.center_coord[1]+current_obj_size)
            ],
            'fill': self.fill_color,
            'i_frame': i_frame
        }
        return params

    def draw_func(self, **kwargs):
        self.draw_obj.rectangle(xy=kwargs['xy'], fill=kwargs['fill'])

        thr_step = (self.noise_max - self.noise_min) / self.n_frames
        thr = self.noise_max - thr_step * kwargs['i_frame']

        self.init_image[self.init_image < thr] = 0
        return 1

