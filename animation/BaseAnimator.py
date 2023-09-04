import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2


class BaseAnimator:
    """
    Отрисовывает 2д бинарную маску формы для покадровой анимации

    """
    def __init__(self, n_frames, image_shape, fig_size):
        self.n_frames = n_frames
        self.image_shape = image_shape
        self.center_coord = image_shape[0] // 2, image_shape[1] // 2
        self.fill_color = 1
        self.fig_size = fig_size


    def draw_func(self, **kwargs):
        pass

    def get_i_frame_params(self, i_frame):
        pass

    def get_frame(self, i_frame, blur=5):
        self.image = Image.fromarray(np.zeros(self.image_shape))
        self.draw_obj = ImageDraw.Draw(self.image)

        draw_params = self.get_i_frame_params(i_frame)
        self.draw_func(**draw_params)
        frame = np.array(self.image)
        if blur:
            # kernel = np.ones((blur, blur), np.float32) / (blur**2)
            frame = cv2.GaussianBlur(frame, (blur, blur), 0)
            frame = frame / frame.max()
        return frame


class Circle(BaseAnimator):
    # def __init__(self, n_frames, image_shape, fig_size):
    #     super().__init__(
    #         n_frames=n_frames,
    #         image_shape=image_shape,
    #         fig_size=fig_size
    #     )

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
# ImageDraw.ellipse(xy, fill=None, outline=None, width=1)
