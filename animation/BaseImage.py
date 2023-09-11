import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

from animation import perlin2d
from animation.colour_mapping import colormap


class BaseImage:
    def __init__(self, image_shape, fig_size):
        self.image_shape = np.array(image_shape)
        self.fig_size = fig_size
        self.center_coord = self.image_shape // 2


class PerlinImage(BaseImage):
    def generate_base_image(self, noise_muls=[4, 1.5, 4]):
        image = np.zeros(self.image_shape)

        size_framed = (2 * self.fig_size, 2 * self.fig_size)
        noise = perlin2d.generate_fractal_noise_2d(size_framed, (8, 8), octaves=4, lacunarity=2, persistence=0.2) * \
                noise_muls[0]
        noise += perlin2d.generate_fractal_noise_2d(size_framed, (16, 16), octaves=4, lacunarity=2,
                                                    persistence=0.2) * noise_muls[1]
        noise += perlin2d.generate_fractal_noise_2d(size_framed, (4, 4), octaves=3, lacunarity=2, persistence=0.2) * \
                 noise_muls[2]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        image[
        self.center_coord[0] - self.fig_size: self.center_coord[0] + self.fig_size,
        self.center_coord[1] - self.fig_size: self.center_coord[1] + self.fig_size,
        ] = noise

        self.noise_max = noise.max() + 1e-6
        self.noise_min = noise.min() - 1e-6

        return image

class FractalImage(BaseImage):
    def generate_base_image(self, iTime=10, iterations=32):
        image = np.zeros(self.image_shape)
        self.size_framed = np.array([2 * self.fig_size, 2 * self.fig_size])

        uv_x = np.ones(self.size_framed) * np.arange(self.size_framed[1])[None, :] / self.size_framed[1]
        uv_y = np.ones(self.size_framed) * np.arange(self.size_framed[0])[:, None] / self.size_framed[0]
        uv = np.stack([uv_x, uv_y])

        uv = 2. * uv - 1.

        uvs = uv * self.size_framed[::-1].reshape((-1, 1, 1)) / max(self.size_framed)
        uvs = np.concatenate([uvs, np.zeros(self.size_framed)[None]], axis=0)

        p = uvs / 6.
        # p = uvs / 4.
        p -= np.reshape(np.array([1, -1.3, 0]), (3, 1, 1))
        p -= .2 * np.sin(np.reshape(np.array([
            iTime / 16.,
            iTime / 12.,
            iTime / 128.
        ]), (3, 1, 1)))

        t = self.field(p, iTime, iterations=iterations)

        vingette = (1. - np.exp((abs(uv[0]) - 1.) * 6.)) * (1. - np.exp((abs(uv[1]) - 1.) * 6.))
        vingette = .4 * (1 - vingette) + 1. * vingette
        vingette = vingette[..., None]

        out = np.stack([
                # 1.8 * t * t * t,
                # 1.4 * t * t,
                t
        ], axis=-1)
        out_vingetted = out * vingette

        self.noise_max = out_vingetted.max() + 1e-6
        self.noise_min = out_vingetted.min() - 1e-6

        image[
        self.center_coord[0] - self.fig_size: self.center_coord[0] + self.fig_size,
        self.center_coord[1] - self.fig_size: self.center_coord[1] + self.fig_size,
        ] = out_vingetted[..., 0]
        return image

    def field(self, p, iTime, iterations=32):
        strength = 7. + .03 * np.log(1.e-6 + np.abs(np.modf(np.sin(iTime) * 4373.11))[0])
        accum = 0.
        prev = 0.
        tw = 0.
        for i in range(iterations):
            mag = (p[0]**2 + p[1]**2 + p[2]**2)
            p = np.abs(p) / mag
            p += np.reshape(np.array([-0.5, -0.4, -1.5]), (3, 1, 1))

            w = np.exp(-i/7.)
            diff = np.abs(mag - prev)
            accum += w * np.exp(-strength * np.power(diff, 2.3))
            tw += w
            prev = mag
        out = np.clip(5. * accum / tw - .7, 0, 999)
        return out


if __name__=="__main__":
    im_obj = FractalImage(image_shape=(768, 512), fig_size=64)
    # im_obj = FractalImage(image_shape=(450, 800), fig_size=64)
    base_image = im_obj.generate_base_image()
