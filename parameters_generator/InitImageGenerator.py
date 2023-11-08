import PIL
import numpy as np

from ..animation.BaseAnimator import *
from ..animation.BaseImage import *


class InitImageGenerator:
  def __init__(self, background_image, fig_size, mask_generators, n_frames, image_alpha, blur=9, always_full=False):
    assert isinstance(mask_generators, list)
    self.mask_generators = mask_generators
    self.n_frames = n_frames
    self.background_image = np.array(background_image).copy()
    self.image_shape = self.background_image.shape[:-1]
    self.image_alpha = image_alpha
    self.blur = blur
    self.always_full = always_full
    self.noise_image = FractalImage(image_shape=self.image_shape, fig_size=fig_size)

    if self.always_full:
      self.init_image = self.__getitem__(n_frames)

  def __getitem__(self, idx):
    if self.always_full:
      return self.init_image

    if idx < 0 or idx > self.n_frames - 1:
      raise IndexError
    animation_mask = np.zeros(self.image_shape)
    for mask_generator in self.mask_generators:
      animation_mask += mask_generator.get_frame(idx, blur=self.blur, base_image=self.noise_image)
    animation_mask /= animation_mask.max()
    animation_mask = animation_mask[..., None]

    init_image_masked = self.background_image*animation_mask
    init_image = (init_image_masked*self.image_alpha).astype(np.uint8)
    init_image = PIL.Image.fromarray(init_image)

    return init_image