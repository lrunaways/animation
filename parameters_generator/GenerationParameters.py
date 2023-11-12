
class GenerationParameters:
  def __init__(self, init_image, n_frames, init_image_generator, diffusion_parameters, resize_factor=1):
    self.init_image = init_image
    self.n_frames = n_frames
    self.init_image_generator = init_image_generator
    self.diffusion_parameters = diffusion_parameters
    self.resize_factor = resize_factor
    # self.do_multiply_by_mask = do_multiply_by_mask

  # def multiply_by_mask(image, mask):
  #   image = image*mask
  #   return image

  def __getitem__(self, idx):
    params = {}
    params.update(self.diffusion_parameters[idx])
    params['image'] = self.init_image_generator[idx]
    params['image_name'] = f"{idx}_test" + '.png'
    if self.resize_factor != 1:
      params['image'] = params['image'].resize(
        (
          int(params['image'].size[0]*self.resize_factor),
          int(params['image'].size[1]*self.resize_factor),
        )
      )
    return params