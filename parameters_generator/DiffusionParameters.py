import torch

from diffuse_animation.helpers import interpolation_funcs


class DiffusionParameters:
  def __init__(
    self,
    N_FRAMES,
    PROMPTS,
    NEGATIVE_PROMPT,
    GUIDANCES,
    STRS,
    SEEDS,
    VARIATION_NOISES,
    INTERPOLATION,
    CYCLED
    ):
    assert INTERPOLATION in interpolation_funcs.keys()
    assert len(GUIDANCES) == len(STRS) == len(SEEDS)
    if isinstance(VARIATION_NOISES, list):
      assert len(VARIATION_NOISES) == len(SEEDS)

    assert isinstance(PROMPTS, list)
    if len(PROMPTS) == 1:
      PROMPTS = PROMPTS*len(SEEDS)
    assert len(PROMPTS) == len(SEEDS)

    self.n_frames = N_FRAMES
    self.prompts = PROMPTS
    self.negative_prompt = NEGATIVE_PROMPT
    self.guidances = GUIDANCES
    self.strengths = STRS
    self.seeds = SEEDS
    self.variation_noises = VARIATION_NOISES
    self.interpolation = INTERPOLATION
    self.interpolation_func = interpolation_funcs[self.interpolation]
    self.cycled = CYCLED

    if self.cycled:
      self.prompts += [self.prompts[0]]
      self.guidances += [self.guidances[0]]
      self.strengths += [self.strengths[0]]
      self.seeds += [self.seeds[0]]
    if self.cycled and isinstance(self.variation_noises, list):
      self.variation_noises += [self.variation_noises[0]]


  def __getitem__(self, idx):
    if idx < 0 or idx > self.n_frames*len(self.seeds)-1:
      raise IndexError
    i_frame = idx % self.n_frames

    idx_global = idx//self.n_frames
    cur_guidance = self.guidances[idx_global]
    next_guidance = self.guidances[idx_global+1]
    cur_seed = self.seeds[idx_global]
    next_seed = self.seeds[idx_global+1]
    cur_strength = self.strengths[idx_global]
    next_strength = self.strengths[idx_global+1]
    cur_prompt = self.prompts[idx_global]
    next_prompt = self.prompts[idx_global+1]

    #----------------------------
    diffusion_parameters = {}
    diffusion_parameters['negative_prompt'] = self.negative_prompt

    if self.variation_noises:
      cur_var_noise = self.variation_noises[idx_global]
      next_var_noise = self.variation_noises[idx_global + 1]
      diffusion_parameters['variation_noise'] = [cur_var_noise, next_var_noise]
    else:
      diffusion_parameters['variation_noise'] = False

    #TODO: different interp funcs
    seed_alpha = (i_frame/self.n_frames)**2.
    diffusion_parameters['seed_alpha'] = seed_alpha
    # seed_alpha_linear = (i_frame/N_FRAMES)
    diffusion_parameters['prompt_alpha'] = (i_frame/self.n_frames)**2.
    diffusion_parameters['interpolation'] = self.interpolation

    diffusion_parameters['guidance'] = interpolation_funcs['lerp'](seed_alpha, cur_guidance, next_guidance)
    diffusion_parameters['strength'] = interpolation_funcs['lerp'](seed_alpha, cur_strength, next_strength)

    # diffusion_parameters['generator_1'] = torch.Generator(device='cuda').manual_seed(cur_seed)
    # diffusion_parameters['generator_2'] = torch.Generator(device='cuda').manual_seed(next_seed)
    diffusion_parameters['seed_1'] = cur_seed
    diffusion_parameters['seed_2'] = next_seed
    diffusion_parameters['generator_1'] = torch.Generator(device='cpu').manual_seed(cur_seed)
    diffusion_parameters['generator_2'] = torch.Generator(device='cpu').manual_seed(next_seed)

    diffusion_parameters['prompt_1'] = cur_prompt
    diffusion_parameters['prompt_2'] = next_prompt
    return diffusion_parameters
