from parameters_generator import GenerationParameters, DiffusionParameters, InitImageGenerator
from animation.BaseAnimator import *
from animation.BaseImage import *

# import sys
# sys.path.append('.')

if __name__=="__main__":
    N_FRAMES = 60
    FIG_SIZE = (128, int(128 * 0.8159258436))
    # ------------------------------------------------------------------------------
    GUIDANCES = [1, 10, 100]
    STRS = [0.5, 0.6, 1.0]
    SEEDS = [123, 321, 256]
    # -------------------------------------------------------------------------------

    IM_SHAPE = [768, 768]
    init_image = np.random.random(IM_SHAPE + [3])
    init_image *= 255
    init_image = init_image.astype(np.uint8)

    prompt = 'test'
    negative_prompt = 'neg_test'

    mask_generators = [
        TriangleGrow(n_frames=N_FRAMES, image_shape=IM_SHAPE, fig_size=FIG_SIZE, type="top"),
        RectangleGrow(n_frames=N_FRAMES, image_shape=IM_SHAPE, fig_size=FIG_SIZE, type="central"),
        TriangleGrow(n_frames=N_FRAMES, image_shape=IM_SHAPE, fig_size=FIG_SIZE, type='bottom')
    ]

    init_image_generator = InitImageGenerator(
        background_image=init_image,
        fig_size=FIG_SIZE,
        mask_generators=mask_generators,
        n_frames=N_FRAMES,
        image_alpha=1.0
    )
    diffusion_parameters = DiffusionParameters(
        N_FRAMES=N_FRAMES,
        PROMPTS=[prompt, prompt, prompt],
        NEGATIVE_PROMPT=negative_prompt,
        GUIDANCES=GUIDANCES,
        STRS=STRS,
        SEEDS=SEEDS,
        VARIATION_NOISES=False,
        INTERPOLATION='lerp',
        CYCLED=True
    )
    params = GenerationParameters(init_image,
                                  n_frames=N_FRAMES,
                                  init_image_generator=init_image_generator,
                                  diffusion_parameters=diffusion_parameters,
                                  resize_factor=1)
    params_i = params[0]
    print("Success!")
