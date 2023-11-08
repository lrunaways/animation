import torch

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    dot = torch.sum(v0 * v1 / (torch.linalg.norm(v0) * torch.linalg.norm(v1)))
    if torch.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2

def lerp(t, v0, v1):
  v = v0*(1-t) + v1*t
  return v


interpolation_funcs = {
    'lerp': lerp,
    'slerp': slerp,
}