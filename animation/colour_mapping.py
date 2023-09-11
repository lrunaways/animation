import numpy as np

def colormap(image,
             a=[0.5, 0.5, 0.5],
             b=[0.5, 0.5, 0.5],
             c=[1.0, 1.0, 1.0],
             d=[0.3, 0.2, 0.1]):
    a = np.array(a).reshape((1, 1, -1))
    b = np.array(b).reshape((1, 1, -1))
    c = np.array(c).reshape((1, 1, -1))
    d = np.array(d).reshape((1, 1, -1))
    return a + b*np.cos(6.28318 * (c * image + d))
