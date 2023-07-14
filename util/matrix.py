import numpy as np

def rot180(x: np.ndarray) -> np.ndarray:
    return np.rot90(x, k=2, axes=(x.ndim - 2, x.ndim - 1))

def im2col(x: np.ndarray, filter_height: int, filter_width: int, stride: int, padding: int) -> np.ndarray: 
    n, c, h, w = x.shape

    o_h = (h - filter_height + 2 * padding) // stride + 1
    o_w = (w - filter_width + 2 * padding) // stride + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    out = np.zeros((n, c, filter_height, filter_width, o_h, o_w))

    for h in range(filter_height): 
        h_end = h + stride * o_h
        for w in range(filter_width): 
            w_end = w + stride * o_w
            out[:, :, h, w, :, :] = x_pad[:, :, h:h_end:stride, w:w_end:stride]
    out = out.transpose(0, 4, 5, 1, 2, 3).reshape(n * o_h * o_w, -1)
    return out

def cross_correlate(x: np.ndarray, filter: np.ndarray, stride: int, padding: int) -> np.ndarray: 
    if x.ndim != 4 or filter.ndim != 4:
        raise ValueError("x and filter must be 4-dimensional")
    
    n, c, h, w = x.shape
    out_ch, _, f_h, f_w = filter.shape
    o_h = (h - f_h + 2 * padding) // stride + 1
    o_w = (w - f_w + 2 * padding) // stride + 1

    cols = im2col(x, f_h, f_w, stride, padding)
    out = cols @ (filter.reshape(out_ch, -1).T)
    return out.reshape(n, o_h, o_w, out_ch).transpose(0, 3, 1, 2)


def convolve(x: np.ndarray, filter: np.ndarray, stride: int, padding: int) -> np.ndarray: 
    return cross_correlate(x, np.flip(filter), stride, padding)
