import cv2
import numpy as np

from matplotlib import colors

BASE_COLORS = list(colors.BASE_COLORS.keys())

def rgb2gray(rgb):
    return np.dot(rgb[...:3], [0.2989, 0.5870, 0.1140])

def _normalize_color(color, image = None):
    if colors.is_color_like(color):
        color = colors.to_rgb(color)
    
    color = np.array(color)
    if np.max(color) > 1.: color = color / 255.
    if image is not None and np.max(image) > 1.: color = (color * 255).astype(image.dtype)
    
    return color
