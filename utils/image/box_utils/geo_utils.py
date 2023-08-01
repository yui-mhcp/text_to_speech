# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import logging
import numpy as np
import tensorflow as tf

from utils.image.box_utils.nms import nms
from utils.image.box_utils.box_functions import BoxFormat, convert_box_format

logger = logging.getLogger(__name__)

"""
General functionalities for 2D-geometry on points, lines and quad. 
In all the functions points are identified by their `(y, x)` coordinates where the `y` corresponds to the vertical axis and `x` to the horizontal axis

Most of the time, the below functions handle numpy parallelization for multiple points / lines / polys
This require to use masking to handle conditions
"""

""" General point / line geometry functions """

def fit_line(p1, p2):
    """
        Arguments :
            - l1, l2 : points, np.ndarray of shape [n_points (optional), 2] where last axis is (y, x)
        Returns : the equation of the line traversing p1 and p2 : ax + by + c = 0
            - np.ndarray with last-axis of length 3 : [a, b, c] coefficients
                if multiple points are given, array of shape [n_points, 3], [3] otherwise
        
        In general, the equation of a line is :
            a = delta_y / delta_x, b = -1, c = x_1 - (a * y_1)
        However, if delta_x == 0 (i.e. the two points are on the same horizontal line) :
            a = 1, b = 0, c = - x_1
    """
    is_multi = len(p1.shape) == 2
    if not is_multi: p1, p2 = np.expand_dims(p1, 0), np.expand_dims(p2, 0)
    lines = np.zeros((p1.shape[0], 3))
    
    mask = p1[..., 0] == p2[..., 0]
    if np.any(mask):
        lines[mask, 0] = 1.
        lines[mask, 2] = - p1[mask, 0]
    
    not_mask = ~mask
    if np.any(not_mask):
        sub_p1, sub_p2 = p1[not_mask], p2[not_mask]
        a = (sub_p2[..., 1] - sub_p1[..., 1]) / (sub_p2[..., 0] - sub_p1[..., 0])
        c = sub_p1[..., 1] - (a * sub_p1[..., 0])
        
        lines[not_mask, 0] = a
        lines[not_mask, 1] = -1.
        lines[not_mask, 2] = c
        
    return lines if is_multi else lines[0]

def point_dist_to_line(p1, p2, p3):
    """
        Computes the distance from p3 to (p1 - p2)
        Arguments :
            - p1, p2, p3 : the 3 points of shape [n_points (optional), 2] of (y, x) coordinates
        Returns : the distance between p3 and (p1 - p2):
            - scalar : if points are vectors of shape (2, )
            - vector of shape (n_points, ) : otherwise
    """
    cross = np.cross(p2 - p1, p1 - p3)
    if len(p1.shape) == 2: cross = np.expand_dims(cross, axis = 1)
    return np.linalg.norm(cross, axis = -1) / np.linalg.norm(p2 - p1, axis = -1)

def line_cross_point(lines1, lines2):
    """
        Computes the cross point of 2 lines
        Arguments :
            - lines1, lines2 : the two lines, np.ndarray of shape [n_lines (optional), 3]
                The last-axis represents the coefficiants [a, b, c] of the line equation `ax + by + c = 0`
        Returns : the cross-point coordinates (y, x), the intersection of the 2 lines
            - points : np.ndarray of shape [n_lines, 2] (if n_lines is given), [2] otherwise
    """
    is_multi = len(lines1.shape) == 2
    if not is_multi: lines1, lines2 = np.expand_dims(lines1, 0), np.expand_dims(lines2, 0)
    
    points = np.zeros((lines1.shape[0], 2))
    
    mask_0      = lines1[..., 1] == 0
    mask_1      = lines2[..., 1] == 0
    mask_n01    = np.logical_and(~mask_0, mask_1)
    if np.any(mask_0):
        x = -lines1[mask_0, 2]
        points[mask_0, 0] = x
        points[mask_0, 1] = lines2[mask_0, 0] * x + lines2[mask_0, 2]
    
    if np.any(mask_n01):
        x = -lines2[mask_n01, 2]
        points[mask_n01, 0] = x
        points[mask_n01, 1] = lines1[mask_n01, 0] * x + lines1[mask_n01, 2]

    mask_n0n1 = np.logical_and(~mask_0, ~mask_1)
    if np.any(mask_n0n1):
        k1, b1 = lines1[mask_n0n1, 0], lines1[mask_n0n1, 2]
        k2, b2 = lines2[mask_n0n1, 0], lines2[mask_n0n1, 2]
        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            x = np.nan_to_num(- (b1 - b2) / (k1 - k2), copy = False)
        points[mask_n0n1, 0] = x
        points[mask_n0n1, 1] = k1 * x + b1
    
    mask_inf = np.logical_and(lines1[..., 0] == 0, lines2[..., 0] == 0)
    if np.any(mask_inf):
        points[mask_inf] = np.inf

    return points if is_multi else points[0]

def line_verticle(lines, points):
    """
        Computes the equation of the vertical line of `lines` going though `points`
        Arguments :
            - lines : the lines, np.ndarray of shape [n (optional), 3]
                The last-axis represents [a, b, c] coefficiants of the line equation `ax + by + c = 0`
            - points : the points, np.ndarray of shape [n (optional), 2], (y, x) coordinates
        Returns : the equation of the vertical lines
            - verticles : np.ndarray of shape [n, 3] (if n is provided), [3] otherwise
    """
    is_multi = len(lines.shape) == 2
    if not is_multi: lines, points = np.expand_dims(lines, 0), np.expand_dims(points, 0)
    
    verticles = np.zeros(lines.shape)
    
    mask = lines[..., 1] == 0
    not_mask = ~mask
    if np.any(mask):
        verticles[mask, 1] = -1
        verticles[mask, 2] = points[mask, 1]
    
    if np.any(not_mask):
        mask_1 = lines[..., 0] == 0
        
        mask_n01    = np.logical_and(not_mask, mask_1)
        mask_n0n1   = np.logical_and(not_mask, ~mask_1)
        
        if np.any(mask_n01):
            verticles[mask_n01, 0] = 1
            verticles[mask_n01, 2] = - points[mask_n01, 0]
        
        if np.any(mask_n0n1):
            verticles[mask_n0n1, 0] = -1. / lines[mask_n0n1, 0]
            verticles[mask_n0n1, 1] = - 1.
            verticles[mask_n0n1, 2] = points[mask_n0n1, 1] - (-1. / lines[mask_n0n1, 0] * points[mask_n0n1, 0])
    
    return verticles if is_multi else verticles[0]


""" General functions on quad polygons """
def polygon_area(poly):
    """
        Computes the area of the polygons of shape [..., 4, 2]
        Depending on the arrangement of points, the area may be negative !
        If the area is negative, rearrange the points `polys[(0, 3, 2, 1)]`
    """
    rolled = np.roll(poly, shift = 1, axis = -2)

    diff = (poly[..., 0] - rolled[..., 0]) * (poly[..., 1] + rolled[..., 1])
    return np.sum(diff, axis = -1) / 2

def check_and_validate_polys(polys, tags, image_shape, min_area = 1):
    """
        Filters valid polygons
        Arguments :
            - polys : np.ndarray of shape [..., 4, 2] where last axis are (y, x)* coordinates
            - tags  : the labels for the polygons
            - image_shape   : tuple (height, width) of the image shape
            - min_area      : the minimal area for a valid polygon
        Returns :
            - valid_polys   : np.ndarray with shape [n_valids, 4, 2]
            - valid_tags    : the tags for the associated valid polys
        
        * the `y` coordinate represents the horizontal axis
    """
    if len(polys) == 0: return polys, tags

    polys[..., 0] = np.clip(polys[..., 0], 0, image_shape[0] - 1)
    polys[..., 1] = np.clip(polys[..., 1], 0, image_shape[1] - 1)
    # Filters valid masks
    areas       = polygon_area(polys)
    valid_mask  = np.abs(areas) >= min_area
    polys, tags = polys[valid_mask], tags[valid_mask]
    # Rearrange points if needed
    rearrange_mask  = areas[valid_mask] > 0 
    
    arranged    = polys[:, (0, 3, 2, 1), :]
    polys[rearrange_mask] = arranged[rearrange_mask]
    
    return polys, tags

def shrink_poly(polys, r, ratio = 0.3):
    """
        Shrinks the polygons' lines according to a given ratio
    """
    shrinked = polys.copy().astype(np.float32)
    
    mask =  np.linalg.norm(
        polys[:, 0] - polys[:, 1], axis = -1) + np.linalg.norm(
        polys[:, 2] - polys[:, 3], axis = -1) > np.linalg.norm(
        polys[:, 0] - polys[:, 3], axis = -1) + np.linalg.norm(
        polys[:, 1] - polys[:, 2], axis = -1
    )
    not_mask = ~mask
    
    if np.any(mask):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        sub_polys, sub_r = shrinked[mask], r[mask]
        
        theta = np.arctan2(
            sub_polys[:, 1, 0] - sub_polys[:, 0, 0],
            sub_polys[:, 1, 1] - sub_polys[:, 0, 1]
        )
        sub_polys[:, 0, 1] += ratio * sub_r[:, 0] * np.cos(theta)
        sub_polys[:, 0, 0] += ratio * sub_r[:, 0] * np.sin(theta)
        sub_polys[:, 1, 1] -= ratio * sub_r[:, 1] * np.cos(theta)
        sub_polys[:, 1, 0] -= ratio * sub_r[:, 1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2(
            sub_polys[:, 2, 0] - sub_polys[:, 3, 0],
            sub_polys[:, 2, 1] - sub_polys[:, 3, 1]
        )
        sub_polys[:, 3, 1] += ratio * sub_r[:, 3] * np.cos(theta)
        sub_polys[:, 3, 0] += ratio * sub_r[:, 3] * np.sin(theta)
        sub_polys[:, 2, 1] -= ratio * sub_r[:, 2] * np.cos(theta)
        sub_polys[:, 2, 0] -= ratio * sub_r[:, 2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2(
            sub_polys[:, 3, 1] - sub_polys[:, 0, 1],
            sub_polys[:, 3, 0] - sub_polys[:, 0, 0]
        )
        sub_polys[:, 0, 1] += ratio * sub_r[:, 0] * np.sin(theta)
        sub_polys[:, 0, 0] += ratio * sub_r[:, 0] * np.cos(theta)
        sub_polys[:, 3, 1] -= ratio * sub_r[:, 3] * np.sin(theta)
        sub_polys[:, 3, 0] -= ratio * sub_r[:, 3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2(
            sub_polys[:, 2, 1] - sub_polys[:, 1, 1],
            sub_polys[:, 2, 0] - sub_polys[:, 1, 0]
        )
        sub_polys[:, 1, 1] += ratio * sub_r[:, 1] * np.sin(theta)
        sub_polys[:, 1, 0] += ratio * sub_r[:, 1] * np.cos(theta)
        sub_polys[:, 2, 1] -= ratio * sub_r[:, 2] * np.sin(theta)
        sub_polys[:, 2, 0] -= ratio * sub_r[:, 2] * np.cos(theta)
        
        shrinked[mask] = sub_polys
    
    if np.any(not_mask):
        sub_polys, sub_r = shrinked[not_mask], r[not_mask]
        ## p0, p3
        theta = np.arctan2(
            sub_polys[:, 3, 1] - sub_polys[:, 0, 1],
            sub_polys[:, 3, 0] - sub_polys[:, 0, 0]
        )
        sub_polys[:, 0, 1] += ratio * sub_r[:, 0] * np.sin(theta)
        sub_polys[:, 0, 0] += ratio * sub_r[:, 0] * np.cos(theta)
        sub_polys[:, 3, 1] -= ratio * sub_r[:, 3] * np.sin(theta)
        sub_polys[:, 3, 0] -= ratio * sub_r[:, 3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2(
            sub_polys[:, 2, 1] - sub_polys[:, 1, 1],
            sub_polys[:, 2, 0] - sub_polys[:, 1, 0]
        )
        sub_polys[:, 1, 1] += ratio * sub_r[:, 1] * np.sin(theta)
        sub_polys[:, 1, 0] += ratio * sub_r[:, 1] * np.cos(theta)
        sub_polys[:, 2, 1] -= ratio * sub_r[:, 2] * np.sin(theta)
        sub_polys[:, 2, 0] -= ratio * sub_r[:, 2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2(
            sub_polys[:, 1, 0] - sub_polys[:, 0, 0],
            sub_polys[:, 1, 1] - sub_polys[:, 0, 1]
        )
        sub_polys[:, 0, 1] += ratio * sub_r[:, 0] * np.cos(theta)
        sub_polys[:, 0, 0] += ratio * sub_r[:, 0] * np.sin(theta)
        sub_polys[:, 1, 1] -= ratio * sub_r[:, 1] * np.cos(theta)
        sub_polys[:, 1, 0] -= ratio * sub_r[:, 1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2(
            sub_polys[:, 2, 0] - sub_polys[:, 3, 0],
            sub_polys[:, 2, 1] - sub_polys[:, 3, 1]
        )
        sub_polys[:, 3, 1] += ratio * sub_r[:, 3] * np.cos(theta)
        sub_polys[:, 3, 0] += ratio * sub_r[:, 3] * np.sin(theta)
        sub_polys[:, 2, 1] -= ratio * sub_r[:, 2] * np.cos(theta)
        sub_polys[:, 2, 0] -= ratio * sub_r[:, 2] * np.sin(theta)
        
        shrinked[not_mask] = sub_polys
    
    return shrinked

def parallelogram_from_quad(polys):
    """ Converts a quad (4-points) into a valid parallelogram of minimal area """
    # Fit parallelograms from the given points
    fitted_parallelograms = np.zeros((len(polys), 8, 4, 2))
    for i in range(4):
        p0, p1, p2, p3 = [polys[:, (i + j) % 4] for j in range(4)]

        edges          = fit_line(p0, p1)
        backward_edges = fit_line(p0, p3)
        forward_edges  = fit_line(p1, p2)
        opposite_edges = np.zeros_like(forward_edges)
        forward_opposite_edges  = np.zeros_like(forward_edges)
        backward_opposite_edges = np.zeros_like(forward_edges)
        
        mask     = point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3)
        mask_1   = edges[:, 1] == 0
        not_mask = ~mask
        
        if np.any(mask):
            mask_01  = np.logical_and(mask, mask_1)
            mask_0n1 = np.logical_and(mask, ~mask_1)
            
            if np.any(mask_01):
                opposite_edges[mask_01, 0] = 1.
                opposite_edges[mask_01, 2] = -p2[mask_01, 0]

            if np.any(mask_0n1):
                opposite_edges[mask_0n1, 0] = edges[mask_0n1, 0]
                opposite_edges[mask_0n1, 1] = -1.
                opposite_edges[mask_0n1, 2] = p2[mask_0n1, 1] - edges[mask_0n1, 0] * p2[mask_0n1, 0]
        
        if np.any(not_mask):
            mask_n01  = np.logical_and(not_mask, mask_1)
            mask_n0n1 = np.logical_and(not_mask, ~mask_1)
            
            if np.any(mask_n01):
                opposite_edges[mask_n01, 0] = 1.
                opposite_edges[mask_n01, 2] = -p3[mask_n01, 0]

            if np.any(mask_n0n1):
                opposite_edges[mask_n0n1, 0] = edges[mask_n0n1, 0]
                opposite_edges[mask_n0n1, 1] = -1.
                opposite_edges[mask_n0n1, 2] = p3[mask_n0n1, 1] - edges[mask_n0n1, 0] * p3[mask_n0n1, 0]
        
        # move forward edge
        new_p2 = line_cross_point(forward_edges, opposite_edges)
        
        mask     = point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3)
        mask_1   = forward_edges[:, 1] == 0
        not_mask = ~mask
        
        if np.any(mask):
            mask_01  = np.logical_and(mask, mask_1)
            mask_0n1 = np.logical_and(mask, ~mask_1)
            
            if np.any(mask_01):
                forward_opposite_edges[mask_01, 0] = 1.
                forward_opposite_edges[mask_01, 2] = -p0[mask_01, 0]

            if np.any(mask_0n1):
                forward_opposite_edges[mask_0n1, 0] = forward_edges[mask_0n1, 0]
                forward_opposite_edges[mask_0n1, 1] = -1.
                forward_opposite_edges[mask_0n1, 2] = p0[mask_0n1, 1] - forward_edges[mask_0n1, 0] * p0[mask_0n1, 0]
        
        if np.any(not_mask):
            mask_n01  = np.logical_and(not_mask, mask_1)
            mask_n0n1 = np.logical_and(not_mask, ~mask_1)
            
            if np.any(mask_n01):
                forward_opposite_edges[mask_n01, 0] = 1.
                forward_opposite_edges[mask_n01, 2] = -p3[mask_n01, 0]

            if np.any(mask_n0n1):
                forward_opposite_edges[mask_n0n1, 0] = forward_edges[mask_n0n1, 0]
                forward_opposite_edges[mask_n0n1, 1] = -1.
                forward_opposite_edges[mask_n0n1, 2] = p3[mask_n0n1, 1] - forward_edges[mask_n0n1, 0] * p3[mask_n0n1, 0]

        new_p0 = line_cross_point(forward_opposite_edges, edges)
        new_p3 = line_cross_point(forward_opposite_edges, opposite_edges)
        fitted_parallelograms[:, i * 2] = np.stack([new_p0, p1, new_p2, new_p3], axis = 1)
        
        # or move backward edge
        new_p0, new_p1, new_p2, new_p3 = p0, p1, p2, p3

        new_p3 = line_cross_point(backward_edges, opposite_edges)
        
        mask     = point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2)
        mask_1   = backward_edges[:, 1] == 0
        not_mask = ~mask
        
        if np.any(mask):
            mask_01  = np.logical_and(mask, mask_1)
            mask_0n1 = np.logical_and(mask, ~mask_1)
            
            if np.any(mask_01):
                backward_opposite_edges[mask_01, 0] = 1.
                backward_opposite_edges[mask_01, 2] = -p1[mask_01, 0]

            if np.any(mask_0n1):
                backward_opposite_edges[mask_0n1, 0] = backward_edges[mask_0n1, 0]
                backward_opposite_edges[mask_0n1, 1] = -1.
                backward_opposite_edges[mask_0n1, 2] = p1[mask_0n1, 1] - backward_edges[mask_0n1, 0] * p1[mask_0n1, 0]
        
        if np.any(not_mask):
            mask_n01  = np.logical_and(not_mask, mask_1)
            mask_n0n1 = np.logical_and(not_mask, ~mask_1)
            
            if np.any(mask_n01):
                backward_opposite_edges[mask_n01, 0] = 1.
                backward_opposite_edges[mask_n01, 2] = -p2[mask_n01, 0]

            if np.any(mask_n0n1):
                backward_opposite_edges[mask_n0n1, 0] = backward_edges[mask_n0n1, 0]
                backward_opposite_edges[mask_n0n1, 1] = -1.
                backward_opposite_edges[mask_n0n1, 2] = p2[mask_n0n1, 1] - backward_edges[mask_n0n1, 0] * p2[mask_n0n1, 0]

        new_p1 = line_cross_point(backward_opposite_edges, edges)
        new_p2 = line_cross_point(backward_opposite_edges, opposite_edges)
        fitted_parallelograms[:, i * 2 + 1] = np.stack(
            [new_p0, new_p1, new_p2, new_p3], axis = 1
        )

    # Sorts the parallelograms based on their area
    areas = np.abs(polygon_area(fitted_parallelograms))

    batch_axis      = np.arange(len(polys))
    parallelograms  = fitted_parallelograms[
        batch_axis, np.argmin(areas, axis = -1)
    ].astype(np.float32)
    
    coord_sum       = np.sum(parallelograms, axis = -1)
    min_coord_idx   = np.argmin(coord_sum, axis = -1)

    parallelograms[:, 0], parallelograms[:, 1], parallelograms[:, 2], parallelograms[:, 3] = [
        parallelograms[batch_axis, (min_coord_idx + i) % 4] for i in range(4)
    ]
    return parallelograms

def rectangle_from_parallelogram(polys):
    def get_ab_cross_x(a, b, x):
        ab = fit_line(a, b)
        ab_verticle = line_verticle(ab, x)

        return line_cross_point(ab, ab_verticle)

    is_multi = len(polys.shape) == 3
    if not is_multi: polys = np.expand_dims(polys, axis = 0)
    
    p0, p1, p2, p3 = [polys[..., i, :] for i in range(4)]
    
    angles = np.arccos(
        np.sum((p1 - p0) * (p3 - p0), axis = -1) / (
            np.linalg.norm(p0 - p1, axis = -1) * np.linalg.norm(p3 - p0, axis = -1)
        )
    )
    
    mask_0 = angles < 0.5 * np.pi
    mask_1 = np.linalg.norm(p0 - p1, axis = -1) > np.linalg.norm(p0 - p3, axis = -1)
    
    rectangles = np.zeros(polys.shape)
    
    if np.any(mask_0):
        mask_01     = np.logical_and(mask_0, mask_1)
        mask_0n1    = np.logical_and(mask_0, ~mask_1)
        if np.any(mask_01):
            sub_p0, sub_p1, sub_p2, sub_p3 = p0[mask_01], p1[mask_01], p2[mask_01], p3[mask_01]
            rectangles[mask_01] = np.stack([
                sub_p0, get_ab_cross_x(sub_p0, sub_p1, sub_p2),
                sub_p2, get_ab_cross_x(sub_p2, sub_p3, sub_p0)
            ], axis = -2)
        
        if np.any(mask_0n1):
            sub_p0, sub_p1, sub_p2, sub_p3 = p0[mask_0n1], p1[mask_0n1], p2[mask_0n1], p3[mask_0n1]
            rectangles[mask_0n1] = np.stack([
                sub_p0, get_ab_cross_x(sub_p1, sub_p2, sub_p0),
                sub_p2, get_ab_cross_x(sub_p0, sub_p3, sub_p2),
            ], axis = -2)
    
    if np.any(~mask_0):
        mask_n01    = np.logical_and(~mask_0, mask_1)
        mask_n0n1   = np.logical_and(~mask_0, ~mask_1)
        if np.any(mask_n01):
            sub_p0, sub_p1, sub_p2, sub_p3 = p0[mask_n01], p1[mask_n01], p2[mask_n01], p3[mask_n01]
            rectangles[mask_n01] = np.stack([
                get_ab_cross_x(sub_p0, sub_p1, sub_p3), sub_p1,
                get_ab_cross_x(sub_p2, sub_p3, sub_p1), sub_p3
            ], axis = -2)
        
        if np.any(mask_n0n1):
            sub_p0, sub_p1, sub_p2, sub_p3 = p0[mask_n0n1], p1[mask_n0n1], p2[mask_n0n1], p3[mask_n0n1]
            rectangles[mask_n0n1] = np.stack([
                get_ab_cross_x(sub_p0, sub_p3, sub_p1), sub_p1,
                get_ab_cross_x(sub_p1, sub_p2, sub_p3), sub_p3
            ], axis = -2)
    
    return rectangles if is_multi else rectangles[0]

def sort_rectangle(rects):
    """ Sorts the rectangles' points in clock-wise order + returns the rectangles' angles """
    def rearrange(mask, indexes):
        sub_r       = rects[mask]
        batch_axis  = np.arange(len(sub_r))
        sorted_r[mask, 0], sorted_r[mask, 1], sorted_r[mask, 2], sorted_r[mask, 3] = [
            sub_r[batch_axis, idx] for idx in indexes
        ]

    is_multi = len(rects.shape) == 3
    if not is_multi: rects = np.expand_dims(rects, 0)
    # First find the lowest point
    p_lowest = np.argmax(rects[:, :, 1], axis = -1)
    
    sorted_r = np.zeros(rects.shape)
    angles   = np.zeros((len(rects), ))

    batch_axis = np.arange(len(rects))
    mask = np.sum(np.equal(rects[:, :, 1], rects[batch_axis, p_lowest, 1][:, np.newaxis]), axis = -1) == 2
    not_mask = ~mask
    if np.any(mask):
        # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(rects[mask], axis = -1), axis = -1)
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        
        rearrange(mask, (p0_index, p1_index, p2_index, p3_index))
    
    if np.any(not_mask):
        # find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left  = (p_lowest + 1) % 4

        thetas         = np.arctan(np.divide(
            - (rects[batch_axis, p_lowest, 1] - rects[batch_axis, p_lowest_right, 1]),
            (rects[batch_axis, p_lowest, 0] - rects[batch_axis, p_lowest_right, 0]),
            where = not_mask
        ))
        angles[not_mask] = thetas[not_mask]
        #angle = np.arctan(-(rects[not_mask, p_lowest[not_mask]][1] - rect[p_lowest_right][1]) / (rect[][0] - rect[p_lowest_right][0]))

        mask_1    = (thetas / np.pi) * 100 > 45
        mask_n01  = np.logical_and(not_mask, mask_1)
        mask_n0n1 = np.logical_and(not_mask, ~mask_1)

        if np.any(mask_n01):
            #this point is p2
            angles[mask_n01] = - (np.pi / 2 - angles[mask_n01])
            p2_index = p_lowest[mask_n01]
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            
            rearrange(mask_n01, (p0_index, p1_index, p2_index, p3_index))
        
        if np.any(mask_n0n1):
            # this point is p3
            p3_index = p_lowest[mask_n0n1]
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4

            rearrange(mask_n0n1, (p0_index, p1_index, p2_index, p3_index))

    return sorted_r, angles

def generate_roi_rotate_para(box, angle, expand_w = 60):
    """
        Generates ROI parameters
        Arguments :
            - box : the points for (possibly multiple) box(es)
                np.ndarray of shape [n_boxes (optional), 4, 2], last-axis are coordinates (y, x)
            - angle : the rotation angle of (each) box(es)
                np.ndarray of shape [n_boxes] or simple scalar
        Returns : (box, rect, angle)
            - box  : [x0, y0, w, h] where (x, y) are the coordinates for the center of the box
            - rect : [x0, y0, x1, y1] coordinates
        
        **Note** : the `y` coordinate is always the vertical axis and `x` the horizontal axis
            When points are given (e.g. the `box` argument), the coordinates are (y, x)
            When it is a box / rectangle, coordinates typically start by `x` (the horizontal axis) (e.g. [x, y, w, h])
    """
    is_multi = len(box.shape) == 3
    if not is_multi: box, angle = np.expand_dims(box, 0), np.expand_dims(angle, 0)

    p0_rect, p1_rect, p2_rect, p3_rect = np.transpose(box, [1, 0, 2])
    center  = (p0_rect + p2_rect) / 2.
    w       = np.linalg.norm(p0_rect - p1_rect, axis = -1)
    h       = np.linalg.norm(p0_rect - p3_rect, axis = -1)
    rrect   = np.stack([center[:, 1], center[:, 0], w, h], axis = -1)

    mins, maxs   = np.min(box, axis = 1), np.max(box, axis = 1)
    x_min, x_max = mins[:, 1], maxs[:, 1]
    y_min, y_max = mins[:, 0], maxs[:, 0]

    bbox = np.stack([x_min, y_min, x_max, y_max], axis = 1)
    if np.any(bbox < -expand_w):
        return None
    
    rrect[:, :2] -= bbox[:, :2]
    rrect[:, :2] -= rrect[:, 2:] / 2
    rrect[:, 2:] += rrect[:, :2]

    bbox[:, 2:] -= bbox[:, :2]

    rrect[:, ::2] = np.clip(rrect[:, ::2], 0, bbox[:, 2:3])
    rrect[:, 1::2] = np.clip(rrect[:, 1::2], 0, bbox[:, 3:4])
    rrect[:, 2:] -= rrect[:, :2]
    
    return bbox.astype(np.int32), rrect.astype(np.int32), - angle

def restore_roi_rotate_para(box):
    rectange, rotate_angle = sort_rectangle(box)
    return generate_roi_rotate_para(rectange, rotate_angle)

def get_rbox_map(polys,
                 img_shape,
                 out_shape  = None,
                 labels     = None,
                 mapping    = None,
                 
                 min_poly_size  = 1,
                 shrink_ratio   = 0,
                 max_wh_factor  = 10
                ):
    """
        Generates score_map and geo_map

        Arguments :
            - polys     : np.ndarray of shape [N, 4, 2] of `(y, x)` coordinates
            - img_shape : (h, w) dimensions of the original image
            - out_shape : (h, w) dimensions of the maps to generate
            
            - labels    : the labels associated to the polys
            - mapping   : the label-to-id mapping
            
            - min_poly_size : the minimal area of a polygon to be valid
            - shrink_ratio  : the shrinking ratio (see `shrink_poly`)
            - max_wh_factor : maximal factor between polys width / height to be valid
        Returns :
            - score_map : np.ndarray of shape `out_shape` with value of 1 for pixels within a box
            - geo_map   : np.ndarray of shape `out_shape + (5, )` where the last axis represents
                the distance between the top / right / bottom / left sides of the rectangle
                and the rotation angle
    """
    if out_shape is None:   out_shape = img_shape
    if labels is None:      labels = [''] * len(polys)
    
    h, w = img_shape
    
    score_map  = np.zeros(out_shape,         dtype = np.float32)
    geo_map    = np.zeros(out_shape + (5, ), dtype = np.float32)
    class_map  = np.zeros(out_shape,         dtype = np.int32) if mapping else None
    poly_mask  = np.zeros(out_shape,         dtype = np.uint8)
    valid_mask = np.ones(out_shape,          dtype = np.uint8)

    h_factor  = out_shape[0] / h
    w_factor  = out_shape[1] / w
    wh_factor = np.array([[[w_factor, h_factor]]])
        
    valid_polys, valid_labels = [], []
    for poly, label in zip(polys, labels):
        poly   = np.reshape(poly, [-1, 2])
        # filters duplicate points
        points = [tuple(p) for p in poly]
        poly   = np.array([
            p_i for i, p_i in enumerate(points) if p_i != (-1, -1) and p_i not in points[:i]
        ])
        if poly.shape == (4, 2):
            valid_polys.append(poly)
            valid_labels.append(label)
        else:
            valid_mask = cv2.fillPoly(
                valid_mask, np.expand_dims(poly, 0).astype(np.int32), 0
            )
    
    polys, labels = check_and_validate_polys(
        np.array(valid_polys), np.array(valid_labels), img_shape
    )
    if len(polys) == 0:
        valid_mask = valid_mask.astype(bool)
        polys   = polys.astype(np.int32)
        return (score_map, geo_map, valid_mask, polys) if not mapping else (score_map, geo_map, class_map, valid_mask, polys)

    polys = polys * wh_factor
    
    r     = np.stack([np.minimum(
        np.linalg.norm(polys[:, i] - polys[:, (i + 1) % 4], axis = -1),
        np.linalg.norm(polys[:, i] - polys[:, (i - 1) % 4], axis = -1)
    ) for i in range(4)], axis = -1)

    shrinked_polys = shrink_poly(polys, r, ratio = shrink_ratio).astype(np.int32)

    # Sets values for the score_map
    for p in shrinked_polys: cv2.fillPoly(score_map, p[np.newaxis, :, :], 1.)
    for i, p in enumerate(shrinked_polys): cv2.fillPoly(poly_mask, p[np.newaxis, :, :], i + 1)

    # Generates parallelograms then rectangles from the 4 points
    parallelograms = parallelogram_from_quad(polys)

    # Computes the rectangle according to the parallelograms
    rectangles     = rectangle_from_parallelogram(parallelograms)
    rectangles, rotate_angles = sort_rectangle(rectangles)

    p0_rect, p1_rect, p2_rect, p3_rect = [rectangles[:, i] for i in range(4)]

    # if the poly is too small, then ignore it during training
    poly_h = np.minimum(
        np.linalg.norm(p0_rect - p3_rect, axis = -1), np.linalg.norm(p1_rect - p2_rect, axis = -1)
    )
    poly_w = np.minimum(
        np.linalg.norm(p0_rect - p1_rect, axis = -1), np.linalg.norm(p2_rect - p3_rect, axis = -1)
    )

    invalids = np.logical_or(
        np.minimum(poly_h, poly_w) < min_poly_size,
        poly_h > poly_w * max_wh_factor
    )
    valids   = ~invalids
    if np.any(invalids):
        for p in polys[invalids].astype(np.int32): cv2.fillPoly(valid_mask, p[np.newaxis, :, :], 0)

    for i in range(len(polys)):
        points      = np.argwhere(poly_mask == i + 1)
        xy_points   = points[..., ::-1]
        tiled_rect  = np.repeat(rectangles[i:i+1], len(points), axis = 0)
        # top
        geo_map[points[:, 0], points[:, 1], 0] = point_dist_to_line(
            tiled_rect[:, 0], tiled_rect[:, 1], xy_points
        )
        # right
        geo_map[points[:, 0], points[:, 1], 1] = point_dist_to_line(
            tiled_rect[:, 1], tiled_rect[:, 2], xy_points
        )
        # down
        geo_map[points[:, 0], points[:, 1], 2] = point_dist_to_line(
            tiled_rect[:, 2], tiled_rect[:, 3], xy_points
        )
        # left
        geo_map[points[:, 0], points[:, 1], 3] = point_dist_to_line(
            tiled_rect[:, 3], tiled_rect[:, 0], xy_points
        )
        # angle
        geo_map[points[:, 0], points[:, 1], 4] = rotate_angles[i]

    valid_mask = valid_mask.astype(bool)
    polys   = polys.astype(np.int32)
    return (score_map, geo_map, valid_mask, polys) if not mapping else (score_map, geo_map, class_map, valid_mask, polys)

# https://github.com/Masao-Taketani/FOTS_OCR
def restore_rectangle_rbox(origin, geometry):
    """
        Restore the RBOX given their original position `origin` and their RBOX geometries `geometry`
            - origin    : the points with shape [N, 2]
            - geometry  : the 
    """
    d, angle = geometry[:, :4], geometry[:, 4]
    rbox     = np.empty((len(angle), 4, 2))
    
    mask     = angle >= 0
    not_mask = ~mask
    
    angle_0, angle_1 = angle[mask], angle[not_mask]
    # for angle > 0
    if len(angle_0) > 0:
        origin_0, d_0 = origin[mask], d[mask]

        zeros = np.zeros((len(d_0), ))
        p = np.array([
            zeros,
            - d_0[:, 0] - d_0[:, 2],
            d_0[:, 1] + d_0[:, 3],
            - d_0[:, 0] - d_0[:, 2],
            d_0[:, 1] + d_0[:, 3],
            zeros,
            zeros,
            zeros,
            d_0[:, 3],
            - d_0[:, 2]
        ]).transpose((1, 0)).reshape([-1, 5, 2])
        
        sin, cos = np.sin(angle_0), np.cos(angle_0)
        rotate_matrix_x = np.repeat(np.array([cos, sin]).T, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
        rotate_matrix_y = np.repeat(np.array([-sin, cos]).T, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        p_rotate = np.concatenate([
            np.sum(rotate_matrix_x * p, axis = 2, keepdims = True),  # N*5*1
            np.sum(rotate_matrix_y * p, axis = 2, keepdims = True)   # N*5*1
        ], axis = 2)

        p3_in_origin = np.expand_dims(origin_0, axis = 1) - p_rotate[:, 4:, :]
        rbox[:len(angle_0)] = p_rotate[:, :4] + p3_in_origin

    if len(angle_1) > 0:
        origin_1, d_1 = origin[not_mask], d[not_mask]
        
        zeros = np.zeros((len(d_1), ))
        p = np.array([
            - d_1[:, 1] - d_1[:, 3],
            - d_1[:, 0] - d_1[:, 2],
            zeros,
            - d_1[:, 0] - d_1[:, 2],
            zeros,
            zeros,
            - d_1[:, 1] - d_1[:, 3],
            zeros,
            - d_1[:, 1],
            - d_1[:, 2]
        ]).transpose((1, 0)).reshape([-1, 5, 2])
        
        sin, cos = np.sin(- angle_1), np.cos(- angle_1)
        rotate_matrix_x = np.repeat(np.array([cos, -sin]).T, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
        rotate_matrix_y = np.repeat(np.array([sin, cos]).T, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        p_rotate = np.concatenate([
            np.sum(rotate_matrix_x * p, axis = 2, keepdims = True),  # N*5*1
            np.sum(rotate_matrix_y * p, axis = 2, keepdims = True)   # N*5*1
        ], axis = 2)

        p3_in_origin = np.expand_dims(origin_1, axis = 1) - p_rotate[:, 4:, :]
        rbox[len(angle_0) :] = p_rotate[:, :4] + p3_in_origin

    return rbox

""" These functions are inspired from https://github.com/SakuraRiven/EAST """
def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def filter_polys(res, input_shape):
    input_shape = np.array(input_shape)[::-1].reshape((1, 1, -1))
    return np.sum(
        np.any(res < 0, axis = -1) | np.any(res >= input_shape, axis = -1), axis = -1
    ) <= 1

def restore_polys(pos, d, angle, input_shape, output_shape):
    scale   = np.array(input_shape) // np.array(output_shape)
    pos     = pos * scale.reshape((1, -1))

    x, y    = pos[:, 0], pos[:, 1]
    
    y_min, y_max    = y - d[:, 0], y + d[:, 1]
    x_min, x_max    = x - d[:, 2], x + d[:, 3]

    rotate_mat  = get_rotation_matrix(- angle)

    temp_x      = np.array([[x_min, x_max, x_max, x_min]]) - x
    temp_y      = np.array([[y_min, y_min, y_max, y_max]]) - y
    coordinates = np.concatenate((temp_x, temp_y), axis = 0)

    res = np.matmul(
        np.transpose(coordinates, [2, 1, 0]),
        np.transpose(rotate_mat, [2, 1, 0])
    )
    res[:, :, 0] += x[:, np.newaxis]
    res[:, :, 1] += y[:, np.newaxis]

    mask = filter_polys(res, input_shape)

    return res[mask], np.argwhere(mask)[:, 0]

def restore_polys_from_map(score_map,
                           geo_map      = None,
                           theta_map    = None,
                           rbox_map     = None,
                           scale        = 1.,
                           threshold    = 0.5,
                           nms_threshold    = 0.25,
                           normalize    = False,
                           return_boxes = False,
                           ** kwargs
                          ):
    assert rbox_map is not None or (geo_map is not None and theta_map is not None)
    
    if rbox_map is not None:
        geo_map     = rbox_map[:, :, :, :4]
        theta_map   = rbox_map[:, :, :, 4]
    
    if len(score_map.shape) == 4:
        return [restore_polys_from_map(
            score_map   = s_map,
            geo_map     = g_map,
            theta_map   = t_map,
            scale   = scale,
            normalize   = normalize,
            threshold   = threshold,
            nms_threshold   = nms_threshold,
            ** kwargs
        ) for s_map, g_map, t_map in zip(score_map, geo_map, theta_map)]
    
    if len(score_map.shape) == 3:
        score_map   = score_map[:, :, 0]
        theta_map   = theta_map[:, :, 0]
    
    # filter the score map
    points = np.argwhere(score_map > threshold)

    # sort the text boxes via the y axis
    points  = points[np.argsort(points[:, 0])]
    scores  = score_map[points[:, 0], points[:, 1]]
    # restore
    input_shape = np.array(score_map.shape) * scale
    valid_polys, valid_indices = restore_polys(
        points[:, ::-1],
        geo_map[points[:, 0], points[:, 1]],
        theta_map[points[:, 0], points[:, 1]],
        input_shape     = input_shape,
        output_shape    = score_map.shape
    )
    scores  = scores[valid_indices]
    
    if normalize:
        input_shape_wh  = np.array(input_shape)[::-1].reshape(1, 1, 2)
        valid_polys     = (valid_polys / input_shape_wh).astype(np.float32)

    boxes, box_mode = valid_polys, BoxFormat.POLY
    if nms_threshold < 1.:
        boxes, scores = nms(
            boxes, scores, box_mode = box_mode, nms_threshold = nms_threshold, ** kwargs
        )
        box_mode    = BoxFormat.CORNERS2

    return convert_box_format(
        boxes, BoxFormat.DICT, box_mode = box_mode, score = scores
    )

""" Tf-graph optimized functions (but they are slower than the numpy version) """
@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def tf_polygon_area(poly : tf.Tensor):
    rolled = tf.roll(poly, shift = 1, axis = -2)

    diff = (poly[..., 0] - rolled[..., 0]) * (poly[..., 1] + rolled[..., 1])
    return tf.abs(tf.reduce_sum(diff, axis = -1) / 2)

def tf_restore_rectangle_rbox(origin, geometry):
    ''' Resotre rectangle tbox'''
    d, angle = geometry[:, :4], geometry[:, 4]
    
    mask     = angle >= 0
    not_mask = tf.logical_not(mask)
    # for angle > 0
    if tf.reduce_any(mask):
        origin_0 = tf.boolean_mask(origin, mask)
        d_0      = tf.boolean_mask(d,      mask)
        angle_0  = tf.boolean_mask(angle,  mask)
        
        zeros = tf.zeros((tf.shape(d_0)[0], ))
        p = tf.reshape(tf.transpose(tf.stack([
            zeros,
            d_0[:, 0] - d_0[:, 2],
            d_0[:, 1] + d_0[:, 3],
            d_0[:, 0] - d_0[:, 2],
            d_0[:, 1] + d_0[:, 3],
            zeros,
            zeros,
            zeros,
            d_0[:, 3],
            d_0[:, 2]
        ], axis = 0)), (-1, 5, 2))  # N*5*2

        sin, cos = tf.sin(angle_0), tf.cos(angle_0)
        rotate_matrix_x = tf.expand_dims(tf.stack([cos,  sin], axis = 1), axis = -1)
        rotate_matrix_x = tf.transpose(tf.tile(rotate_matrix_x, [1, 1, 5]), (0, 2, 1))  # N*5*2

        rotate_matrix_y = tf.expand_dims(tf.stack([- sin, cos], axis = 1), axis = -1)
        rotate_matrix_y = tf.transpose(tf.tile(rotate_matrix_y, [1, 1, 5]), (0, 2, 1))  # N*5*2

        p_rotate = tf.concat([
            tf.reduce_sum(rotate_matrix_x * p, axis = 2, keepdims = True),  # N*5*1
            tf.reduce_sum(rotate_matrix_y * p, axis = 2, keepdims = True)   # N*5*1
        ], axis = 2)

        p3_in_origin = tf.expand_dims(origin_0, axis = 1) - p_rotate[:, 4:, :]
        new_p_0 = p_rotate[:, :4] + p3_in_origin
    else:
        new_p_0 = tf.zeros((0, 4, 2))

    if tf.reduce_any(not_mask):
        origin_1 = tf.boolean_mask(origin, not_mask)
        d_1      = tf.boolean_mask(d,      not_mask)
        angle_1  = tf.boolean_mask(angle,  not_mask)
        
        zeros = tf.zeros((tf.shape(d_1)[0], ))

        p = tf.reshape(tf.transpose(tf.stack([
            -d_1[:, 1] - d_1[:, 3],
            d_1[:, 0] - d_1[:, 2],
            zeros,
            -d_1[:, 0] - d_1[:, 2],
            zeros,
            zeros,
            -d_1[:, 1] - d_1[:, 3],
            zeros,
            -d_1[:, 1],
            -d_1[:, 2]
        ], axis = 0)), [-1, 5, 2])

        sin, cos = tf.sin(- angle_1), tf.cos(- angle_1)
        rotate_matrix_x = tf.expand_dims(tf.stack([cos, sin], axis = 1), axis = -1)
        rotate_matrix_x = tf.transpose(tf.tile(rotate_matrix_x, [1, 1, 5]), (0, 2, 1))  # N*5*2

        rotate_matrix_y = tf.expand_dims(tf.stack([sin, - cos], axis = 1), axis = -1)
        rotate_matrix_y = tf.transpose(tf.tile(rotate_matrix_y, [1, 1, 5]), (0, 2, 1))  # N*5*2

        p_rotate = tf.concat([
            tf.reduce_sum(rotate_matrix_x * p, axis = 2, keepdims = True),  # N*5*1
            tf.reduce_sum(rotate_matrix_y * p, axis = 2, keepdims = True)   # N*5*1
        ], axis = 2)

        p3_in_origin = tf.expand_dims(origin_1, axis = 1) - p_rotate[:, 4:, :]
        new_p_1 = p_rotate[:, :4] + p3_in_origin
    else:
        new_p_1 = tf.zeros((0, 4, 2))
    return tf.concat([new_p_0, new_p_1], axis = 0)

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None, None, 6), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32)
])
def tf_decode_output(model_output, threshold = 0.5, box_threshold = 0.1, nms_threshold = 0.2):
    score_map = model_output[0, :, :, :1]
    geo_map   = model_output[0, :, :, 1:] * score_map

    # filter the score map
    xy_text = tf.where(score_map[:, :, 0] > threshold)

    # sort the text boxes via the y axis
    
    xy_text = tf.gather(xy_text, tf.argsort(xy_text[:, 0]))

    # restore
    return tf_restore_rectangle_rbox(
        tf.cast(xy_text[:, ::-1], tf.float32), tf.gather_nd(geo_map, xy_text)
    )
