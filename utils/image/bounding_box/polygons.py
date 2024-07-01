# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np

from loggers import timer

logger = logging.getLogger(__name__)

""" These functions are inspired from https://github.com/SakuraRiven/EAST """

def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def filter_polys(res, input_shape):
    input_shape = input_shape[::-1][None, None, :]
    return np.count_nonzero(
        np.any(res < 0, axis = -1) | np.any(res >= input_shape, axis = -1), axis = -1
    ) <= 1

@timer
def restore_polys(pos, d, angle, input_shape, output_shape):
    scale   = np.array(input_shape) // np.array(output_shape)
    pos     = pos * scale[None]

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

@timer
def restore_polys_from_map(score_map,
                           geo_map      = None,
                           theta_map    = None,
                           rbox_map     = None,
                           
                           scale    = 1.,
                           normalize    = False,
                           
                           threshold    = 0.5,
                           
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

    return {
        'boxes' : valid_polys, 'scores' : scores, 'format' : 'poly'
    }

