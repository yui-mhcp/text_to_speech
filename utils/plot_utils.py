
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
import math
import logging
import datetime
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

_tick_label_limit = 30

_numeric_type   = (int, float, np.integer, np.floating)
_data_iterable  = (list, tuple, np.ndarray)

_keys_to_propagate  = (
    'x', 'hlines', 'vlines', 'hlines_kwargs', 'vlines_kwargs', 'legend_kwargs',
    'xtick_labels', 'ytick_labels', 'tick_labels', 'marker', 'marker_kwargs'
)

_default_audio_plot_config  = {
    'title' : 'Audio signal',
    'xlabel'    : 'time (sec)'
}

_default_spectrogram_plot_config = {
    'title'         : 'Spectrograms',
    'use_subplots'  : True,
    'horizontal'    : False,
    'ncols'         : 1,
    'x_size'        : 5,
    'y_size'        : 3,
    'with_colorbar' : True,
    'orientation'   : 'horizontal',
    'xlabel'        : 'Frames',
    'ylabel'        : 'Frequency (Hz)',
    'plot_type'     : 'imshow'
}

_default_cm_plot_config = {
    'title'     : 'Confusion Matrix',
    'xlabel'    : 'Predicted label',
    'ylabel'    : 'True label',
    'factor_size'   : 0.75,
    'with_colorbar' : True
}

_default_matrix_plot_config = {
    'title'     : 'Matrix',
    'xtick_rotation'    : 45,
    'ytick_rotation'    : 45,
    'with_colorbar' : True
}

_default_classification_plot_config = {
    'title'     : 'Top-k scores',
    'xlabel'    : 'Label',
    'ylabel'    : 'Score (%)',
    'xtick_rotation'    : 45
}

_default_embedding_plot_config  = {
    'title'         : 'Embedding space',
    'figsize'       : (10, 10),
    'tick_labels'   : [],
    'with_grid'     : True,
    'cmap'          : 'tab10'
}

def _normalize_colors(config):
    c = config.get('c', None)
    if c is not None:
        if isinstance(c, _data_iterable) and all([isinstance(ci, int) for ci in c]):
            mapper = plt.cm.ScalarMappable(cmap = config.pop('cmap', None))
            config['c'] = mapper.to_rgba(c)

def plot(x, y = None, * args, ax = None, figsize = None, xlim = None, ylim = None,

         title = None, xlabel = None, ylabel = None, 
         xtick_labels = None, ytick_labels = None, tick_labels = None,
         xtick_pos = None, ytick_pos = None,
         xtick_rotation = 0, ytick_rotation = 0, tick_rotation = 0,
         titlesize = 15, fontsize = 13, labelsize = 11, fontcolor = 'w',
         
         with_legend = True, legend_fontsize = 11, legend_kwargs = {},
         
         vlines = None, vlines_kwargs = {}, hlines = None, hlines_kwargs = {},
         
         with_colorbar = False, orientation = 'vertical',
         
         with_grid = False, grid_kwargs = {},
         
         date_format = '%H : %M : %S',
         
         linewidth = 2.5, color = 'red', facecolor = 'black',
         
         filename = None, show = True, close = True, new_fig = True, 
         
         plot_type = 'plot', ** kwargs
        ):
    """
        Plot functions that combines multiple matplotlib functions. 
        Arguments :
            - x     : values for the x axis
            - y     : values for y axis - If y is None, x = range(len(x)) and y = x
            - args  : additional lines to plot
            - figsize   : size for the figure (useless if ax is specified or not new_fig)
            - ax    : the axes on which to apply functions (if None, plt.gca())
            - xlim / ylim   : limits of values for axes
            
            - title : title of the figure
            - xlabel / ylabel   : labels for axis names
            - xtick_labels / ytik_labels / tick_labels  : labels for values on axis
            - titlesize / fontsize / labelsize / fontcolor  : parameters to modify size and colors of different labels
            
            - legend_fontsize / legend_kwargs   : kwargs for legend (only relevant if y is a dict)

            - vlines / vlines_kwargs / hlines / hlines_kwargs   : vertical / horizontal lines

            - with_colorbar / orientation   : whether to add a colorbar or not (useful for some images such as spectrogram)
            
            - with_grid / grid_kwargs   : whether to add a grid
            
            - date_format   : how to show dates if x is an array of datetime.datetime
            
            - linewidth / color / facecolor : style of the line
            
            - filename  : where to save the figure
            - show      : whether to show final result or not
            - close     : whether to close the figure or not
            - new_fig   : if ax is None, whether to create new fig or not
            - plot_type : a name of plt function to call for plot (plot / scatter / imshow / bar / hist / boxplot / ...)
            - kwargs    : additional kwargs to the plot function
    """
    def _plot(ax, p_type, datas, ** kwargs):
        try:
            return getattr(ax, p_type)(* datas, ** kwargs)
        except Exception as e:
            logger.error('Error while calling `plt.{}` with data {}\n  Config : {}'.format(
                p_type, datas, kwargs
            ))
            raise e
    
    def _maybe_add_legend():
        if with_legend:
            config = legend_kwargs.copy()
            config.setdefault('facecolor', facecolor)
            legend_fontcolor = config.pop('fontcolor', fontcolor)
            
            leg = ax.legend(fontsize = legend_fontsize, ** config)
            if 'title' in config: leg.get_title().set_color(legend_fontcolor)
            for text in leg.get_texts(): text.set_color(legend_fontcolor)
        
    def _select_datas(label, datas, data_labels, plot_config, specific_config):
        selected_datas = [
            [d for i, d in enumerate(datas_i) if data_labels[i] == label]
            for datas_i in datas
        ]
        selected_config   = {
            k : v if not isinstance(v, _data_iterable) else [
                v_i for i, v_i in enumerate(v) if data_labels[i] == label
            ] for k, v in plot_config.items()
        }
        selected_config.update(specific_config.get(label, {}))
        return selected_datas, selected_config
        
    def normalize_xy(x, y, config):
        p_type, p_config = plot_type, config.copy()
        if callable(y):
            if x is None:
                assert xlim is not None, "When y is a callable, you must specify x or xlim"

                x0, x1 = xlim
                x = np.linspace(x0, x1, int((x1 - x0) * 10))
            y = y(x)
        elif isinstance(y, dict):
            x, yi, p_type = y.pop('x', x), y.pop('y'), y.pop('plot_type', plot_type)

            p_config, y = {** p_config, ** y}, yi
        
        datas = (x, y) if x is not None else (y, )
        _normalize_colors(p_config)
        return datas, p_type, p_config
    
    def _plot_data(ax, x, y, config):
        datas, p_type, plot_config = normalize_xy(x, y, config)

        if isinstance(plot_config.get('label', ''), _data_iterable):
            labels, labels_config = plot_config.pop('label'), plot_config.pop('label_kwargs', {})
            im = None
            for l in np.unique(labels):
                datas_label, config_label = _select_datas(
                    l, datas, labels, plot_config, labels_config
                )
                im = _plot_data(ax, * datas_label, {'label' : l, ** config_label})
            _maybe_add_legend()
            return im
        elif p_type == 'scatter' and isinstance(plot_config.get('marker', 'o'), _data_iterable):
            markers, markers_config = plot_config.pop('marker'), plot_config.pop('marker_kwargs', {})
            im = None
            for m in np.unique(markers):
                datas_marker, config_marker = _select_datas(
                    m, datas, markers, plot_config, markers_config
                )
                m = config_marker.pop('marker', m)
                im = _plot(ax, p_type, datas_marker, marker = m, ** config_marker)
            return im
        else:
            if p_type != 'scatter': plot_config.pop('marker', None)
            return _plot(ax, p_type, datas, ** plot_config)

    if not hasattr(plt, plot_type):
        raise ValueError("`plot_type` must be a valid matplotlib.pyplot method (such as plot, scatter, hist, imshow, ...)\n  Got : {}".format(plot_type))
    # Maybe create the figure and get the axis
    if y is None: x, y = None, x
    if new_fig and ax is None:
        fig = plt.figure(figsize = figsize)
        if facecolor: fig.set_facecolor(facecolor)
    
    if ax is None: ax = plt.gca()
    # Modify the axis' style and set labels
    if xlim is not None:    ax.set_xlim(xlim)
    if ylim is not None:    ax.set_ylim(ylim)
    if title is not None:   ax.set_title(title, fontsize = titlesize, color = fontcolor)
    if xlabel is not None:  ax.set_xlabel(xlabel, fontsize = fontsize, color = fontcolor)
    if ylabel is not None:  ax.set_ylabel(ylabel, fontsize = fontsize, color = fontcolor)
    if facecolor is not None:   ax.set_facecolor(facecolor)
    
    ax.tick_params(
        axis = 'both', labelsize = labelsize, labelcolor = fontcolor, color = fontcolor
    )
    
    color = kwargs.get('c', color)
    
    if isinstance(y, (tf.Tensor, np.ndarray)) and len(y.shape) == 3: plot_type = 'imshow'
    
    if plot_type == 'boxplot':
        kwargs.update({
            'patch_artist'  : True,
            'boxprops'      : {'color' : color, 'facecolor' : color},
            'capprops'      : {'color' : color},
            'whiskerprops'  : {'color' : color},
            'flierprops'    : {'color' : color, 'markeredgecolor' : color},
            'medianprops'   : {'color' : color}
        })
    elif plot_type != 'imshow':
        kwargs['linewidth'] = linewidth
        if 'c' not in kwargs: kwargs['color'] = color

    if plot_type == 'bar' and x is None and isinstance(y, dict):
        xtick_labels, y = list(zip(* y.items()))
    if plot_type == 'imshow' and y.ndim == 3 and y.shape[-1] == 1:
        y = y[:,:,0]
    if plot_type in ('bar', 'scatter') and x is None:
        x = np.arange(len(y))
    
    if x is not None and isinstance(x[0], datetime.datetime):
        formatter = mdates.DateFormatter(date_format)
        ax.xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate()
    
    
    im = None
    if isinstance(y, dict):
        if len(y) > 0: kwargs.pop('color', None)
        for label, data in y.items():
            im = _plot_data(ax, x, data, {'label' : label, ** kwargs})
        
        _maybe_add_legend()
    else:
        im = _plot_data(ax, x, y, kwargs)
    
    if hlines is not None:
        hlines_kwargs.setdefault('colors', color)
        if xlim is None:
            h_colors = hlines_kwargs.pop('colors')
            if not isinstance(hlines, _data_iterable): hlines = [hlines]
            if not isinstance(h_colors, _data_iterable): h_colors = [h_colors]
            if len(h_colors) < len(hlines): h_colors = h_colors * len(hlines)
            for line, lc in zip(hlines, h_colors): ax.axhline(line, color = lc, ** hlines_kwargs)
        else:
            xmin, xmax = xlim
            ax.hlines(hlines, xmin, xmax, ** hlines_kwargs)
    
    if vlines is not None:
        vlines_kwargs.setdefault('colors', color)
        if ylim is None:
            v_colors = vlines_kwargs.pop('colors')
            if not isinstance(vlines, _data_iterable): vlines = [vlines]
            if not isinstance(v_colors, _data_iterable): v_colors = [v_colors]
            if len(v_colors) < len(vlines): v_colors = v_colors * len(vlines)
            for line, lc in zip(vlines, v_colors): ax.axvline(line, color = lc, ** vlines_kwargs)
        else:
            ymin, ymax = ylim
            ax.vlines(vlines, ymin, ymax, ** vlines_kwargs)
    
    if with_colorbar and plot_type == 'imshow':
        cb = ax.figure.colorbar(im, orientation = orientation, ax = ax)
        cb.ax.tick_params(
            axis = 'both', labelsize = labelsize, labelcolor = fontcolor, color = fontcolor
        )
    
    if xtick_labels is None: xtick_labels = tick_labels
    if ytick_labels is None: ytick_labels = tick_labels

    if xtick_labels is not None and len(xtick_labels) > 0 and len(xtick_labels) < _tick_label_limit:
        xtick_labels = [str(l) for l in xtick_labels]
        if xtick_pos is None:
            xtick_pos = np.linspace(0, int(ax.dataLim.x1), len(xtick_labels))
        ax.set_xticks(xtick_pos, labels = xtick_labels)
        
    if ytick_labels is not None and len(ytick_labels) > 0 and len(ytick_labels) < _tick_label_limit:
        if ytick_pos is None:
            ytick_pos = np.linspace(0, int(ax.dataLim.y1), len(ytick_labels))
        ax.set_yticks(ytick_pos, labels = ytick_labels)
    
    if xtick_rotation == 0: xtick_rotation = tick_rotation
    if ytick_rotation == 0: ytick_rotation = tick_rotation

    if xtick_rotation != 0: ax.tick_params(axis = 'x', labelrotation = xtick_rotation)
    if ytick_rotation != 0: ax.tick_params(axis = 'y', labelrotation = ytick_rotation)

    if with_grid: ax.grid(** grid_kwargs)
    
    plt.tight_layout()
    
    if filename is not None: 
        plt.savefig(
            filename, edgecolor = fontcolor, facecolor = facecolor
        )
    if show: plt.show()
    if close: 
        plt.close()
    else:
        return ax, im

def plot_multiple(* args, size = 5, x_size = None, y_size = None, ncols = 2, nrows = None,
                  use_subplots = False, horizontal = False,
                  # for pd.DataFrame grouping
                  by = None, corr = None,
                  color_corr = None, color_order = None,
                  shape_corr = None, shape_order = None,
                  link_from_to = None, links = [],
                  
                  title = None, filename = None, show = False, close = True,
                  ** kwargs
                 ):
    """
        Plot multiple data in a single graph
        Arguments : 
            - args  : data to plot. Can be :
                - tuple : (name, data)
                - dict  : with key 'name' or 'label' for the title
                - pd.DataFrame  : either plot each column or group them on a column (`by` kwarg) or each column in relation with another (`corr` kwarg)
                - else (list, np.ndarray, tf.Tensor)    : adds it as raw data
            - x_size / y_size / nrows / ncols   : information related to subplots' size
            - use_subplots  : whether to plot all data in the same plot or not (default False if `imshow`)
            - horizontal    : whether to make subplots horizontal aligned or vertical
            - title / filename / show / close   : same as plot() function
            - kwargs    : either data to plot or kwargs to pass to each plot() call
        
        pd.DataFrame correlation arguments :
            - by    : the `.groupby()` argument in order to see how other columns evolve depending on the 'by' column
            - corr  : make a subplot with all columns as `x-axis` and `corr` is the `y-axis` value
                For instance you can plot models' performances (x-axis) and see the evolution of the validation loss (y-axis) by setting `corr` to `val_loss`
            - color_corr / shape_corr   : assign a specifi color / shape depending on the column's value
            - color_order / shape_order : associates the value to a color / shape
                - if list   : uses the value's index to get the color in the list (the value index is determined by its position in the result of `.unique()`)
                - if dict   : the key is the column's value and the value is the color / shape
    """
    def _parse_arg(datas, v):
        if isinstance(v, tuple) and len(v) == 2:
            datas.append(v)
        elif isinstance(v, dict):
            datas.append((v.pop('name', v.pop('label', None)), v))
        elif isinstance(v, pd.DataFrame):
            if by is not None:
                for value, datas_i in v.groupby(by):
                    datas_i.pop(by)
                    datas.append(('{} = {}'.format(by, value), {'x' : datas_i.to_dict('list')}))
            elif corr is not None:
                corr_config, corr_colors, corr_shapes = {}, None, None
                if color_corr is not None:
                    if color_corr not in v.columns:
                        logger.error('Color correlation {} is not in data !'.format(color_corr))
                    else:
                        unique_values = list(v[color_corr].unique())
                        corr_colors = [
                            unique_values.index(corr_val_i) for corr_val_i in v[color_corr].values
                        ]
                        if color_order is not None:
                            if len(color_order) < len(unique_values):
                                logger.warning('Not enough colors : {} vs {}'.format(
                                    len(color_order), len(unique_values)
                                ))
                            elif isinstance(color_order, dict):
                                corr_colors = [
                                    color_order[corr_val_i] for corr_val_i in v[color_corr].values
                                ]
                            else:
                                corr_colors = [color_order[color_idx] for color_idx in corr_colors]
                        corr_config['c'] = corr_colors
                
                if shape_corr is not None:
                    if shape_corr not in v.columns:
                        logger.error('Shape correlation {} is not in data !'.format(shape_corr))
                    else:
                        unique_values = list(v[shape_corr].unique())
                        corr_shapes = [
                            unique_values.index(corr_val_i) for corr_val_i in v[shape_corr].values
                        ]
                        if shape_order is not None:
                            if len(shape_order) < len(unique_values):
                                logger.warning('Not enough shapes : {} vs {}'.format(
                                    len(shape_order), len(unique_values)
                                ))
                            elif isinstance(shape_order, dict):
                                corr_shapes = [
                                    shape_order.get(corr_val_i, 'o') for corr_val_i in v[shape_corr].values
                                ]
                            else:
                                corr_shapes = [shape_order[shape_idx] for shape_idx in corr_shapes]
                        corr_config['marker'] = corr_shapes
                
                link_from, link_to = (None, None) if not link_from_to else link_from_to
                links   = []
                if link_from is not None:
                    for i, link_val in enumerate(v[link_from].values):
                        links.append((i, np.where(v[link_to].values == link_val)[0]))
                
                corr_values = v[corr].values
                for col in v.columns:
                    if col in (corr, link_from, link_to): continue
                    col_config  = corr_config.copy()
                    try:
                        unique_vals = v[col].unique()
                        if any([isinstance(val_i, str) for val_i in unique_vals]):
                            col_values = [str(val_i) for val_i in v[col].values]
                            if len(set(col_values)) > 2:
                                col_values = [val_i[:15] for val_i in col_values]
                                if len(set(col_values)) >= 5:
                                    col_config['xtick_rotation'] = 45
                            
                        else:
                            col_values = v[col].values
                    except TypeError:
                        col_values = [str(val_i) for val_i in v[col].values]
                    
                    col_data = {'x' : col_values, 'y' : corr_values}
                    if links:
                        link_values = {}
                        for idx1, links_to in links:
                            links_to = list(links_to)
                            for idx2 in links_to:
                                link_values['link{}'.format(len(link_values))] = {
                                    'x' : [col_values[idx1], col_values[idx2]],
                                    'y' : [corr_values[idx1], corr_values[idx2]],
                                    'plot_type' : 'plot', 'zorder' : 1,
                                    'c' : corr_colors[idx1] if corr_colors else 0
                                }
                        col_data = {'x' : {
                            'points' : {** col_data, 'zorder' : 2}, ** link_values
                        }, 'with_legend' : False}
                    
                    datas.append((col, {
                        ** col_data, 'ylabel' : corr, 'plot_type' : 'scatter', ** col_config
                    }))
            else:
                for k, v_i in v.to_dict('list').items():
                    datas.append((k, v_i))
        else:
            datas.append((None, v))
        
    datas = []
    for v in args:
        if isinstance(v, pd.DataFrame): use_subplots = True
        _parse_arg(datas, v)
    
    data_names = [
        k for k, v in kwargs.items()
        if (isinstance(v, (list, dict, np.ndarray, tf.Tensor)) or callable(v))
        and k not in _keys_to_propagate
    ]
    datas += [(k, kwargs.pop(k)) for k in data_names]
    
    if len(datas) == 0:
        raise ValueError("No valid data to plot ! See help(plot_multiple) to check valid types\n  Got : {}".format(
            '\n'.join([' - {} (type {}) : {}'.format(k, type(v), v) for k, v in kwargs.items()])
        ))
    
    if size is not None:
        if x_size is None: x_size = size
        if y_size is None: y_size = size
    
    use_subplots = use_subplots or kwargs.get('plot_type', '') == 'imshow'
    if use_subplots:
        if len(datas) == 1:
            ncols, nrows = 1, 1
        elif nrows is not None or ncols is not None:
            if ncols is None: ncols = math.ceil(len(datas) / nrows)
            if nrows is None: nrows = math.ceil(len(datas) / ncols)
        elif horizontal:
            nrows, ncols = 1, len(datas)
        else:
            nrows, ncols = len(datas), 1
    else:
        nrows, ncols = 1, 1
    
    if x_size is None or y_size is None:
        figsize = None
    else:
        figsize = (x_size * ncols, y_size * nrows)
    
    fig = plt.figure(figsize = figsize)
    fig.set_facecolor(kwargs.get('facecolor', 'black'))
    
    if title is not None:
        fig.text(
            0.5, 0.99, title, horizontalalignment = 'center', verticalalignment = 'top',
            fontsize    = kwargs.get('fontsize', 15), 
            color       = kwargs.get('fontcolor', 'w')
        )
    
    default_axes_config = {'filename' : None,'show' : False,'close' : False,'new_fig' : False}

    axes = []
    for i, (name, val) in enumerate(datas):
        ax = fig.add_subplot(nrows, ncols, i + 1) if use_subplots else None
        
        if not isinstance(val, dict):
            val = {'x' : val} if 'x' not in kwargs else {'y' : val}

        config_ax = {** kwargs, ** val, 'ax' : ax}

        if use_subplots:
            config_ax.setdefault('title', name)
        else:
            config_ax['color'] = None
            config_ax.setdefault('label', name)
        
        config_ax.update(default_axes_config)
        plot_method = _plot_methods.get(config_ax.get('plot_type'), plot)
        
        axes.append(plot_method(** config_ax)[0])
    
    if filename is not None:
        plt.savefig(
            filename,
            edgecolor = kwargs.get('fontcolor', 'w'),
            facecolor = kwargs.get('facecolor', 'black')
        )
    if show or filename is None: plt.show()
    if close: plt.close()
    else: return axes

def plot_audio(rate, audio = None, x = None, channels = None, n_labels = 10, ** kwargs):
    assert audio is not None or x is not None
    if audio is None: audio = x
    if hasattr(audio, 'numpy'): audio = audio.numpy()
    
    times = np.linspace(0, audio.shape[-1], n_labels)
    kwargs.setdefault('xtick_pos', times)
    kwargs.setdefault('xtick_labels', ['{:.2f}'.format(t / rate) for t in times])

    if len(audio.shape) == 2:
        audio_range = np.max(audio) - np.min(audio)
        audio = [
            channel + i * audio_range for i, channel in enumerate(audio)
        ]
        if channels is None:
            channels = [
                'Ch {}'.format(i) for i in range(len(audio))
            ]
        kwargs.setdefault('ytick_labels', channels)
    
    for k, v in _default_audio_plot_config.items():
        kwargs.setdefault(k, v)
    kwargs['plot_type'] = 'plot'

    if channels and len(channels) > 1:
        return plot({ch : sign for ch, sign in zip(channels, audio)}, with_legend = False, ** kwargs)
    else:
        return plot(audio, ** kwargs)
    
def plot_spectrogram(* args, ** kwargs):
    """
        Call plot_multiple() after normalizing spectrograms : making them 2D images and rotate them by 90° to put the time on x-axis (as models usually generate [B, T, F] spectrograms)
    """
    def _normalize_spect(v):
        if not isinstance(v, (np.ndarray, tf.Tensor)) or len(v.shape) not in (2, 3):
            return v
        if len(v.shape) == 3:
            if len(v) > 1:
                logger.warning('Spectrogram with shape {} : taking only spect[0]'.format(v.shape))
            v = v[0]
        return np.rot90(v)
    
    args    = [_normalize_spect(v) for v in args]
    kwargs  = {k : _normalize_spect(v) for k, v in kwargs.items()}
    
    for k, v in _default_spectrogram_plot_config.items():
        kwargs.setdefault(k, v)
    kwargs.update({'plot_type' : 'imshow'})
    
    return plot_multiple(* args, ** kwargs)

def plot_confusion_matrix(cm = None, true = None, pred = None, x = None, labels = None, ** kwargs):
    """
        Plot a confusion matrix 
        Arguments : 
            - cm    : the confusion matrix (2-D square array)
            - true / pred   : the true labels and predicted labels (used to build `cm` if not provided)
            - labels    : name for each label index
            - kwargs    : forwarded to `plot_matrix`
        
        This function can be used as subplot in `plot_multiple` with `plot_type = 'cm'`
    """
    assert x is not None or cm is not None or (true is not None and pred is not None)
    
    if cm is None:      cm = confusion_matrix(true, pred) if true is not None else x
    if labels is None:  labels = range(cm.shape[0])
    
    for k, v in _default_cm_plot_config.items():
        kwargs.setdefault(k, v)
    
    return plot_matrix(
        cm, y_labels = labels, x_labels = labels, ** kwargs
    )

def plot_matrix(matrix = None, x = None, y_labels = None, x_labels = None, norm = False,
                factor_size = 1., cmap = 'magma', ticksize = 13,
                          
                filename = None, show = True, close = True, ** kwargs
               ):
    """
        Plots a matrix and possibly adds text (the matrix' values). It is a generalization of `plot_confusion_matrix` but with any 2-D matrix (not required to be square)
        
        Arguments :
            - matrix    : the 2-D array of scores
            - {y / x}_labels    : the labels associated with x / y matrix' axes
            - norm      : whether to normalize rows such that sum(matrix[i]) == 1
            - factor_size   : the factor to multiply the matrix' shape to determine `figsize`
            - cmap  : the color map to use
            - filename / show / close   : same as `plot`
            - kwargs    : forwarded to `plot`
        
        This function can be used as subplot in `plot_multiple` with `plot_type = 'matrix'`
    """
    assert matrix is not None or x is not None
    if matrix is None: matrix = x
    if hasattr(matrix, 'numpy'): matrix = matrix.numpy()
    if norm: matrix = matrix.astype('float') / matrix.sum(axis = 1)[:, np.newaxis]
    
    if y_labels is None: y_labels = list(range(matrix.shape[0]))
    if x_labels is None: x_labels = list(range(matrix.shape[1]))
    
    n_y = min(_tick_label_limit, matrix.shape[0])
    n_x = min(_tick_label_limit, matrix.shape[1])
    for k, v in _default_matrix_plot_config.items():
        kwargs.setdefault(k, v)
    kwargs.setdefault('figsize', (n_x * factor_size, n_y * factor_size))
    kwargs.setdefault('ytick_labels', y_labels)
    kwargs.setdefault('xtick_labels', x_labels)
    kwargs['plot_type'] = 'imshow'
    
    ax, im = plot(matrix, show = False, close = False, ** kwargs)
    
    if len(x_labels) <= _tick_label_limit and len(y_labels) <= _tick_label_limit:
        matrix = np.around(matrix, decimals = 2)
        
        threshold = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = 'white' if matrix[i, j] < threshold else 'black'
                im.axes.text(
                    j, i, str(matrix[i, j]), color = color, fontsize = ticksize,
                    verticalalignment = 'center', horizontalalignment = 'center'
                )
    
    if filename is not None: plt.savefig(filename)
    if show or filename is None: plt.show()
    if close: plt.close()
    else: return ax, im

def plot_classification(scores = None, labels = None, k = 5, x = None, ** kwargs):
    """
        Plot classification's scores in decreasing order
        
        Arguments :
            - scores    : 1-D array, the classes' scores
            - labels    : 1-D array (or list), the labels associated to scores
            - k     : the top-k to display
            - kwargs    : forwarded to `plot`
        
        This function can be used inside `plot` or `plot_multiple` with `plot_type = 'classification'`
    """
    assert scores is not None or x is not None
    if scores is None: scores = x
    if hasattr(scores, 'numpy'): scores = scores.numpy()
    if labels is None: labels = np.arange(len(scores))
    
    indexes = np.flip(np.argsort(scores))[:k]
    
    scores  = scores[indexes]
    labels  = np.array(labels)[indexes]
    
    for k, v in _default_classification_plot_config.items():
        kwargs.setdefault(k, v)
    kwargs.setdefault('xtick_labels', labels)
    kwargs['plot_type'] = 'bar'

    return plot(scores, ** kwargs)
    
def plot_embedding(embeddings = None, ids = None, marker = None, random_state = None,
                   remove_extreme = False, x = None, ** kwargs):
    """
        Plot embeddings using the UMAP projection
        Arguments : 
            - embeddings    : list of embeddings (np.ndarray) or pd.DataFrame with 'embedding' col
            - ids       : corresponding ids for each embedding (in order to make same color for embeddings from same ids)
            - marker    : special markers to use for each embedding
            - marker_kwargs : config for markers
            - remove_extreme    : whether to remove extremas points which are to far away from other points
            - kwargs    : general config passed to plot()
    """
    def filter_x(x, y, marker = None):
        sorted_idx = np.argsort(x)
        
        x = x[sorted_idx]
        y = y[sorted_idx]
        if marker is not None: merker = marker[sorted_idx]
        
        subset = x[len(x) // 20 : - len(x) // 20]
        mean = np.mean(subset)
        mean_dist = np.mean(np.abs(subset - mean))
        
        dist = np.abs(x - mean)
        keep = np.where(dist < mean_dist * 5)
        
        x = x[keep]
        y = y[keep]
        if marker is not None: marker = marker[keep]
        
        return x, y, marker
    
    #tsne = TSNE(early_exaggeration = 5, random_state = 10)
    
    #tsne_embeddings = tsne.fit_transform(embeddings)
    
    
    import umap
    
    assert embeddings is not None or x is not None
    if embeddings is None: embeddings = x

    if isinstance(embeddings, pd.DataFrame):
        from utils.embeddings import embeddings_to_np
        
        if 'id' in embeddings and ids is None:
            ids = embeddings['id'].values
        embeddings = embeddings_to_np(embeddings)
    elif isinstance(embeddings, dict):
        if ids is None: ids = embeddings.get('ids', embeddings.get('label', None))
        embeddings = embeddings['embeddings'] if 'embeddings' in embeddings else embeddings['x']

    if embeddings.shape[1] > 2:
        reducer = umap.UMAP(random_state = random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings
    
    x, y = reduced_embeddings[:, 0], reduced_embeddings[:, 1]

    if remove_extreme:
        x, y, marker = filter_x(x, y, marker)
        y, x, marker = filter_x(y, x, marker)
    
    kwargs.update({'x' : x, 'y' : y, 'plot_type' : 'scatter', 'marker' : marker})
    if ids is not None:
        unique_ids  = np.unique(ids).tolist()
        size = min(math.sqrt(len(unique_ids)) * 2, 10)
        
        colors  = [unique_ids.index(i) for i in ids]
        color_mapping   = kwargs.pop('c', None)
        if isinstance(color_mapping, dict):
            colors = [color_mapping.get(c, c) for c in colors]
        elif isinstance(color_mapping, list):
            if len(color_mapping) == len(colors):
                colors = color_mapping
            elif len(color_mapping) >= len(unique_ids):
                colors = [color_mapping[c] for c in colors]
        
        kwargs.setdefault('figsize', (size, size))
        kwargs.update({'label' : ids, 'c' : colors})
    
    for k, v in _default_embedding_plot_config.items():
        kwargs.setdefault(k, v)

    return plot(** kwargs)

_plot_methods   = {
    'cm'    : plot_confusion_matrix,
    'audio' : plot_audio,
    'matrix'    : plot_matrix,
    'classification'    : plot_classification,
    'confusion_matrix'  : plot_confusion_matrix,
    #'spectrogram'   : plot_spectrogram,
    'embedding' : plot_embedding
}