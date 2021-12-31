import cv2
import math
import umap
import datetime
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

_tick_label_limit = 20

def plot(x, y = None, *args, ax = None, figsize = None, xlim = None, ylim = None,

         title = None, xlabel = None, ylabel = None, 
         xtick_labels = None, ytick_labels = None, tick_labels = None,
         titlesize = 15, fontsize = 13, labelsize = 11, fontcolor = 'w',
         
         legend_fontsize = 11, legend_kwargs = {},
         
         vlines = None, vlines_kwargs = {}, hlines = None, hlines_kwargs = {},
         
         with_colorbar = False, orientation = 'vertical',
         
         with_grid = False, grid_kwargs = {},
         
         date_format = '%H : %M : %S',
         
         linewidth = 2.5, color = 'red', facecolor = 'black',
         
         filename=None, show=True, close = True, new_fig = True, 
         
         plot_type='plot', **kwargs
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
    if not hasattr(plt, plot_type):
        raise ValueError("plot_type doit être le nom d'une fonction d'affichage du module matplotlib.pyplot !\n  Reçu : {}\n  Acceptés : {}".format(plot_type, ('plot', 'bar', 'scatter', 'imshow', '...')))
    
    if y is None: x, y = None, x
    if new_fig and ax is None: fig = plt.figure(figsize = figsize)
    
    if ax is None: ax = plt.gca()
    
    if xlim is not None:    ax.set_xlim(xlim)
    if ylim is not None:    ax.set_ylim(ylim)
    if xlabel is not None:  ax.set_xlabel(xlabel, fontsize = fontsize, color = fontcolor)
    if ylabel is not None:  ax.set_ylabel(ylabel, fontsize = fontsize, color = fontcolor)
    if title is not None:   ax.set_title(title, fontsize = titlesize, color = fontcolor)
    if facecolor is not None:   ax.set_facecolor(facecolor)
    
    ax.tick_params(
        axis='both', labelsize = labelsize, labelcolor = fontcolor, color = fontcolor
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

    if plot_type == 'imshow' and y.ndim == 3 and y.shape[-1] == 1:
        y = y[:,:,0]
    if plot_type == 'scatter' and x is None:
        x = np.arange(len(y))
    
    if x is not None and isinstance(x[0], datetime.datetime):
        formatter = mdates.DateFormatter(date_format)
        ax.xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate()
    
    if callable(y):
        if x is None:
            if xlim is None:
                raise ValueError("When y is a callbable, you must specify x or xlim")
            x0, x1 = xlim
            x = np.linspace(x0, x1, int((x1 - x0) * 10))
        y = y(x)
    
    im = None
    if isinstance(y, dict):
        if len(y) > 0: kwargs.pop('color', None)
        for label, data in y.items():
            im = getattr(ax, plot_type)(data, label = label, ** kwargs)
        
        ax.legend(fontsize = legend_fontsize, ** legend_kwargs)
    else:
        vals = (y,) if x is None else (x, y)
        args = vals + args

        im = getattr(ax, plot_type)(* args, ** kwargs)
    
    if hlines is not None:
        if xlim is None:
            if x is None: xmin, xmax = 0, len(y)
            else: xmin, xmax = min(x), max(x)
        else:
            xmin, xmax = xlim
        hlines_kwargs.setdefault('colors', color)
        ax.hlines(hlines, xmin, xmax, ** hlines_kwargs)
    
    if vlines is not None:
        if ylim is None:
            if not isinstance(y, dict):
                all_y = y
            else:
                all_y = []
                for _, y_i in y.items(): all_y.extend(y_i)
            
            all_y = [y_i for y_i in all_y if y_i is not None]
            ymin, ymax = min(all_y), max(all_y)
        else:
            ymin, ymax = ylim
        vlines_kwargs.setdefault('colors', color)
        ax.vlines(vlines, ymin, ymax, ** vlines_kwargs)
    
    if with_colorbar and plot_type == 'imshow': 
        ax.figure.colorbar(im, orientation = orientation, ax = ax)
    
    if xtick_labels is None: xtick_labels = tick_labels
    if ytick_labels is None: ytick_labels = tick_labels
        
    if xtick_labels is not None and len(xtick_labels) < _tick_label_limit:
        xtick_labels = [str(l) for l in xtick_labels]
        if len(xtick_labels) > 0: ax.set_xticks(np.arange(len(xtick_labels) + 1) - 0.5)
        ax.set_xticklabels(xtick_labels, rotation = 45)
        
    if ytick_labels is not None and len(ytick_labels) < _tick_label_limit:
        if len(ytick_labels) > 0: ax.set_yticks(np.arange(len(ytick_labels) + 1) - 0.5)
        ax.set_yticklabels(ytick_labels)
    
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

def plot_multiple(*args, size = 3, x_size = None, y_size = None, ncols = 2, nrows = None,
                  use_subplots = False, horizontal = False,
                  by = None, corr = None, color_corr = None, color_order = None, # for pd.DataFrame grouping
                  
                  x = None, vlines = None, hlines = None, vlines_kwargs = {}, hlines_kwargs = {},
                  
                  title = None, filename = None, show = False, close = True,
                  ** kwargs
                 ):
    """
        Plot multiple data in a single graph
        Arguments : 
            - args  : data to plot (can be a tuple (name, data))
            - x_size / y_size / nrows / ncols   : information related to build subplots
            - use_subplots  : whether to plot all data in the same plot or not
            - horizontal    : whether to make subplots horizontal aligned or not
            - title / filename / show / close   : same as plot() function
            - by    : the .groupby() argument if a data is a pd.DataFrame in order to see how other columns evolve depending on 'by' column
            - kwargs    : either data to plot or kwargs to pass to each plot() call
        
        Datas can be either : 
            - list / np.ndarray / tf.Tensor : classical data
            - dict  : specific config forthis plot, must contain at least 'x' field representing the data to plot (when calling plot())
            - pd.DataFrame  : plot each column in sepearate plots and can be grouped on 'by'
                Note : pd.DataFrame are only supported in 'args' and not as'kwargs' currently
    """
    datas = []
    for v in args:
        if isinstance(v, tuple) and len(v) == 2:
            datas.append(v)
        elif isinstance(v, dict):
            datas.append((v.pop('name', v.pop('label', None)), v))
        elif isinstance(v, pd.DataFrame):
            use_subplots = True
            if by is not None:
                for value, datas_i in v.groupby(by):
                    datas_i.pop(by)
                    name_i = '{} = {}'.format(by, value)
                    datas.append((name_i, {'x' : datas_i.to_dict('list')}))
            elif corr is not None:
                corr_colors = None
                if color_corr is not None:
                    if color_corr not in v.columns:
                        logging.error('Color correlation color {} is not in data !'.format(color_corr))
                    else:
                        unique_values = list(v[color_corr].unique())
                        corr_colors = [
                            unique_values.index(corr_val_i) for corr_val_i in v[color_corr].values
                        ]
                        if color_order is not None:
                            if len(color_order) < len(unique_values):
                                logging.info('Not enough colors : {} vs {}'.format(len(color_order), len(unique_values)))
                            else:
                                corr_colors = [color_order[color_idx] for color_idx in corr_colors]
                for col in v.columns:
                    if col == corr: continue
                    try:
                        v[col].unique()
                        col_values = v[col].values
                    except TypeError:
                        col_values = [str(val_i) for val_i in v[col].values]
                    datas.append((col, {
                        'x' : col_values, 'y' : v[corr].values, 'ylabel' : corr,
                        'plot_type' : 'scatter'
                    }))
                    if corr_colors is not None:
                        datas[-1][1]['c'] = corr_colors
            else:
                for k, v_i in v.to_dict('list').items():
                    datas.append((k, v_i))
        else:
            datas.append((None, v))
    data_names = [
        k for k, v in kwargs.items()
        if isinstance(v, (list, dict, np.ndarray, tf.Tensor)) or callable(v)
    ]
    datas += [(k, kwargs.pop(k)) for k in data_names]
    
    if len(datas) == 0:
        raise ValueError("Aucune donnée valide à plot !\n  Acceptés : ('list', 'dict', 'np.ndarray')\n  Reçu : \n{}".format(
            '\n'.join([' - {} type : {}'.format(k, v) for k, v in kwargs.items()])
        ))
    
    if size is not None:
        if x_size is None: x_size = size
        if y_size is None: y_size = size
    
    use_subplots = use_subplots or kwargs.get('plot_type', '') == 'imshow'
    if use_subplots:
        if nrows is not None or ncols is not None:
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
        x_size = x_size * ncols
        y_size = y_size * nrows
        figsize = (x_size, y_size)
        
    fig = plt.figure(figsize = figsize)
    
    if title is not None:
        fig.text(
            0.5, 0.99, title, horizontalalignment = 'center', 
            fontsize    = kwargs.get('fontsize', 15), 
            color       = kwargs.get('fontcolor', 'w'), 
            verticalalignment   = 'top'
        )
        
    default_axes_config = {
        'filename'  : None,
        'show'      : False,
        'close'     : False,
        'new_fig'   : False
    }
    
    if x is not None: kwargs['x'] = x
    if vlines is not None: kwargs.update({'vlines' : vlines, 'vlines_kwargs' : vlines_kwargs})
    if hlines is not None: kwargs.update({'hlines' : hlines, 'hlines_kwargs' : hlines_kwargs})

    for i, (name, val) in enumerate(datas):
        ax = fig.add_subplot(nrows, ncols, i+1) if use_subplots else None
        
        if not isinstance(val, dict):
            val = {'x' : val} if 'x' not in kwargs else {'y' : val}

        config_ax = {** kwargs, ** val}

        config_ax['ax'] = ax
        
        if use_subplots:
            config_ax.setdefault('title', name)
        else:
            config_ax['color'] = None
            config_ax.setdefault('label', name)
        config_ax = {** config_ax, ** default_axes_config}
        
        plot(** config_ax)
        
    if filename is not None:
        plt.savefig(
            filename, edgecolor = kwargs.get('fontcolor', 'w'), 
            facecolor = kwargs.get('facecolor', 'black')
        )
        if show: plt.show()
    else:
        plt.show()
    if close: plt.close()
    else: return axes
    
def plot_spectrogram(*args, **kwargs):
    """
        Call plot_multiple() after normalizing spectrograms : making them 2D images and rotate them by 90° as models usually generate [B, T, F] spectrograms but we wat time to be on x-axis
    """
    args = list(args)
    for i, v in enumerate(args):
        if isinstance(v, (np.ndarray, tf.Tensor)):
            if len(v.shape) == 3: v = np.squeeze(v)
            args[i] = np.rot90(v)
    
    for k, v in kwargs.items():
        if isinstance(v, (np.ndarray, tf.Tensor)):
            if len(v.shape) == 3: v = np.squeeze(v)
            kwargs[k] = np.rot90(v)
    
    default_config = {
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
    plot_config = {** default_config, ** kwargs}
    plot_multiple(* args, ** plot_config)
    
def plot_confusion_matrix(cm = None, true = None, pred = None, labels = None,
                          ax = None, figsize = None, factor_size = 0.75, 
                          
                          cmap = 'magma', ticksize = 13, 
                          
                          filename=None, show=True, close = True, new_fig = True, 
                          **kwargs
                         ):
    """
        Plot a confusion matrix 
        Arguments : 
            - cm    : the confusion matrix
            - true / pred   : the true labels and predicted labels
            - labels        : name for each label index
            - ...   : other arguments refers to general plot() arguments
    """
    assert cm is not None or (true is not None and pred is not None)
    
    if cm is None: cm = confusion_matrix(true, pred)
    if labels is None: labels = range(cm.shape[0])
    
    if new_fig and ax is None:
        if figsize is None and len(labels) <= 20:
            figsize = (len(labels) * factor_size, len(labels) * factor_size)

        fig = plt.figure(figsize = figsize)
    
    kwargs.setdefault('title', 'Confusion Matrix')
    kwargs.setdefault('xlabel', 'Predicted label')
    kwargs.setdefault('ylabel', 'True label')
    kwargs.setdefault('with_colorbar', True)
    kwargs.setdefault('tick_labels', labels)
    kwargs['plot_type'] = 'imshow'
    
    ax, im = plot(cm, figsize = figsize, show = False, close = False, ** kwargs)
        
    if len(labels) <= _tick_label_limit:
        cm = np.around(cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis], decimals = 2)
        
        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] < threshold else 'black'
                im.axes.text(j, i, str(cm[i, j]), color = color, fontsize = ticksize, verticalalignment = 'center', horizontalalignment = 'center')
    
    if filename is not None: plt.savefig(filename)
    if show or filename is None: plt.show()
    if close: plt.close()

def plot_embedding(embeddings, ids = None, marker = None, 
                   marker_kwargs = {}, remove_extreme = False, ** kwargs):
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
    
    
    if isinstance(embeddings, pd.DataFrame):
        from utils.embeddings import embeddings_to_np
        
        if 'id' in embeddings and ids is None:
            ids = embeddings['id'].values
        embeddings = embeddings_to_np(embeddings)
    
    reducer = umap.UMAP()
    tsne_embeddings = reducer.fit_transform(embeddings)
    
    x, y = tsne_embeddings[:, 0], tsne_embeddings[:, 1]

    if remove_extreme:
        x, y, marker = filter_x(x, y, marker)
        y, x, marker = filter_x(y, x, marker)
    
    kwargs['x'], kwargs['y'] = x, y
        
    kwargs['plot_type'] = 'scatter'
    if ids is not None:
        unique_ids = np.unique(ids).tolist()
        kwargs['c'] = np.array([unique_ids.index(i) for i in ids])
        size = min(sqrt(len(unique_ids)) * 2, 10)
        kwargs.setdefault('figsize', (size, size))
    
    kwargs.setdefault('title', 'Embedding space')
    kwargs.setdefault('figsize', (10, 10))
    kwargs.setdefault('tick_labels', [])
    kwargs.setdefault('with_grid', True)
    kwargs.setdefault('cmap', 'tab10')
    
    if marker is not None and len(np.unique(marker)) > 1:
        x, y, c = kwargs['x'], kwargs['y'], kwargs['c']
        last_kwargs = {
            'show'      : kwargs.pop('show', True), 
            'filename'  : kwargs.pop('filename', None),
            'close'     : kwargs.pop('close', False)
        }
        kwargs['close'] = False
        kwargs['show'] = False
        ax = kwargs.pop('ax', None)
        unique_marker = np.unique(marker)
        for i, m in enumerate(unique_marker):
            idx = np.where(marker == m)
            kwargs_m = {
                ** kwargs,
                'x' : x[idx], 'y' : y[idx], 'c' : c[idx], 'marker' : m,
                ** marker_kwargs.get(m, {})
            }
            if i == len(unique_marker) - 1: kwargs_m = {** kwargs_m, ** last_kwargs}
            
            ax, _ = plot(ax = ax, ** kwargs_m)
    else:
        return plot(** kwargs)
