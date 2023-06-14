import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Any
from itertools import chain

'''
    Thin wrapper of matplotlib subplot class
'''
class GridCanvas:
    def __init__(self, row, col):
        self.row, self.col = row, col

        self._fig, self._axs = plt.subplots(row, col)
    def __iter__(self):
        if self.row == 1:
            for j in range(self.col):
                yield self._axs[j]
        else:
            for i in range(self.row):
                for j in range(self.col):
                    yield self._axs[i, j]
    @property
    def figure(self):
        return self._fig


def _set_bin_color(patches, cmap):
    for (patch, color) in zip(patches, cmap):
        patch.set_facecolor(color)

def _prepare_canvas(layout):
    row, col = layout
    return GridCanvas(row, col)

def _get_discrete_cmap(n: int, cmap: str='rainbow', alpha=1.):
    '''
    reference: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    '''
    base_cmap = cm.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, n), alpha=alpha)

    return colors

def _flatten_list(arr: List[List[Any]]):
    return list(chain.from_iterable(arr))


def draw_bin_grid(
        X, layout,
        cmap='rainbow',
        alpha=1.,
        title_prefix=None,
    ):
    N = len(np.unique(_flatten_list(X)))
    canvas = _prepare_canvas(layout)
    cmap = _get_discrete_cmap(N, cmap, alpha)
    for i, (x, subplot) in enumerate(zip(X, canvas)):
        _, _, pathces = subplot.hist(x)
        subplot.set(xlim=[0, N-1])
        if title_prefix is not None:
            subplot.set_title(f'{title_prefix} {i}')
        _set_bin_color(pathces, cmap)
    canvas.figure.tight_layout()
    plt.show()
    

