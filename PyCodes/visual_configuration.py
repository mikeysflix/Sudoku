import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import matplotlib.animation as ani

class VisualConfiguration():

    def __init__(self, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, bias='left'):
        """

        """
        super().__init__()
        self._savedir = None
        self.ticksize = ticksize
        self.labelsize = labelsize
        self.textsize = textsize
        self.titlesize = titlesize
        self.headersize = headersize
        self.empty_label = '  '

    @property
    def savedir(self):
        return self._savedir

    @staticmethod
    def select_facecolors(counts, cmap=None, default_color='darkorange'):
        """

        """
        if cmap is None:
            return [default_color]*counts.size
        elif isinstance(cmap, (tuple, list, np.ndarray)):
            nc = len(cmap)
            if nc != counts.size:
                raise ValueError("{} colors for {} bins".format(nc, counts.size))
            return list(cmap)
        else:
            norm = Normalize(vmin=np.min(counts), vmax=np.max(counts))
            f = plt.get_cmap(cmap)
            return f(norm(counts))

    @staticmethod
    def get_number_of_legend_columns(labels):
        """

        """
        if isinstance(labels, int):
            n = labels
        else:
            n = len(labels)
        if n > 2:
            if n % 3 == 0:
                ncol = 3
            else:
                ncol = n // 2
        else:
            ncol = n
        return ncol

    @staticmethod
    def get_empty_handle(ax):
        """

        """
        empty_handle = ax.scatter([np.nan], [np.nan], color='none', alpha=0)
        return empty_handle

    def update_save_directory(self, savedir):
        self._savedir = savedir

    def update_legend_design(self, leg, title=None, textcolor=None, facecolor=None, edgecolor=None, borderaxespad=None):
        """

        """
        if title:
            leg.set_title(title, prop={'size': self.labelsize, 'weight' : 'semibold'})
            if textcolor:
                leg.get_title().set_color(textcolor)
            # leg.get_title().set_ha("center")
        leg._legend_box.align = "center"
        frame = leg.get_frame()
        if facecolor:
            frame.set_facecolor(facecolor)
        if edgecolor:
            frame.set_edgecolor(edgecolor)
        if textcolor:
            for text in leg.get_texts():
                text.set_color(textcolor)
        return leg

    def subview_legend(self, fig, handles, labels, empty_handle=None, title=''):
        """

        """
        if len(labels) == 1:
            ncol = 3
            handles = [empty_handle, handles[0], empty_handle]
            labels = [self.empty_label, labels[0], self.empty_label]
        else:
            ncol = self.get_number_of_legend_columns(labels)
        fig.subplots_adjust(bottom=0.2)
        leg = fig.legend(handles=handles, labels=labels, ncol=ncol, loc='lower center', mode='expand', borderaxespad=0.1, fontsize=self.labelsize)
        leg = self.update_legend_design(leg, title=title, textcolor='darkorange', facecolor='k', edgecolor='steelblue')

    def subview_statistics(self, fig, ax, statistics, loc=None, decimals=2):
        """

        """
        handles, labels = [], []
        empty_handle = self.get_empty_handle(ax)
        for key, value in statistics.items():
            if key == 'mode':
                ...
            else:
                if loc is None:
                    label = r'{}: ${:.2f}$'.format(key, np.round(value, decimals=decimals))
                else:
                    label = r'{}: ${:.2f}$'.format(key, np.round(value[loc], decimals=decimals))
                labels.append(label)
                handles.append(empty_handle)
        self.subview_legend(fig, handles, labels, empty_handle, title='Statistics')

    def display_image(self, fig, savename=None, dpi=800, bbox_inches='tight', pad_inches=0.1, extension='.png', **kwargs):
        """

        """
        if savename is None:
            plt.show()
        elif isinstance(savename, str):
            if self.savedir is None:
                raise ValueError("cannot save plot; self.savedir is None")
            savepath = '{}{}{}'.format(self.savedir, savename, extension)
            fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
        else:
            raise ValueError("invalid type(savename): {}".format(type(savename)))
        plt.close(fig)

    def display_animation(self, anim, fps, savename=None, extension='.mp4', **kwargs):
        if savename is None:
            plt.show()
        elif isinstance(savename, str):
            if self.savedir is None:
                raise ValueError("cannot save plot; self.savedir is None")
            savepath = '{}{}{}'.format(self.savedir, savename, extension)
            anim.save(
                savepath,
                fps=fps,
                **kwargs)
        else:
            raise ValueError("invalid type(savename): {}".format(type(savename)))
        plt.close(fig)

##
