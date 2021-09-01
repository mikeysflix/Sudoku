from board_configuration import *
from visual_configuration import *

class VisualGraphNetwork(BoardConfiguration):

    def __init__(self, solved_board, savedir=None):
        super().__init__()
        self.solved_board = solved_board
        self.visual_configuration = VisualConfiguration()
        self.visual_configuration.update_save_directory(
            savedir=savedir)

    @staticmethod
    def initialize_plot(ndim, **kwargs):
        if ndim == 2:
            fig, ax = plt.subplots(**kwargs)
        elif ndim == 3:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            raise ValueError("invalid ndim: {}".format(ndim))
        ax.set_axis_off()
        return fig, ax

    @staticmethod
    def get_pre_animation_objects(ax):
        lines, x_line, y_line = [], [], []
        scats, scat_offsets = [], []
        for line in ax.lines:
            lines.append(line)
            x_line.append(line.get_xdata())
            y_line.append(line.get_ydata())
        for scat in ax.collections:
            scats.append(scat)
            scat_offsets.append(scat.get_offsets())
        return lines, x_line, y_line, scats, scat_offsets

    @staticmethod
    def animate(i, ax, lines, x_line, y_line, scats, scat_offsets):
        if 0 < i < 720:
            ax.elev += 1
        if 360 < i < 1080:
            ax.azim += 1
        for line, xl, yl in zip(lines, x_line, y_line):
            line_coordinates = [x_line, y_line]
            line.set_data(
                *line_coordinates)
        for scat, offset in zip(scats, scat_offsets):
            scat.set_offsets(
                offset)
        return lines, scats

    def get_nodes_colormap(self, vertex_facecolor=None, vertex_cmap=None):
        if (vertex_facecolor is None) and (vertex_cmap is None):
            return None
        else:
            if (vertex_facecolor is not None) and (vertex_cmap is not None):
                raise ValueError("input either vertex_facecolor or vertex_cmap; not both")
            node_facecolors = self.visual_configuration.select_facecolors(
                counts=self.available_characters,
                cmap=vertex_cmap,
                default_color=vertex_facecolor)
            node_to_color = dict(zip(self.available_characters, node_facecolors))
            return node_to_color

    def get_row_constraints_colormap(self, row_facecolor=None, row_cmap=None):
        if (row_facecolor is None) and (row_cmap is None):
            return None
        else:
            if (row_facecolor is not None) and (row_cmap is not None):
                raise ValueError("input either row_facecolor or row_cmap")
            row_numbers = np.arange(self.nrows, dtype=int)
            if row_cmap is None:
                facecolors = [row_facecolor for row_number in row_numbers]
            else:
                facecolors = self.visual_configuration.select_facecolors(
                    counts=row_numbers,
                    cmap=row_cmap,
                    default_color=row_facecolor)
            row_to_color = dict(zip(row_numbers, facecolors))
            return row_to_color

    def get_column_constraints_colormap(self, column_facecolor=None, column_cmap=None):
        if (column_facecolor is None) and (column_cmap is None):
            return None
        else:
            if (column_facecolor is not None) and (column_cmap is not None):
                raise ValueError("input either column_facecolor or column_cmap")
            column_numbers = np.arange(self.ncols, dtype=int)
            if column_cmap is None:
                facecolors = [column_facecolor for column_number in column_numbers]
            else:
                facecolors = self.visual_configuration.select_facecolors(
                    counts=column_numbers,
                    cmap=column_cmap,
                    default_color=column_facecolor)
            column_to_color = dict(zip(column_numbers, facecolors))
            return column_to_color

    def get_block_constraints_colormap(self, block_facecolor=None, block_cmap=None):
        if (block_facecolor is None) and (block_facecolor is None):
            return None
        else:
            if (block_facecolor is not None) and (block_cmap is not None):
                raise ValueError("input either column_facecolor or column_cmap")
            block_numbers = np.arange(self.mrows * self.mcols, dtype=int)
            if block_cmap is None:
                facecolors = [block_facecolor for block_number in block_numbers]
            else:
                facecolors = self.visual_configuration.select_facecolors(
                    counts=block_numbers,
                    cmap=block_cmap,
                    default_color=block_facecolor)
            block_to_color = dict(zip(block_numbers, facecolors))
            return block_to_color

    def initialize_graph(self, ax, ndim, node_to_color=None, row_to_color=None, column_to_color=None, block_to_color=None, show_node_labels=False):
        ax.set_title(
            'Sudoku Network Diagram',
            fontsize=self.visual_configuration.titlesize)
        for ci, i_column in enumerate(self.solved_board.T):
            for ri, i_value in enumerate(i_column[::-1]):
                node_coordinates = (ci, ri) if ndim == 2 else (ci, ri, 0) # (9-ri, 9-ci)
                ## plot nodes
                if node_to_color is not None:
                    ax.scatter(
                        *node_coordinates,
                        color=node_to_color[i_value],
                        marker='.',
                        alpha=0.85)
                ## label nodes
                if show_node_labels:
                    ax.text(
                        *node_coordinates,
                        '{}'.format(i_value),
                        fontsize=self.visual_configuration.labelsize,
                        horizontalalignment='center',
                        verticalalignment='center',
                        alpha=0.7)
                for cj, j_column in enumerate(self.solved_board.T):
                    for rj, j_value in enumerate(j_column[::-1]):
                        if row_to_color is not None:
                            if ri == rj:
                                xrow = [ci, cj]
                                yrow = [ri, rj]
                                row_coordinates = (xrow, yrow) if ndim == 2 else (xrow, yrow, (0, 0))
                                ax.plot(
                                    *row_coordinates,
                                    color=row_to_color[ri],
                                    alpha=0.7,
                                    linestyle='-')
                        if column_to_color is not None:
                            if ci == cj:
                                xcolumn = [ci, cj]
                                ycolumn = [ri, rj]
                                column_coordinates = (xcolumn, ycolumn) if ndim == 2 else (xcolumn, ycolumn, (0, 0))
                                ax.plot(
                                    *column_coordinates,
                                    color=column_to_color[ci],
                                    alpha=0.7,
                                    linestyle='-')
                        if block_to_color is not None:
                            bi = self.get_block_number(ri, ci)
                            bj = self.get_block_number(rj, cj)
                            # bi = (ri // self.mrows) * self.mrows + (ci // self.mcols)
                            # bj = (rj // self.mrows) * self.mrows + (cj // self.mcols)
                            if (bi == bj) and (i_value != j_value):
                                xblock = [ci, cj]
                                yblock = [ri, rj]
                                block_coordinates = (xblock, yblock) if ndim == 2 else (xblock, yblock, (0, 0))
                                ax.plot(
                                    *block_coordinates,
                                    color=block_to_color[bi],
                                    alpha=0.7,
                                    linestyle='-')

    def view_animated_network_graph(self, fps=None, vertex_facecolor=None, vertex_cmap=None, row_facecolor=None, row_cmap=None, column_facecolor=None, column_cmap=None, block_facecolor=None, block_cmap=None, show_node_labels=False, save=False, **kwargs):
        ndim = 3
        fig, ax = self.initialize_plot(
            ndim=ndim,
            **kwargs)
        node_to_color = self.get_nodes_colormap(
            vertex_facecolor=vertex_facecolor,
            vertex_cmap=vertex_cmap)
        row_to_color = self.get_row_constraints_colormap(
            row_facecolor=row_facecolor,
            row_cmap=row_cmap)
        column_to_color = self.get_column_constraints_colormap(
            column_facecolor=column_facecolor,
            column_cmap=column_cmap)
        block_to_color = self.get_block_constraints_colormap(
            block_facecolor=block_facecolor,
            block_cmap=block_cmap)
        self.initialize_graph(
            ax=ax,
            ndim=ndim,
            node_to_color=node_to_color,
            row_to_color=row_to_color,
            column_to_color=column_to_color,
            block_to_color=block_to_color,
            show_node_labels=False)
        lines, x_line, y_line, scats, scat_offsets = self.get_pre_animation_objects(
            ax=ax)
        anim = ani.FuncAnimation(
            fig,
            self.animate,
            fargs=(ax, lines, x_line, y_line, scats, scat_offsets),
            # init_func=init,
            frames=1200,
            interval=5,
            blit=False)
        savename = '3d_network_diagram_sudoku' if save else None
        self.visual_configuration.display_animation(
            anim=anim,
            savename=savename,
            fps=fps,
            extra_args=['-vcodec', 'libx264'])

    def view_network_graph(self, ndim=2, vertex_facecolor=None, vertex_cmap=None, row_facecolor=None, row_cmap=None, column_facecolor=None, column_cmap=None, block_facecolor=None, block_cmap=None, show_node_labels=False, save=False, **kwargs):
        fig, ax = self.initialize_plot(
            ndim=ndim,
            **kwargs)
        node_to_color = self.get_nodes_colormap(
            vertex_facecolor=vertex_facecolor,
            vertex_cmap=vertex_cmap)
        row_to_color = self.get_row_constraints_colormap(
            row_facecolor=row_facecolor,
            row_cmap=row_cmap)
        column_to_color = self.get_column_constraints_colormap(
            column_facecolor=column_facecolor,
            column_cmap=column_cmap)
        block_to_color = self.get_block_constraints_colormap(
            block_facecolor=block_facecolor,
            block_cmap=block_cmap)
        self.initialize_graph(
            ax=ax,
            ndim=ndim,
            node_to_color=node_to_color,
            row_to_color=row_to_color,
            column_to_color=column_to_color,
            block_to_color=block_to_color,
            show_node_labels=show_node_labels)
        savename = '{}d_network_diagram_sudoku'.format(ndim) if save else None
        self.visual_configuration.display_image(fig, savename)

#
# solved_board = np.array([
#     [4, 8, 9, 1, 5, 3, 2, 6, 7],
#     [1, 5, 3, 2, 6, 7, 4, 8, 9],
#     [2, 6, 7, 4, 8, 9, 1, 5, 3],
#     [8, 9, 1, 5, 3, 2, 6, 7, 4],
#     [5, 3, 2, 6, 7, 4, 8, 9, 1],
#     [6, 7, 4, 8, 9, 1, 5, 3, 2],
#     [9, 1, 5, 3, 2, 6, 7, 4, 8],
#     [3, 2, 6, 7, 4, 8, 9, 1, 5],
#     [7, 4, 8, 9, 1, 5, 3, 2, 6]])
#
# VGN = VisualGraphNetwork(
#     solved_board=solved_board,
#     savedir='/Users/mikeyshmikey/Desktop/Hub/Programming/Sudoku/Figures/')
#
# VGN.view_network_graph(
#     ndim=2,
#     show_node_labels=True,
#     row_facecolor='darkorange')
# VGN.view_network_graph(
#     ndim=2,
#     vertex_cmap='jet',
#     column_facecolor='k')
# VGN.view_network_graph(
#     ndim=3,
#     row_cmap='jet')
# VGN.view_network_graph(
#     ndim=3,
#     vertex_cmap='jet',
#     show_node_labels=True)
# VGN.view_network_graph(
#     ndim=3,
#     vertex_facecolor='silver',
#     row_cmap='Oranges',
#     column_cmap='Blues',
#     block_cmap='Greens',
#     # row_facecolor='steelblue',
#     # column_facecolor='darkgreen',
#     # block_facecolor='darkorange',
#     show_node_labels=True)
#
# # VGN.view_animated_network_graph(
# #     fps=30,
# #     vertex_facecolor='silver',
# #     row_facecolor='darkorange',
# #     column_facecolor='steelblue',
# #     block_facecolor='darkgreen',
# #     save=True)

##
