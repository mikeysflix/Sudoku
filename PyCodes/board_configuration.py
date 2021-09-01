import numpy as np
import time

class PresetSudokuBoards():

    def __init__(self):
        super().__init__()
        self.preset_boards = {
            'easy' : np.array([
                [9, 0, 0, 3, 0, 0, 0, 7, 1],
                [4, 3, 7, 8, 0, 0, 2, 5, 0],
                [0, 0, 5, 0, 2, 0, 0, 4, 9],
                [0, 5, 8, 4, 0, 9, 0, 3, 0],
                [7, 0, 0, 1, 0, 0, 0, 9, 8],
                [2, 9, 0, 0, 3, 0, 0, 0, 4],
                [0, 8, 0, 0, 1, 3, 0, 0, 0],
                [3, 0, 4, 6, 8, 7, 0, 0, 0],
                [1, 0, 0, 2, 5, 0, 0, 0, 0]], dtype=int), ## https://sudoku.com/easy/
            'medium' : np.array([
                [0, 7, 0, 0, 0, 5, 1, 0, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 7],
                [0, 1, 3, 7, 2, 6, 8, 0, 0],
                [0, 6, 0, 0, 0, 0, 0, 0, 2],
                [2, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 8, 4, 0, 9, 0, 0],
                [9, 0, 0, 1, 0, 0, 7, 0, 8],
                [0, 0, 8, 0, 3, 0, 0, 4, 0],
                [0, 0, 0, 0, 8, 9, 6, 5, 1]], dtype=int), ## https://sudoku.com/medium/
            'hard' : np.array([
                [0, 0, 0, 0, 0, 0, 0, 3, 0],
                [3, 4, 9, 0, 0, 1, 2, 0, 0],
                [0, 5, 0, 0, 0, 0, 9, 0, 0],
                [0, 2, 0, 5, 0, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 8, 6, 7],
                [6, 0, 0, 0, 0, 0, 0, 0, 4],
                [0, 0, 0, 6, 0, 0, 0, 0, 0],
                [5, 9, 8, 7, 0, 4, 0, 0, 0],
                [7, 0, 0, 2, 0, 0, 0, 0, 8]], dtype=int), ## https://sudoku.com/hard/
            'expert' : np.array([
                [4, 7, 9, 0, 0, 5, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 8],
                [0, 0, 0, 0, 0, 0, 0, 6, 0],
                [3, 4, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 6, 0, 5, 0, 0, 0, 9],
                [8, 0, 0, 0, 0, 0, 0, 0, 6],
                [0, 0, 0, 0, 0, 0, 4, 2, 7],
                [0, 0, 7, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 9, 0, 0, 0, 0]], dtype=int), ## https://sudoku.com/expert/
            'worlds toughest puzzle' : np.array([
                [8, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 6, 0, 0, 0, 0, 0],
                [0, 7, 0, 0, 9, 0, 2, 0, 0],
                [0, 5, 0, 0, 0, 7, 0, 0, 0],
                [0, 0, 0, 0, 4, 5, 7, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 3, 0],
                [0, 0, 1, 0, 0, 0, 0, 6, 8],
                [0, 0, 8, 5, 0, 0, 0, 1, 0],
                [0, 9, 0, 0, 0, 0, 4, 0, 0]], dtype=int)} ## https://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html


class BoardInitialization(PresetSudokuBoards):

    def __init__(self):
        super().__init__()
        self.nrows = 9
        self.ncols = 9
        self.mrows = 3
        self.mcols = 3
        self.nblocks = self.nrows * self.ncols // (self.mrows * self.mcols)
        self.available_characters = np.linspace(1, 9, 9, dtype=int)
        self.missing_character = 0
        self.region_sum = np.sum(self.available_characters)
        self.region_product = np.prod(self.available_characters)
        self._original_board = None
        self._user_board = None
        self._is_solved = False

    @property
    def original_board(self):
        return self._original_board

    @property
    def user_board(self):
        return self._user_board

    @property
    def is_solved(self):
        return self._is_solved

    @staticmethod
    def get_nremaining(board):
        return board.size - np.count_nonzero(board)

    @staticmethod
    def apply_clockwise_rotation(board):
        board = list(zip(*board[::-1]))
        return np.array(board)

    @staticmethod
    def reflect_board(board, horizontal=False, vertical=False):
        # if not any([horizontal, vertical]):
        #     raise ValueError("'horizontal' and/or 'vertical' should be set to True")
        j = -1 if horizontal else 1
        i = -1 if vertical else 1
        return board[::i, ::j]

    def get_board_text(self, board):
        text = ''
        for r, row in enumerate(board):
            t = ''.join(row.astype(str).tolist())
            t = ' '.join(t[i:i+self.mcols] for i in range(0, len(t), self.mcols))
            if r % 3 == 0:
                t = '\n{}'.format(t)
            t = t.replace('0', ' ')
            text += '\n{}'.format(t)
        return text

    def make_board(self):
        values = self.available_characters.copy()
        board = [values.copy()]
        divisors = [self.mrows]
        while True:
            prev_div = divisors[-1]
            curr_div = prev_div + self.mrows
            if curr_div >= self.nrows:
                curr_div = curr_div - self.nrows + 1
            divisors.append(curr_div)
            if len(divisors) == self.nrows - 1:
                break
        indices = np.array(divisors)
        for i in indices:
            row = np.array(values[i:].tolist() + values[:i].tolist())
            board.append(row)
        return np.array(board, dtype=int)

    def accept_input_board(self, board):
        if isinstance(board, str):
            if board in list(self.preset_boards.keys()):
                board = self.preset_boards[board].copy()
            else:
                raise ValueError("invalid board: {}".format(board))
        else:
            if isinstance(board, list):
                board = np.array(board, dtype=int)
            if not isinstance(board, np.ndarray):
                raise ValueError("invalid type(board): {}".format(type(board)))
            if board.shape != (self.nrows, self.ncols):
                raise ValueError("invalid board.shape: {}".format(board.shape))
            if np.any(board > self.nrows):
                raise ValueError("values on board cannot exceed {}".format(self.nrows))
            if np.any(board < 0):
                raise ValueError("values on board cannot be less than zero")
        return board

    def make_missing_entries(self, board, nmissing=None):
        if nmissing is not None:
            if not isinstance(nmissing, int):
                raise ValueError("nmissing must be None or an integer")
            if nmissing < 1:
                raise ValueError("nmissing must be None or at least one")
            if np.any(board == self.missing_character):
                raise ValueError("the board is already missing entries")
            flat_board = board.reshape(-1)
            indices = np.arange(board.size, dtype=int)
            loc = np.random.choice(indices, size=nmissing, replace=False)
            flat_board[loc] = self.missing_character
            board = flat_board.reshape(board.shape)
        return board

    def randomize_character_mapping(self, board):
        characters = self.available_characters.copy()
        repl_chars = np.random.choice(characters, size=characters.size, replace=False)
        mapping = dict(zip(characters, repl_chars))
        if np.any(board == self.missing_character):
            mapping[self.missing_character] = self.missing_character
        for row in range(self.nrows):
            for col in range(self.ncols):
                original_value = board[row, col]
                updated_value = mapping[original_value]
                board[row, col] = updated_value
        return board

    def swap_row_bands(self, board, ri, rj):
        rbi = ri // self.mrows * self.mrows
        rbj = rj // self.mrows * self.mrows
        band_a = board[rbi : rbi + self.mrows, :].copy()
        band_b = board[rbj : rbj + self.mrows, :].copy()
        board[rbi : rbi + self.mrows, :] = band_b
        board[rbj : rbj + self.mrows, :] = band_a
        return board

    def swap_column_bands(self, board, ci, cj):
        cbi = ci // self.mrows * self.mrows
        cbj = cj // self.mrows * self.mrows
        band_a = board[:, cbi : cbi + self.mcols].copy()
        band_b = board[:, cbj : cbj + self.mcols].copy()
        board[:, cbi : cbi + self.mcols] = band_b
        board[:, cbj : cbj + self.mcols] = band_a
        return board

    def get_initial_board(self, board=None, nmissing=None, nmaps=0, nrots=0, reflect_horizontal=False, reflect_vertical=False, row_band_swaps=None, column_band_swaps=None):
        if board is None:
            board = self.make_board()
        else:
            board = self.accept_input_board(
                board=board)
        board = self.make_missing_entries(
            board=board,
            nmissing=nmissing)
        if not isinstance(nmaps, int):
            raise ValueError("invalid type(nmaps): {}; nmaps should be an integer".format(type(nmaps)))
        if nmaps < 0:
            raise ValueError("nmaps must be at least zero")
        if nmaps > 0:
            for i in range(nmaps):
                board = self.randomize_character_mapping(
                    board=board)
        if not isinstance(nrots, int):
            raise ValueError("invalid type(nrots): {}; nrots should be an integer".format(type(nrots)))
        if nrots < 0:
            raise ValueError("nrots must be at least zero")
        if nrots > 0:
            for i in range(nrots):
                board = self.apply_clockwise_rotation(
                    board=board)
        if any([reflect_horizontal, reflect_vertical]):
            board = self.reflect_board(
                board=board,
                horizontal=reflect_horizontal,
                vertical=reflect_vertical)
        if row_band_swaps is not None:
            for ri, rj in zip(*row_band_swaps):
                board = self.swap_row_bands(
                    board=board,
                    ri=ri,
                    rj=rj)
        if column_band_swaps is not None:
            for ci, cj in zip(*column_band_swaps):
                board = self.swap_column_bands(
                    board=board,
                    ci=ci,
                    cj=cj)
        return board

class BoardConfiguration(BoardInitialization):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_row(r, board):
        return board[r, :]

    @staticmethod
    def get_column(c, board):
        return board[:, c]

    def get_block(self, r, c, board):
        i = r // self.mrows * self.mrows
        j = c // self.mcols * self.mcols
        return board[i:i+3, j:j+3]

    def get_block_number(self, r, c):
        b = (r // self.mrows) * self.mrows + (c // self.mcols)
        return b

    def get_all_blocks(self, board):
        result = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if (r % self.mrows == 0) and (c % self.mcols == 0):
                    block = self.get_block(
                        r=r,
                        c=c,
                        board=board)
                    result.append(block)
        return np.array(result)

    def get_missing_locations(self):
        rows, cols = np.where(self.user_board == self.missing_character)
        return rows, cols

    def get_cell_candidates(self, values):
        candidates = np.setdiff1d(self.available_characters, values[values != self.missing_character])
        return np.sort(candidates)

    def get_mutual_cell_candidates(self, row_candidates, column_candidates, block_candidates):
        mutual_candidates = set(row_candidates) & set(column_candidates) & set(block_candidates)
        return np.sort(list(mutual_candidates))

    def validate_region(self, region):
        if np.all(np.diff([len(set(region.tolist())), region.size, self.available_characters.size]) != 0):
            raise ValueError("invalid region is not unique:\n{}".format(region))
        if not np.all(region > 0):
            raise ValueError("invalid region contains values less than or equal to zero:\n{}".format(region))
        if not np.all(region <= self.available_characters[-1]):
            raise ValueError("invalid region contains values greater than {}:\n{}".format(self.available_characters[-1], region))

    def validate_solved_board(self):
        for r in range(self.nrows):
            row = self.get_row(
                r=r,
                board=self.user_board)
            self.validate_region(row)
        for c in range(self.ncols):
            column = self.get_column(
                c=c,
                board=self.user_board)
            self.validate_region(column)
        for r in range(self.nrows):
            for c in range(self.ncols):
                block = self.get_block(
                    r=r,
                    c=c,
                    board=self.user_board)
                self.validate_region(block.reshape(-1))
        self._is_solved = True

    def generate_board(self, board=None, nmissing=None, nmaps=0, nrots=0, reflect_horizontal=False, reflect_vertical=False, row_band_swaps=None, column_band_swaps=None):
        board = self.get_initial_board(
            board=board,
            nmissing=nmissing,
            nmaps=nmaps,
            nrots=nrots,
            reflect_horizontal=reflect_horizontal,
            reflect_vertical=reflect_vertical,
            row_band_swaps=row_band_swaps,
            column_band_swaps=column_band_swaps)
        self._original_board = board.copy()
        self._user_board = board.copy()
        if not np.any(board == self.missing_character):
            self.validate_solved_board()
        del board

    def reset_user_board(self):
        self._user_board = np.copy(self.original_board)

    def reset_save_state(self):
        self._is_solved = False

    def show_original_board_text(self):
        s = self.get_board_text(self.original_board)
        print("\n .. ORIGINAL BOARD:\n{}\n".format(s))

    def show_user_board_text(self):
        s = self.get_board_text(self.user_board)
        print("\n .. USER BOARD:\n{}\n".format(s))

class SolverConfiguration(BoardConfiguration):

    def __init__(self):
        super().__init__()
        self._elapsed = None

    @property
    def elapsed(self):
        return self._elapsed



##
