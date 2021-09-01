from board_configuration import *

class RecursiveSolver(SolverConfiguration):

    def __init__(self):
        super().__init__()
        self.solver_method = 'recursive solver with back-tracking'
        self._cell_selection_method = None
        self._select_cell = None
        self._select_backtracked_cell_replacement = None
        self._modified_locations = []
        self._nbacks = 0
        self._nremaining = None
        self._niter = 0

    def __repr__(self):
        return "%r()" % self.__class__.__name__

    def __str__(self):
        s = "\n .. SOLVER METHOD:\n\t{}\n".format(self.solver_method)
        s += "\n .. IS SOLVED:\n\t{}\n".format(self.is_solved)
        s += "\n .. CELL SELECTION METHOD:\n\t{}\n".format(self.cell_selection_method)
        s += "\n .. NUMBER OF CELLS REMAINING:\n\t{}\n".format(self.nremaining)
        s += "\n .. NUMBER OF BACKTRACKS:\n\t{}\n".format(self.nbacks)
        s += "\n .. NUMBER OF ITERATIONS:\n\t{}\n".format(self.niter)
        s += "\n .. ELAPSED TIME:\n\t{}\n".format(self.elapsed)
        return s

    @property
    def cell_selection_method(self):
        return self._cell_selection_method

    @property
    def select_cell(self):
        return self._select_cell

    @property
    def select_backtracked_cell_replacement(self):
        return self._select_backtracked_cell_replacement

    @property
    def modified_locations(self):
        return self._modified_locations

    @property
    def nbacks(self):
        return self._nbacks

    @property
    def nremaining(self):
        return self._nremaining

    @property
    def niter(self):
        return self._niter

    @staticmethod
    def select_monotonically_increasing_candidates(prev_candidates, prev_cell_value):
        updt_candidates = prev_candidates[prev_candidates > prev_cell_value]
        return updt_candidates

    @staticmethod
    def select_non_monotonically_increasing_candidates(prev_candidates, prev_cell_value):
        updt_candidates = prev_candidates[prev_candidates != prev_cell_value]
        return updt_candidates

    def update_cell_selection_method(self, cell_selection_method):
        if cell_selection_method == 'naive':
            self._select_cell = lambda *args, **kwargs : self.select_cell_naively(*args, **kwargs)
            self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_monotonically_increasing_candidates(*args, **kwargs)
        elif cell_selection_method == 'random':
            self._select_cell = lambda *args, **kwargs : self.select_cell_randomly(*args, **kwargs)
            self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_non_monotonically_increasing_candidates(*args, **kwargs)
            # self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_monotonically_increasing_candidates(*args, **kwargs)
        elif cell_selection_method == 'probabilistic':
            self._select_cell = lambda *args, **kwargs : self.select_cell_probabilistically(*args, **kwargs)
            self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_non_monotonically_increasing_candidates(*args, **kwargs)
        elif cell_selection_method == 'inverse frequency':
            self._select_cell = lambda *args, **kwargs : self.select_cell_inversely(*args, **kwargs)
            self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_non_monotonically_increasing_candidates(*args, **kwargs)
        elif cell_selection_method == 'adaptive':
            self._select_cell = lambda *args, **kwargs : self.select_cell_adaptively(*args, **kwargs)
            self._select_backtracked_cell_replacement = lambda *args, **kwargs : self.select_non_monotonically_increasing_candidates(*args, **kwargs)
        else:
            raise ValueError("invalid cell_selection_method: {}".format(cell_selection_method))
        self._cell_selection_method = cell_selection_method

    def select_cell_naively(self, missing_rows, missing_cols, k=0):
        r, c = missing_rows[k], missing_cols[k]
        row = self.get_row(
            r=r,
            board=self.user_board)
        column = self.get_column(
            c=c,
            board=self.user_board)
        block = self.get_block(
            r=r,
            c=c,
            board=self.user_board)
        row_candidates = self.get_cell_candidates(row)
        column_candidates = self.get_cell_candidates(column)
        block_candidates = self.get_cell_candidates(block.reshape(-1))
        mutual_candidates = self.get_mutual_cell_candidates(row_candidates, column_candidates, block_candidates)
        return r, c, mutual_candidates

    def select_cell_randomly(self, missing_rows, missing_cols):
        k = np.random.randint(low=0, high=missing_rows.size, size=None)
        r, c, candidates = self.select_cell_naively(
            missing_rows=missing_rows,
            missing_cols=missing_cols,
            k=k)
        return r, c, candidates

    def select_cell_probabilistically(self, missing_rows, missing_cols):
        r, c, candidates = None, None, None
        freq = int(self.nrows * self.ncols)
        for ri, cj in zip(missing_rows, missing_cols):
            row = self.get_row(
                r=ri,
                board=self.user_board)
            column = self.get_column(
                c=cj,
                board=self.user_board)
            block = self.get_block(
                r=ri,
                c=cj,
                board=self.user_board)
            row_candidates = self.get_cell_candidates(row)
            column_candidates = self.get_cell_candidates(column)
            block_candidates = self.get_cell_candidates(block.reshape(-1))
            mutual_candidates = self.get_mutual_cell_candidates(row_candidates, column_candidates, block_candidates)
            if mutual_candidates.size == 1:
                r, c = ri, cj
                candidates = mutual_candidates.copy()
                break
            else:
                if mutual_candidates.size < freq:
                    r, c = ri, cj
                    candidates = mutual_candidates.copy()
                    freq = candidates.size
        return r, c, candidates

    def select_cell_inversely(self, missing_rows, missing_cols):
        r, c, candidates = None, None, None
        smallest_weight = int(self.nrows * self.ncols)
        unq_elements, unq_counts = np.unique(self.user_board[self.user_board != self.missing_character], return_counts=True)
        ordering = np.argsort(unq_counts)
        choices = unq_elements[ordering]
        inv_wts = unq_counts[ordering]
        character_to_inverse_weight_mapping = dict(zip(choices, inv_wts))
        for ri, cj in zip(missing_rows, missing_cols):
            row = self.get_row(
                r=ri,
                board=self.user_board)
            column = self.get_column(
                c=cj,
                board=self.user_board)
            block = self.get_block(
                r=ri,
                c=cj,
                board=self.user_board)
            row_candidates = self.get_cell_candidates(row)
            column_candidates = self.get_cell_candidates(column)
            block_candidates = self.get_cell_candidates(block.reshape(-1))
            mutual_candidates = self.get_mutual_cell_candidates(row_candidates, column_candidates, block_candidates)
            curr_weight = np.sum([character_to_inverse_weight_mapping[candidate] for candidate in mutual_candidates])
            if curr_weight < smallest_weight:
                r, c = ri, cj
                candidates = mutual_candidates.copy()
                smallest_weight = curr_weight
        return r, c, candidates

    def select_cell_adaptively(self, missing_rows, missing_cols):
        prob_r, prob_c, prob_candidates = self.select_cell_probabilistically(missing_rows, missing_cols)
        if prob_candidates.size == 1:
            return prob_r, prob_c, prob_candidates
        else:
            dz = self.get_nremaining(self.original_board)
            if (dz == self.nremaining) and (self.niter > 0):
                random_r, random_c, random_candidates = self.select_cell_randomly(missing_rows, missing_cols)
            else:
                if self.nremaining > self.user_board.size / 2.12:
                    inv_r, inv_c, inv_candidates = self.select_cell_inversely(missing_rows, missing_cols)
                    return inv_r, inv_c, inv_candidates
                else:
                    return prob_r, prob_c, prob_candidates

    def update_cell(self, r, c, value):
        self._user_board[r, c] = value
        self._modified_locations.append((r, c))
        self._nremaining -= 1
        self._niter += 1

    def back_track(self):
        if len(self.modified_locations):
            self._nbacks += 1
            self._nremaining += 1
            self._niter += 1
            (r, c) = self._modified_locations.pop()
            prev_row = self.get_row(
                r=r,
                board=self.user_board)
            prev_column = self.get_column(
                c=c,
                board=self.user_board)
            prev_block = self.get_block(
                r=r,
                c=c,
                board=self.user_board)
            prev_cell_value = self.user_board[r, c]
            row_candidates = self.get_cell_candidates(prev_row)
            column_candidates = self.get_cell_candidates(prev_column)
            block_candidates = self.get_cell_candidates(prev_block.reshape(-1))
            prev_candidates = self.get_mutual_cell_candidates(row_candidates, column_candidates, block_candidates)
            updt_candidates = self.select_backtracked_cell_replacement(prev_candidates, prev_cell_value)
            if updt_candidates.size:
                self.update_cell(r, c, updt_candidates[0])
            else:
                self._user_board[r, c] = self.missing_character
                self.back_track()
        else:
            if np.array_equal(self.user_board, self.original_board):
                raise ValueError("cannot back-track from original board")
            else:
                raise ValueError("debug me")

    def update_board(self):
        missing_rows, missing_cols = self.get_missing_locations()
        if missing_rows.size:
            r, c, candidates = self.select_cell(missing_rows, missing_cols)
            if candidates.size:
                self.update_cell(r, c, candidates[0])
            else:
                self.back_track()
            self.update_board()
        else:
            self.validate_solved_board()

    def solve(self, suppress_error=False):
        start = time.process_time()
        self._nremaining = self.get_nremaining(self.user_board)
        try:
            self.update_board()
        except RuntimeError as err:
            if 'maximum recursion depth exceeded' in err.args[0]:
                if suppress_error:
                    print(err)
                else:
                    raise ValueError(err)
            else:
                raise ValueError("debug me")
        self._elapsed = time.process_time() - start

##
