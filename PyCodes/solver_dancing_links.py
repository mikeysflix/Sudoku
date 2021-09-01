import itertools
from board_configuration import *

class DancingLinksExactCoverSolver(SolverConfiguration):

    def __init__(self):
        super().__init__()
        self.solver_method = 'dancing links with exact cover'
        self._choices = dict()
        self._constraints = dict()
        self._incidence_matrix = None
        self._solutions = []

    def __repr__(self):
        return "%r()" % self.__class__.__name__

    def __str__(self):
        s = "\n .. SOLVER METHOD:\n\t{}\n".format(self.solver_method)
        s += "\n .. IS SOLVED:\n\t{}\n".format(self.is_solved)
        s += "\n .. ELAPSED TIME:\n\t{}\n".format(self.elapsed)
        return s

    @property
    def choices(self):
        return self._choices

    @property
    def constraints(self):
        return self._constraints

    @property
    def incidence_matrix(self):
        return self._incidence_matrix

    @property
    def solutions(self):
        return self._solutions

    def initialize_choices(self):
        indices = self.available_characters - 1
        for r, c, cell_value in itertools.product(indices, indices, self.available_characters):
            b = self.get_block_number(r, c)
            key = (r, c, cell_value)
            self._choices[key] = [
                ("r+c", (r, c)),
                ("r", (r, cell_value)),
                ("c", (c, cell_value)),
                ("b", (b, cell_value))]

    def initialize_constraints(self):
        indices = self.available_characters - 1
        for r, c in itertools.product(indices, indices):
            cell_value = c + 1
            self._constraints[("r+c", (r, c))] = set()
            self._constraints[("r", (r, cell_value))] = set()
            self._constraints[("c", (r, cell_value))] = set()
            self._constraints[("b", (r, cell_value))] = set()
        for constraint_id, constraints in self.choices.items():
            for constraint in constraints:
                self._constraints[constraint].add(constraint_id)

    def initialize_incidence_matrix(self):
        shape = (len(self.choices), len(self.constraints))
        mat = np.zeros(shape, dtype=int)
        # for _ in ...:
        #     mat[...] = 1
        self._incidence_matrix = mat

    def solve(self, suppress_error=False):
        start = time.process_time()
        self.initialize_choices()
        self.initialize_constraints()
        self.initialize_incidence_matrix()

        self.validate_solved_board()
        self._elapsed = time.process_time() - start

# # board = np.array([
# #     [0, 4, 5, 3, 8, 9, 6, 1, 2],
# #     [3, 0, 9, 6, 1, 2, 7, 4, 5],
# #     [6, 1, 2, 7, 4, 5, 3, 8, 9],
# #     [4, 5, 3, 8, 9, 6, 1, 2, 7],
# #     [8, 0, 0, 1, 2, 7, 4, 5, 3],
# #     [1, 2, 7, 4, 5, 3, 8, 9, 6],
# #     [5, 3, 8, 9, 6, 1, 2, 7, 4],
# #     [9, 6, 1, 2, 7, 4, 5, 3, 8],
# #     [2, 7, 4, 5, 3, 8, 9, 6, 1]])
#
# board = np.array([
#     [0, 4, 5, 3, 8, 9, 6, 1, 2],
#     [3, 0, 9, 6, 1, 2, 7, 4, 5],
#     [6, 1, 2, 7, 4, 5, 3, 8, 9],
#     [4, 5, 3, 8, 9, 6, 1, 2, 7],
#     [8, 0, 0, 1, 2, 7, 4, 5, 3],
#     [1, 2, 7, 4, 5, 3, 8, 9, 6],
#     [5, 3, 8, 9, 6, 1, 2, 7, 4],
#     [9, 6, 1, 2, 7, 4, 5, 3, 8],
#     [2, 7, 4, 5, 3, 8, 9, 6, 1]])
#
# solver = DancingLinksExactCoverSolver()
# solver._user_board = board.copy()
# solver._original_board = board.copy()
# solver.solve()

##






##
