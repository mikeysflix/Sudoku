from collections import OrderedDict
from solver_recursive_backtracking import *
from solver_constrained_system_equations import *
from solver_stochastic import *
from solver_dancing_links import *
from graph_network import *

class SudokuBackEnd(BoardConfiguration):

    def __init__(self):
        super().__init__()
        self._difficulty_level = None
        self._solved_board = None
        self._solvers = []

    @property
    def difficulty_level(self):
        return self._difficulty_level

    @property
    def solved_board(self):
        return self._solved_board

    @property
    def solvers(self):
        return self._solvers

    def verify_number_of_clues(self):
        n = np.sum(self.user_board != self.missing_character)
        if n <= 16:
            ## https://arxiv.org/pdf/1201.0749.pdf
            raise ValueError("invalid sudoku; board contains more than one solution")

    def assign_difficulty_level(self):
        self.add_solver(
            solver=RecursiveSolver(),
            cell_selection_method='adaptive')
        solver = self._solvers.pop()
        solver.solve(
            suppress_error=True)
        if solver.is_solved:
            level_ids = ('easy', 'medium', 'hard')
            level_fences = [0.0425, 0.1] # 0.04 # 0.08 # 0.09275]
            curr_level = np.searchsorted(level_fences, solver.elapsed)
            self._difficulty_level = level_ids[curr_level]
        else:
            self._difficulty_level = 'expert'

    def add_solver(self, solver, *args, **kwargs):
        solver._original_board = self.original_board.copy()
        solver._user_board = self.user_board.copy()
        if solver.solver_method == 'recursive solver with back-tracking':
            solver.update_cell_selection_method(*args, **kwargs)
        elif solver.solver_method == 'system of equations solver with constraints':
            solver.initialize()
            solver.update_constraint_equations(*args, **kwargs)
        elif solver.solver_method == 'stochastic solver':
            solver.initialize_prerequisites()
            solver.select_stochastic_method_and_parameters(*args, **kwargs)
        elif solver.solver_method == 'dancing links with exact cover':
            ...
            # solver.initialize_choices()
            # solver.initialize_constraints()
        else:
            raise ValueError("invalid solver.solver_method")
        self._solvers.append(solver)

    def update_solved_board(self): #, solver=None):
        self.add_solver(
            solver=DancingLinksExactCoverSolver())
        solver = self._solvers.pop()
        solver.solve()
        self._solved_board = solver.user_board.copy()
        # if solver is None:
        #     self.add_solver(
        #         solver=DancingLinksExactCoverSolver())
        #     solver = self._solvers.pop()
        #     solver.solve()
        #     self._solved_board = solver.user_board.copy()
        # else:
        #     if solver.is_solved:
        #         self._solved_board = solver.user_board.copy()
        #     else:
        #         self.update_solved_board(
        #             solver=None)



    def solve(self, suppress_error=False, solver_index=None):
        # print("\n .. DIFFICULTY LEVEL:\n\t{}\n".format(self.difficulty_level))
        if solver_index is not None:
            solver = self.solvers[solver_index]
            solver.solve(suppress_error=suppress_error)
            solver.show_user_board_text()
            print(solver)
        else:
            if len(self.solvers) == 0:
                raise ValueError("no solvers are initialized")
            for i in range(len(self.solvers)):
                self.solve(
                    suppress_error=suppress_error,
                    solver_index=i)















##
