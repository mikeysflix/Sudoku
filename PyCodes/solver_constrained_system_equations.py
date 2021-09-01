import sympy
from board_configuration import *

class CoefficientMatrixMapping(SolverConfiguration):

    def __init__(self):
        super().__init__()
        self._additive_coefficient_matrix = None
        self._multiplicative_coefficient_matrix = None
        self._coefficients_by_row = []
        self._coefficients_by_column = []
        self._coefficients_by_block = []

    @property
    def additive_coefficient_matrix(self):
        return self._additive_coefficient_matrix

    @property
    def multiplicative_coefficient_matrix(self):
        return self._multiplicative_coefficient_matrix

    @property
    def coefficients_by_row(self):
        return self._coefficients_by_row

    @property
    def coefficients_by_column(self):
        return self._coefficients_by_column

    @property
    def coefficients_by_block(self):
        return self._coefficients_by_block

    def update_additive_coefficient_matrix(self):
        missing_loc = (self.user_board == self.missing_character)
        nmissing = np.sum(missing_loc)
        self._additive_coefficient_matrix = np.zeros(self.user_board.shape, dtype=int)
        self._additive_coefficient_matrix[missing_loc] = np.arange(1, nmissing+1, dtype=int)

    def update_coefficients_by_row(self):
        for r, row in enumerate(self.additive_coefficient_matrix):
            loc = (row != self.missing_character)
            if np.any(loc):
                indices = row[loc] - 1
                self._coefficients_by_row.append(indices.tolist())
            else:
                self._coefficients_by_row.append([])

    def update_coefficients_by_column(self):
        for c, column in enumerate(self.additive_coefficient_matrix.T):
            loc = (column != self.missing_character)
            if np.any(loc):
                indices = column[loc] - 1
                self._coefficients_by_column.append(indices.tolist())
            else:
                self._coefficients_by_column.append([])

    def update_coefficients_by_block(self):
        blocks = self.get_all_blocks(board=self.additive_coefficient_matrix)
        for _block in blocks:
            block = _block.reshape(-1)
            loc = (block != self.missing_character)
            if np.any(loc):
                indices = block[loc] - 1
                self._coefficients_by_block.append(indices.tolist())
            else:
                self._coefficients_by_block.append([])

    def update_multiplicative_coefficient_matrix(self):
        mask = np.copy(self.user_board)
        mask[mask == self.missing_character] = 1
        self._multiplicative_coefficient_matrix = mask

class ConstraintMapping(CoefficientMatrixMapping):

    def __init__(self):
        super().__init__()
        self._variable_parameters = None
        self._constraint_equations = {
            'row' : {
                'additive' : {},
                'multiplicative' : {}},
            'column' : {
                'additive' : {},
                'multiplicative' : {}},
            'block' : {
                'additive' : {},
                'multiplicative' : {}}}

    @property
    def variable_parameters(self):
        return self._variable_parameters

    @property
    def constraint_equations(self):
        return self._constraint_equations

    def update_variable_parameters(self):
        nz = self.get_nremaining(self.user_board)
        s_params = ', '.join(('x_{}'.format(i) for i in range(nz)))
        self._variable_parameters = sympy.symbols(s_params, integer=True)

    def get_constraints(self, d, result=None):
        if result is None:
            result = []
        if isinstance(d, dict):
            for k, v in d.items():
                result = self.get_constraints(v, result=result)
        else:
            result.extend(d)
        return result

    def update_additive_row_constraints(self, exponent):
        equations = []
        if exponent == 1:
            constraints = self.region_sum - np.sum(self.user_board, axis=1)
        else:
            constraints = np.sum(np.power(self.available_characters, exponent)) - np.sum(np.power(self.user_board, exponent), axis=1)
        for r, constraint in enumerate(constraints):
            indices = self.coefficients_by_row[r]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.sum(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['row']['additive'][exponent] = equations

    def update_multiplicative_row_constraints(self, exponent):
        equations = []
        constraints = np.power(self.region_product, exponent) // np.prod(np.power(self.multiplicative_coefficient_matrix, exponent), axis=1)
        for r, constraint in enumerate(constraints):
            indices = self.coefficients_by_row[r]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.prod(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['row']['multiplicative'][exponent] = equations

    def update_additive_column_constraints(self, exponent):
        equations = []
        if exponent == 1:
            constraints = self.region_sum - np.sum(self.user_board, axis=0)
        else:
            constraints = np.sum(np.power(self.available_characters, exponent)) - np.sum(np.power(self.user_board, exponent), axis=0)
        for c, constraint in enumerate(constraints):
            indices = self.coefficients_by_column[c]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.sum(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['column']['additive'][exponent] = equations

    def update_multiplicative_column_constraints(self, exponent):
        equations = []
        constraints = np.power(self.region_product, exponent) // np.prod(np.power(self.multiplicative_coefficient_matrix, exponent), axis=0)
        for c, constraint in enumerate(constraints):
            indices = self.coefficients_by_column[c]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.prod(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['column']['multiplicative'][exponent] = equations

    def update_additive_block_constraints(self, exponent):
        equations = []
        blocks = self.get_all_blocks(
            board=self.user_board)
        total = np.sum(np.power(self.available_characters, exponent))
        for b, _block in enumerate(blocks):
            block = _block.reshape(-1)
            constraint = total - np.sum(np.power(block, exponent))
            indices = self.coefficients_by_block[b]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.sum(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['block']['additive'][exponent] = equations

    def update_multiplicative_block_constraints(self, exponent):
        equations = []
        blocks = self.get_all_blocks(
            board=self.multiplicative_coefficient_matrix)
        total = np.power(self.region_product, exponent)
        for b, _block in enumerate(blocks):
            block = _block.reshape(-1)
            constraint = total // np.prod(np.power(block, exponent))
            indices = self.coefficients_by_block[b]
            if len(indices) > 0:
                symbolic_values = [self.variable_parameters[i]**exponent for i in indices]
                eqn = sympy.Eq(np.prod(symbolic_values), constraint)
                equations.append(eqn)
        self._constraint_equations['block']['multiplicative'][exponent] = equations

    def update_absolute_constraints(self):
        equations = []
        for param in self.variable_parameters:
            eqn = sympy.Eq(param, np.abs(param))
            equations.append(eqn)
        self._constraint_equations['absolute'] = equations

class ConstrainedEquationSystemSolver(ConstraintMapping):

    def __init__(self):
        super().__init__()
        self.solver_method = 'system of equations solver with constraints'
        self._solution = None

    def __repr__(self):
        return "%r()" % self.__class__.__name__

    def __str__(self):
        s = "\n .. SOLVER METHOD:\n\t{}\n".format(self.solver_method)
        s += "\n .. IS SOLVED:\n\t{}\n".format(self.is_solved)
        s += "\n .. VARIABLE PARAMETERS:\n\t{}\n".format(self.variable_parameters)
        row_constraints = self.constraint_equations['row']
        column_constraints = self.constraint_equations['column']
        block_constraints = self.constraint_equations['block']
        additive_row_constraints = row_constraints['additive']
        additive_column_constraints = column_constraints['additive']
        additive_block_constraints = block_constraints['additive']
        multiplicative_row_constraints = row_constraints['multiplicative']
        multiplicative_column_constraints = column_constraints['multiplicative']
        multiplicative_block_constraints = block_constraints['multiplicative']
        for exponent, equation in additive_row_constraints.items():
            s += "\n .. ROW CONSTRAINT (+; exponent={}):\n{}\n".format(exponent, equation)
        for exponent, equation in additive_column_constraints.items():
            s += "\n .. COLUMN CONSTRAINT (+; exponent={}):\n{}\n".format(exponent, equation)
        for exponent, equation in additive_row_constraints.items():
            s += "\n .. BLOCK CONSTRAINT (+; exponent={}):\n{}\n".format(exponent, equation)
        for exponent, equation in multiplicative_row_constraints.items():
            s += "\n .. ROW CONSTRAINT (*; exponent={}):\n{}\n".format(exponent, equation)
        for exponent, equation in multiplicative_column_constraints.items():
            s += "\n .. COLUMN CONSTRAINT (*; exponent={}):\n{}\n".format(exponent, equation)
        for exponent, equation in multiplicative_block_constraints.items():
            s += "\n .. BLOCK CONSTRAINT (*; exponent={}):\n{}\n".format(exponent, equation)
        if 'absolute' in list(self.constraint_equations.keys()):
            absolute_constraints = self.constraint_equations['absolute']
            s += "\n .. ABSOLUTE CONSTRAINT\n\tall unfilled cells are positive\n"
        s += "\n .. SOLUTION TO SYSTEM OF EQUATIONS:\n{}\n".format(self.solution)
        s += "\n .. ELAPSED TIME:\n\t{}\n".format(self.elapsed)
        return s

    @property
    def solution(self):
        return self._solution

    def initialize(self):
        self.update_variable_parameters()
        self.update_additive_coefficient_matrix()
        self.update_coefficients_by_row()
        self.update_coefficients_by_column()
        self.update_coefficients_by_block()
        self.update_multiplicative_coefficient_matrix()

    def update_solution(self, solutions):
        if isinstance(solutions, dict):
            self._solution = solutions
        elif isinstance(solutions, list):
            if len(solutions) == 0:
                raise ValueError("solution does not exist")
            else:
                solution_indices = []
                for i, solution in enumerate(solutions):
                    self.populate_board_from_solution(solution)
                    self.validate_solved_board()
                    if self.is_solved:
                        solution_indices.append(i)
                    self.reset_user_board()
                    self.reset_save_state()
                nsolutions = len(solution_indices)
                if nsolutions == 0:
                    raise ValueError("insufficient constraints")
                elif nsolutions == 1:
                    j = solution_indices[0]
                    self._solution = solutions[j]
                else:
                    raise ValueError("multiple solutions exist")
        else:
            raise ValueError("debug me")

    def populate_board_from_solution(self, solution):
        rows, columns = self.get_missing_locations()
        if isinstance(solution, dict):
            for r, c, (k, v) in zip(rows, columns, solution.items()):
                self._user_board[r, c] = v
        elif isinstance(solution, (tuple, list, np.ndarray)):
            if len(solution) != rows.size:
                raise ValueError("debug me")
            for r, c, v in zip(rows, columns, solution):
                self._user_board[r, c] = v
        else:
            raise ValueError("debug me")

    def solve_system_of_equations(self):
        constraints = self.get_constraints(self.constraint_equations)
        solutions = sympy.solve(constraints, self.variable_parameters, force=True)
        self.update_solution(solutions)

    def update_constraint_equations(self, exponents, additive=False, multiplicative=False, by_row=False, by_column=False, by_block=False, absolute=False):
        if absolute:
            self.update_absolute_constraints()
        if isinstance(exponents, (int, float)):
            if by_row:
                if additive:
                    self.update_additive_row_constraints(exponent=exponents)
                if multiplicative:
                    self.update_multiplicative_row_constraints(exponent=exponents)
            if by_column:
                if additive:
                    self.update_additive_column_constraints(exponent=exponents)
                if multiplicative:
                    self.update_multiplicative_column_constraints(exponent=exponents)
            if by_block:
                if additive:
                    self.update_additive_block_constraints(exponent=exponents)
                if multiplicative:
                    self.update_multiplicative_block_constraints(exponent=exponents)
        elif isinstance(exponents, (tuple, list, np.ndarray)):
            for exponent in exponents:
                self.update_constraint_equations(
                    exponents=exponent,
                    additive=additive,
                    multiplicative=multiplicative,
                    by_row=by_row,
                    by_column=by_column,
                    by_block=by_block,
                    absolute=False)
        else:
            raise ValueError("invalid type(exponents): {}".format(type(exponents)))

    def solve(self, suppress_error=False):
        start = time.process_time()
        try:
            self.solve_system_of_equations()
        except ValueError as err:
            if ("solution does not exist" in err.args[0]) or ("insufficient constraints" in err.args[0]) or ("multiple solutions exist" in err.args[0]):
                if suppress_error:
                    print(err)
                else:
                    raise ValueError(err)
            else:
                raise ValueError("debug me")
        self.populate_board_from_solution(self.solution)
        self.validate_solved_board()
        self._elapsed = time.process_time() - start
















##
