from collections import Counter
from board_configuration import *

from scipy.optimize import minimize, basinhopping

class StochasticMetropolisHastingsSolver(SolverConfiguration):

    def __init__(self):
        super().__init__()
        self.solver_method = 'stochastic solver'
        self._permutable_characters = None
        self._permutable_indices = None
        self._missing_rows = None
        self._missing_cols = None
        self._stochastic_method = None
        self._schedule_by = None
        self._temperature = []
        self._energy = []
        self._max_iter = None
        self._niter = 0
        self._update_temperature = None
        self._f_solve = None

    def __repr__(self):
        return "%r()" % self.__class__.__name__

    def __str__(self):
        s = "\n .. SOLVER METHOD:\n\t{}\n".format(self.solver_method)
        s += "\n .. IS SOLVED:\n\t{}\n".format(self.is_solved)
        s += "\n .. STOCHASTIC METHOD:\n\t{}\n".format(self.stochastic_method)
        s += "\n .. SCHEDULE BY:\n\t{}\n".format(self.schedule_by)
        s += "\n .. TEMPERATURE:\n\t{}\n".format(self.temperature[-1])
        s += "\n .. ENERGY:\n\t{}\n".format(self.energy[-1])
        s += "\n .. NUMBER OF ITERATIONS:\n\t{}\n".format(self.niter)
        s += "\n .. ELAPSED TIME:\n\t{}\n".format(self.elapsed)
        return s

    @property
    def missing_rows(self):
        return self._missing_rows

    @property
    def permutable_characters(self):
        return self._permutable_characters

    @property
    def permutable_indices(self):
        return self._permutable_indices

    @property
    def missing_cols(self):
        return self._missing_cols

    @property
    def stochastic_method(self):
        return self._stochastic_method

    @property
    def schedule_by(self):
        return self._schedule_by

    @property
    def temperature(self):
        return self._temperature

    @property
    def energy(self):
        return self._energy

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def niter(self):
        return self._niter

    @property
    def update_temperature(self):
        return self._update_temperature

    @property
    def f_solve(self):
        return self._f_solve

    @staticmethod
    def get_number_of_violations_per_flat_dimension(board, axis):
        s = np.sort(board, axis=axis)
        d = np.diff(s != 0, axis=axis)
        return np.sum(d, axis=axis)

    def scheduling_by_boltzmann(self, k):
        t = self.temperature[0] / np.log(k)
        self._temperature.append(t)

    def scheduling_by_linear_decay(self, k):
        ## y = mx + b ==> (x, y) = (k, self.temperature[k]) ==> m = -1 * T[0] / self.max_iter, b = T[0]
        t = -1 * self.temperature[0] / self.max_iter + self.temperature[0]
        self._temperature.append(t)

    def scheduling_by_standard(self, k):
        t = self.temperature[-1] * (1 - (k + 1) / self.max_iter) ## self.temperature[-1] * 0.99
        self._temperature.append(t)

    def initialize_prerequisites(self):
        self._missing_rows, self._missing_cols = self.get_missing_locations()
        found_counter = Counter(self.user_board.reshape(-1))
        missing_counter = {
            char : self.available_characters.size - found_counter[char] for char in self.available_characters}
        permutable_characters = []
        for char, freq in missing_counter.items():
            for i in range(freq):
                permutable_characters.append(char)
        self._permutable_characters = np.array(permutable_characters)
        self._permutable_indices = np.arange(self.permutable_characters.size, dtype=int)
        for r, c, value in zip(self.missing_rows, self.missing_cols, self.permutable_characters):
            self._user_board[r, c] = value

    def select_stochastic_method_and_parameters(self, stochastic_method, initial_temperature, schedule_by, max_iter=1e5):
        ## max_iter
        if max_iter > 0:
            self._max_iter = int(max_iter)
        else:
            raise ValueError("invalid max_iter: {}".format(max_iter))
        ## stochastic_method
        stochastic_map = {
            'hill climbing' : self.stochastic_hill_climbing,
            'simulated annealing' : self.simulate_annealing}
        if stochastic_method not in list(stochastic_map.keys()):
            raise ValueError("invalid stochastic_method: {}".format(stochastic_method))
        self._f_solve = stochastic_map[stochastic_method]
        self._stochastic_method = stochastic_method
        ## temperature and schedule_by
        self._temperature.append(initial_temperature)
        schedule_to_temperature_map = {
            'boltzmann' : self.scheduling_by_boltzmann,
            'linear decay' : self.scheduling_by_linear_decay,
            'standard' : self.scheduling_by_standard}
        if schedule_by not in list(schedule_to_temperature_map.keys()):
            raise ValueError("invalid schedule_by: {}".format(schedule_by))
        self._update_temperature = schedule_to_temperature_map[schedule_by]
        self._schedule_by = schedule_by

    def get_probability(self, delta_energy):
        return np.exp(-1 * delta_energy / self.temperature[-1])

    def get_number_of_constraint_violations(self):
        row_v = self.get_number_of_violations_per_flat_dimension(
            board=self.user_board,
            axis=1)
        col_v = self.get_number_of_violations_per_flat_dimension(
            board=self.user_board,
            axis=0)
        block_v = 0
        for block in self.get_all_blocks(self.user_board):
            block_v += (self.available_characters.size - len(set(block.reshape(-1))))
        return np.sum(row_v) + np.sum(col_v) + block_v

    def swap_coordinate_pair(self, ri, rj, ci, cj):
        self._user_board[ri, ci], self._user_board[rj, cj] = self._user_board[rj, cj], self._user_board[ri, ci]

    def select_neighbor_index(self, i):
        j = np.random.choice(self.permutable_indices)
        if i == j:
            j = self.select_neighbor_index(i=i)
        return j

    def stochastic_hill_climbing(self):
        for k in range(self.max_iter):
            self._niter += 1
            prev_energy = self.get_number_of_constraint_violations()
            if prev_energy == 0:
                break
            else:
                i, j = np.random.choice(
                    self.permutable_indices,
                    size=2,
                    replace=False)
                self.swap_coordinate_pair(
                    ri=self.missing_rows[i],
                    rj=self.missing_rows[j],
                    ci=self.missing_cols[i],
                    cj=self.missing_cols[j])
                curr_energy = self.get_number_of_constraint_violations()
                if curr_energy > prev_energy:
                    rng = np.random.uniform(low=0, high=1)
                    delta_energy = curr_energy - prev_energy
                    threshold = self.get_probability(delta_energy)
                    if rng > threshold:
                        self._energy.append(curr_energy)
                    else:
                        self.swap_coordinate_pair(
                            ri=self.missing_rows[i],
                            rj=self.missing_rows[j],
                            ci=self.missing_cols[i],
                            cj=self.missing_cols[j])
                        self._energy.append(prev_energy)
                else:
                    self._energy.append(curr_energy)
                self.update_temperature(k)

    def simulate_annealing(self):
        i = np.random.choice(self.permutable_indices)
        for k in range(self.max_iter):
            self._niter += 1
            prev_energy = self.get_number_of_constraint_violations()
            if prev_energy == 0:
                break
            else:
                j = self.select_neighbor_index(i)
                self.swap_coordinate_pair(
                    ri=self.missing_rows[i],
                    rj=self.missing_rows[j],
                    ci=self.missing_cols[i],
                    cj=self.missing_cols[j])
                curr_energy = self.get_number_of_constraint_violations()
                if curr_energy > prev_energy:
                    rng = np.random.uniform(low=0, high=1)
                    delta_energy = curr_energy - prev_energy
                    threshold = self.get_probability(delta_energy)
                    if rng > threshold:
                        self._energy.append(curr_energy)
                        i = j
                    else:
                        self.swap_coordinate_pair(
                            ri=self.missing_rows[i],
                            rj=self.missing_rows[j],
                            ci=self.missing_cols[i],
                            cj=self.missing_cols[j])
                        self._energy.append(prev_energy)
                else:
                    self._energy.append(curr_energy)
                    i = j
                self.update_temperature(k)

    def solve(self, suppress_error=False):
        start = time.process_time()
        self.f_solve()
        if self.energy[-1] == 0:
            self._is_solved = True
        else:
            if not suppress_error:
                raise ValueError("solution not found")
        self._elapsed = time.process_time() - start


# user_board = np.array([
#     [4, 8, 9, 1, 5, 3, 2, 6, 7],
#     [1, 5, 3, 2, 0, 7, 0, 8, 9],
#     [2, 6, 7, 4, 8, 9, 1, 5, 3],
#     [8, 0, 1, 5, 3, 2, 0, 7, 4],
#     [5, 3, 2, 6, 7, 4, 8, 9, 1],
#     [6, 7, 4, 8, 9, 1, 5, 3, 2],
#     [9, 1, 5, 0, 2, 6, 7, 4, 8],
#     [3, 2, 6, 7, 4, 8, 9, 1, 5],
#     [7, 4, 8, 9, 1, 5, 3, 2, 6]])
#
# solver = StochasticMetropolisHastingsSolver()
# solver._user_board = user_board.copy()
# solver._original_board = user_board.copy()
# solver.initialize_prerequisites()
# solver.select_stochastic_method_and_parameters(
#     # stochastic_method='hill climbing',
#     stochastic_method='simulated annealing',
#     initial_temperature=0.8,
#     # schedule_by='boltzmann',
#     schedule_by='standard',
#     # schedule_by='linear decay',
#     max_iter=1000)
# solver.solve()
# print(solver)


##
