from frontend_configuration import *
from performance_evaluation import *

if __name__ == '__main__':

    ## check performance evaluation
    # variable_parameter = 'difficulty level'
    # performance_evaluation = PerformanceEvaluation()
    # performance_evaluation.perform_multiple_runs(
    #     variable_parameter=variable_parameter,
    #     use_recursive_backtracker=True,
    #     use_constrained_system_equations=False,
    #     use_dancing_links=False,
    #     nruns=100)
    # performance_evaluation.update_evaluations(variable_parameter)

    ## solve single board using multiple methods
    sudoku = SudokuBackEnd()
    # sudoku.generate_board(
    #     nmissing=4,
    #     nmaps=10,
    #     nrots=4)
    # sudoku.generate_board(
    #     nmissing=10,
    #     nmaps=10,
    #     nrots=4)
    # sudoku.generate_board(
    #     nmissing=20,
    #     nmaps=10,
    #     nrots=4)
    # sudoku.generate_board(
    #     nmissing=30,
    #     nmaps=10,
    #     nrots=4)
    # sudoku.generate_board(
    #     board='easy',
    #     reflect_horizontal=True,
    #     reflect_vertical=True,
    #     row_band_swaps=([0], [7]),
    #     column_band_swaps=([1], [5]))
    # sudoku.generate_board(
    #     board='easy')
    # sudoku.generate_board(
    #     board='medium')
    # sudoku.generate_board(
    #     board='hard')
    # sudoku.generate_board(
    #     board='expert')
    # sudoku.generate_board(
    #     board='worlds toughest puzzle')
    sudoku.verify_number_of_clues()
    sudoku.assign_difficulty_level()
    # sudoku.show_original_board_text()
    sudoku.add_solver(
        solver=RecursiveSolver(),
        cell_selection_method='naive')
    sudoku.add_solver(
        solver=RecursiveSolver(),
        cell_selection_method='random')
    sudoku.add_solver(
        solver=RecursiveSolver(),
        cell_selection_method='probabilistic')
    sudoku.add_solver(
        solver=RecursiveSolver(),
        cell_selection_method='inverse frequency')
    sudoku.add_solver(
        solver=RecursiveSolver(),
        cell_selection_method='adaptive')
    sudoku.add_solver(
        solver=ConstrainedEquationSystemSolver(),
        exponents=(1, 2), # 1
        additive=True,
        multiplicative=True,
        by_row=True,
        by_column=True,
        by_block=True,
        absolute=False)
    # sudoku.add_solver(
    #     solver=DancingLinksExactCoverSolver())
    sudoku.solve() # suppress_error=True



##
