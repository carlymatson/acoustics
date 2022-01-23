import math
import numpy as np
from typing import Tuple, List, Callable

# Point grid
# List of vectors
# Compute matrix of dot products
#


def points_of_grid(
    x_bounds: Tuple[int, int], y_bounds: Tuple[int, int]
) -> List[Tuple[int, int]]:
    # FIXME Add ability to sort.
    return [(x, y) for x in range(*x_bounds) for y in range(*y_bounds)]


def get_evenly_spaced_vectors():
    pass


def dot_product(point, vector) -> float:
    return 0


def get_wave_function(amplitude: float, vector) -> Callable:
    def wave(x, y):
        dot = dot_product(vector, (x, y))
        return amplitude * math.cos(dot * 2 * math.pi)

    return wave


def get_function_sum(functions):
    def function_sum(x, y):
        outputs = [f(x, y) for f in functions]
        return sum(outputs)

    return function_sum


def get_wave_matrix(points, vectors):
    pass


def get_root_of_unity(n: int) -> complex:
    return 0j


def get_hermitian_matrix(matrix):
    """Takes the transpose and the complex conjugate of each entry."""
    nrows, ncols = matrix.shape
    conjugate_matrix = np.matrix(
        [[matrix[row, col].conjugate() for col in range(ncols)] for row in range(nrows)]
    )
    return conjugate_matrix.transpose()
