import numpy as np
from math import ceil
"""
#used for debugging
class DataType:
    def __init__(self, value, cycle):
        self.value = value
        self.cycle = cycle

    def __repr__(self):
        return f"DataType({self.value}, {self.cycle})"
"""

def systolic_array_psums(DataType, A, B, tile_size):
    # Get dimensions of the input matrices
    m, n = A.shape
    n_b, p = B.shape
    
    assert n == n_b, "Inner dimensions of matrices must match for multiplication."
    
    # Calculate the number of tiles
    m_tiles = m // tile_size
    n_tiles = n // tile_size
    p_tiles = p // tile_size
    
    # Initialize the partial sums list
    partial_sums = [[] for _ in range(tile_size)]

    # Iterate over tiles
    for i in range(p_tiles):
        row_sums = []
        for j in range(n_tiles):
            psum = 0
            for k in range(m_tiles):
                # Extract the tiles
                A_tile = A[k*tile_size:(k+1)*tile_size, j*tile_size:(j+1)*tile_size]
                B_tile = B[j*tile_size:(j+1)*tile_size, i*tile_size:(i+1)*tile_size]
                # Compute the partial sum for the current tile
                psums = np.dot(A_tile, B_tile)
                for row in range(len(psums)):
                    for column in range(len(psums[0])):
                        partial_sums[column].append(DataType(psums[row][column], 1))

    return partial_sums


def transform_weights(DataType, B, tile_size, II, iterations):
    # Get the dimensions of matrix B
    rows, cols = B.shape

    # Check if the dimensions are divisible by the tile size
    assert rows % tile_size == 0 and cols % tile_size == 0, "Matrix dimensions must be divisible by tile size."

    transformed_weights = [[] for _ in range(tile_size*tile_size)]

    # Iterate over the tiles
    for j in range(0, cols, tile_size):
        for i in range(0, rows, tile_size):

            tile_nb = 0

            # Iterate over the elements within the tile
            for tj in range(tile_size):
                for ti in range(tile_size):
                    
                    # Get the value of the element in the tile
                    value = B[i + ti][j + tj]

                    # Append DataType elements II times
                    for _ in range(0, II*iterations):
                        transformed_weights[tile_nb].append(DataType(value, 1))
                    tile_nb += 1
    return transformed_weights


def transform_inputs(DataType, A, tile_size):
    transformed_elements = []

    # Get the number of rows and columns of the array
    num_rows, num_cols = A.shape

    # Iterate over each column of the array
    for j in range(0, num_cols, tile_size):
        # Iterate over each row within the column
        for i in range(0, num_rows, tile_size):
            # Append DataType instance with cycle 1
            for tj in range(tile_size):
                for ti in range(tile_size):
                    transformed_elements.append(DataType(A[i + ti][j + tj], 1))

    return transformed_elements

def get_mem_load_offsets(DataType, width, II, iterations, k):
    get_mem_load_offsets = [[] for _ in range(width)]
    for _ in range(ceil(k/width)):
        counter = 0
        for _ in range(iterations):
            for tile in range(width):
                preload_tile = [DataType(0, 0) for _ in range(tile)]
                for _ in range(width):
                    preload_tile.append(DataType(counter, 1))
                    counter += 1
                while len(preload_tile) < II:
                    preload_tile.append(DataType(0, 0))
                get_mem_load_offsets[tile].extend(preload_tile)
    
    return get_mem_load_offsets


def get_workload( DataType, I, W, width, II ):
  #O[b][k] += W[k][c] * I[b][c]
  b, c = np.shape(I)
  k = np.shape(W)[1]
  iterations = ceil(b*c/(width*width))
  weights_in_array_iterations = ceil(b/width)

  mem_load_offsets= get_mem_load_offsets(DataType, width, II, iterations, k)
  
  weights = transform_weights(DataType, W, width, II, weights_in_array_iterations)
  
  preload_const = mem_load_offsets + weights
  inputs = transform_inputs(DataType, I, width)
  psums = systolic_array_psums(DataType, I, W, width)
  cycles = len(preload_const[0])
  data_mem_size = len(inputs)

  return preload_const, inputs, psums, cycles, data_mem_size

"""
#used for debugging
#define workload
A = np.array([
    [1, 3, 2, 5],
    [2, 4, 8, 9],
    [2, 5, 3, 4],
    [4, 3, 2, 1]
    ])

B = np.array([
    [2, 6, 7, 3],
    [4, 8, 2, 1],
    [7, 6, 3, 5],
    [9, 8, 6, 9],
    ])

preload_const, inputs, psums, cycles, data_mem_size = get_workload( DataType, A, B, 2, 5 )
print(preload_const, "\n\n", inputs, "\n\n", psums,"\n\n", cycles)
"""