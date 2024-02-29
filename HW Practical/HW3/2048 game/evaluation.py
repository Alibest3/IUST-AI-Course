import numpy as np

import game_functions as gf
import math

def evaluate_state(board: np.ndarray) -> float:
    """
    Returns the score of the given board state.
    :param board: The board state for which the score is to be calculated.
    :return: The score of the given board state.
    """
    # TODO: Complete evaluate_state function to return a score for the current state of the board
    # Hint: You may need to use the np.nonzero function to find the indices of non-zero elements.
    # Hint: You may need to use the gf.within_bounds function to check if a position is within the bounds of the board.
    grid = np.array(grid)

    score = 0

    big_t = np.sum(np.power(grid, 2))

    smoothness = 0
    s_grid = np.sqrt(grid)

    smoothness -= np.sum(np.abs(s_grid[:, 0] - s_grid[:, 1]))
    smoothness -= np.sum(np.abs(s_grid[:, 1] - s_grid[:, 2]))
    smoothness -= np.sum(np.abs(s_grid[:, 2] - s_grid[:, 3]))
    smoothness -= np.sum(np.abs(s_grid[0, :] - s_grid[1, :]))
    smoothness -= np.sum(np.abs(s_grid[1, :] - s_grid[2, :]))
    smoothness -= np.sum(np.abs(s_grid[2, :] - s_grid[3, :]))

    monotonic_up = 0
    monotonic_down = 0
    monotonic_left = 0
    monotonic_right = 0

    for x in range(4):
      current = 0
      next = current + 1
      while next < 4:
        while next < 3 and not grid[next, x]:
          next += 1
        current_cell = grid[current, x]
        current_value = math.log(current_cell, 2) if current_cell else 0
        next_cell = grid[next, x]
        next_value = math.log(next_cell, 2) if next_cell else 0
        if current_value > next_value:
          monotonic_up += (next_value - current_value)
        elif next_value > current_value:
          monotonic_down += (current_value - next_value)
        current = next
        next += 1

    for y in range(4):
      current = 0
      next = current + 1
      while next < 4:
        while next < 3 and not grid[y, next]:
          next += 1
        current_cell = grid[y, current]
        current_value = math.log(current_cell, 2) if current_cell else 0
        next_cell = grid[y, next]
        next_value = math.log(next_cell, 2) if next_cell else 0
        if current_value > next_value:
          monotonic_left += (next_value - current_value)
        elif next_value > current_value:
          monotonic_right += (current_value - next_value)
        current = next
        next += 1

    monotonic = max(monotonic_up, monotonic_down) + max(monotonic_left, monotonic_right)
  
    empty_w = 100000
    smoothness_w = 3
    monotonic_w = 10000
    
    empty_u = n_empty * empty_w
    smooth_u = smoothness ** smoothness_w
    monotonic_u = monotonic * monotonic_w
    
    score += big_t
    score += empty_u
    score += smooth_u
    score += monotonic_u
    
    return score
    
    #raise NotImplementedError("Evaluation function not implemented yet.")
