import numpy as np
from itertools import product
from collections import defaultdict
import math
import random
def generate_sudoku():
    
    """
        Function to generate a Sudoku grid

        
        Fill the diagonal boxes of the Sudoku grid
        Solve the Sudoku grid
        Remove elements from the grid to create a puzzle
    """
    n = 9
    sub_table_size = 3
    grid = [[0 for i in range(9)] for i in range(9)]
    fill_diagonal(grid)
    l = [0,9]
    i = l[0]
    j = l[1]
    #find_unassigned_location(grid,l)
    fill_unassigned_locations(grid,i,j)
    remove_elements(grid,math.floor(random.random() * 60  + 1))
    return grid
    
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############



def fill_diagonal(grid):
    
    """
        Function to fill the diagonal boxes of the Sudoku grid
    """
    for i in range(0, 9, 3):
        fill_box(grid, i, i)


def fill_box(grid, row, col):

    """
        Function to fill a box with random values
    """
    num = 0
    for i in range(3):
        for j in range(3):
            while True:
                num = math.floor(random.random() * 9 + 1)
                if not used_in_box(grid,row, col, num):
                    break
            grid[row + i][col + j] = num
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############

def is_safe(grid, row, col, num):

    """
        Function to check if it is safe to place a number in a particular position
    """

    return (
        not used_in_row(grid, row, num)
        and not used_in_column(grid, col, num)
        and not used_in_box(grid, row - row % 3, col - col % 3, num)
    )


# 
def used_in_row(grid, row, num):

    """
        Function to check if a number is used in a row
    """
    for i in range(9):
        if grid[row][i]  == num:
            return True
    return False
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############


def used_in_column(grid, col, num):

    """
        Function to check if a number is used in a column
    """
    for i in range(9):
        if grid[i][col]  == num:
            return True
    return False
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############


def used_in_box(grid, box_start_row, box_start_col, num):

    """
        Function to check if a number is used in a 3x3 box
    """
    for i in range(3):
        for j in range(3):
            if grid[box_start_row + i][box_start_col + j] == num:
                return True
    return False
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############

def find_unassigned_location(grid,l):

    """
        Function to find an unassigned location in the grid
    """
    for i in range(9):
        for j in range(9):
            if(grid[i][j]== 0):
                l[0]= i
                l[1]= j
                return True
    return False
         
    ############
    ##        ##
    ##  Code  ##
    ##        ##
    ############
    
def fill_unassigned_locations(grid,i,j):
    if i == 8 and j == 9:
        return True
    
    if j == 9:
        i += 1
        j = 0
    if grid[i][j] != 0:
        return fill_unassigned_locations(grid,i,j + 1)
    
    for num in range(1,10):
        if is_safe(grid,i,j,num):
            grid[i][j] = num
            if fill_unassigned_locations(grid,i,j + 1):
                return True
            grid[i][j] = 0
    return False
def remove_elements(grid, num_elements):
    
    """
        Function to remove elements from the grid
    """
    
 
    while (num_elements != 0):
        i = math.floor(random.random() * 9 + 1) - 1
        j = math.floor(random.random() * 9 + 1) - 1
        if (grid[i][j] != 0):
            num_elements -= 1
            grid[i][j] = 0
 
    return grid

def solve_sudoku_csp(grid):
    """
    Function to solve the Sudoku grid using Constraint Satisfaction Problem (CSP)
    """
    steps = 0
    def create_domains(grid):
        """
        Function to create domains for each cell in the grid
        """
        domains = {}
        for i in range(9):
            for j in range(9):
                domains[(i, j)] = []
                if grid[i][j] == 0:
                    #domains[(i, j)] = set(range(1, 10))
                    for k in range(1,10):
                        if is_valid_assignment(i, j, k, grid):
                            domains[(i,j)].append(k)
                else:
                    domains[(i, j)] = grid[i][j]
        return domains

    def is_valid_assignment(i, j, val, assignment):
        """
        Function to check if assigning a value to a cell is valid
        Check if the value is already used in the same row
        Check if the value is already used in the same column
        Check if the value is already used in the same 3x3 box
        """
        
        if is_safe(grid,i,j,val):
            return True
        return False
        '''for row in range(9):
            if assignment[row][j] == val or assignment[i][row] == val:
                return False

        start_row, start_col = 3 * (i // 3), 3 * (j // 3)
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                if assignment[row][col] == val:
                    return False

        return True'''

    def find_unassigned_location(assignment):
        """
        Function to find an unassigned location in the grid
        """
        for i in range(9):
            for j in range(9):
                if not type(assignment[(i,j)]) == int:
                    return i, j
        return -1, -1

    def solve_csp(assignment):
        nonlocal domains
        nonlocal steps
        global final_grid
        
        i, j = find_unassigned_location(assignment)
        if i == -1 and j == -1:
            return True  
        assignment = domains
        for val in assignment[(i, j)]:
            steps += 1
            #if is_valid_assignment(i, j, val, assignment):
            x = assignment[(i, j)]
            y = grid[i][j]
            assignment[(i,j)] = val
            domains[(i, j)] = val
            grid[i][j] = val
            domains = create_domains(grid)
            if solve_csp(assignment):
                final_grid = domains
                return True
            assignment[(i,j)] = x
            grid[i][j] = y
        return False
            #result = solve_csp(assignment)
            #if result:
            #    return True  
        #return False  

    domains = create_domains(grid)
    assignment = domains
    if solve_csp(assignment):
        grid = [[final_grid[(i, j)] for j in range(9)] for i in range(9)]
        print (f'{steps} steps')
        return grid
    return None
# Function to initialize the Sudoku grid
def initializing_grid():
    print("\nInitial Sudoku\n")
    sudoku_grid = generate_sudoku()
    for i in sudoku_grid:
        for j in i:
            print(j, end=' ')
        print("")
    print(end='\n\n\n\n\n')
    return sudoku_grid


# Function to solve the Sudoku grid using CSP
def csp_answer(sudoku_grid):
    print("CSP Answer:\n")
    solved_grid = solve_sudoku_csp(sudoku_grid)
    if solved_grid is not None:
        print("Sudoku solved successfully:")
        for row in solved_grid:
            for r1 in row:
                print(r1, end=' ')
            print()
    else:
        print("No solution exists for the given Sudoku.")
    print(end='\n\n\n\n\n')
    
# Generate and display the initial Sudoku grid
generate_sudoko = initializing_grid()


# Solve the Sudoku grid using backtracking


# Solve the Sudoku grid using CSP
csp_answer(generate_sudoko)