import numpy as np
import random
from itertools import product
from collections import defaultdict
import copy


def generate_sudoku():

    """
        Function to generate a Sudoku grid

        
        Fill the diagonal boxes of the Sudoku grid
        Solve the Sudoku grid
        Remove elements from the grid to create a puzzle
    """
    grid=[[0 for j in range(9)] for i in range(9)]
    
    fill_diagonal(grid)

    solve_sudoku_csp(grid)
    
    remove_elements(grid,random.randint(30,50))
    
    return grid

def fill_diagonal(grid):
    
    """
        Function to fill the diagonal boxes of the Sudoku grid
    """

    for i in range(0, 9, 3):
        fill_box(grid, i, i)
    

def used_in_box(grid, box_start_row, box_start_col, num):

    """
        Function to check if a number is used in a 3x3 box
    """

    for i in range(box_start_row,box_start_row+3):
        for j in range(box_start_col,box_start_col+3):
            if grid[i][j]==num:
                return True
    return False
def fill_box(grid, row, col):

    """
        Function to fill a box with random values
    """

    for i in range(row,row+3):
        for j in range(col,col+3):
            valid=True
            while valid:
                number=random.randint(1,9)
                if not used_in_box(grid, row, col, number):
                    grid[i][j]=number
                    valid=False
                    


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
    for j in range(9):
        if grid[row][j]==num:
            return True
    return False
    


def used_in_column(grid, col, num):

    """
        Function to check if a number is used in a column
    """
    for i in range(9):
        if grid[i][col]==num:
            return True
    return False

def find_unassigned_location(grid):
    """
        Function to find an unassigned location in the grid
    """
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return i,j
    return -1,-1


def remove_elements(grid, num_elements):
    
    """
        Function to remove elements from the grid
    """
    for k in range(num_elements):
        i=random.randint(0,8)
        j=random.randint(0,8)
        grid[i][j]=0



"Backtracking"
backtracking_step=0
def solve_sudoku(grid):
    global backtracking_step
    """
        Function to solve the Sudoku grid using backtracking
    """
    i, j = find_unassigned_location(grid)
        
    if i == -1 and j == -1:
        return True
    
    # Try assigning each possible value to the unassigned location
    for val in range(1,10):
        backtracking_step+=1
        if is_safe(grid,i,j,val):
            t2=grid[i][j]
            grid[i][j]=val
            if solve_sudoku(grid):
                return True
            grid[i][j]=t2
    
    return False




   
    

def display_grid(grid):

    """
        Function to display the Sudoku grid
    """

    for i in range(9):
        for j in range(9):
            print(grid[i][j], end=" ")
        print()






        "CSP"
csp_step=0
def solve_sudoku_csp(grid):
    
    """
        Function to solve the Sudoku grid using Constraint Satisfaction Problem (CSP)
    """

    def create_domains(grid):

        """
            Function to create domains for each cell in the grid
        """
        domains={}
        for i in range(9):
            for j in range(9):
                domains[(i,j)]=[]
                if grid[i][j]!=0:
                    domains[(i,j)]=grid[i][j]
                    continue
                for k in range(1,10):
                    if is_safe(grid,i,j,k):
                        domains[(i,j)].append(k)

        return domains
    

    def is_valid_assignment(i, j, val, assignment):

        """
            Function to check if assigning a value to a cell is valid

            Check if the value is already used in the same row
            Check if the value is already used in the same column
            Check if the value is already used in the same 3x3 box
        """
        
        # for key in assignment.keys():
        #     if type(assignment[key])==int:
        #         grid[key[0]][key[1]]=assignment[key]
        #     else:
        #         grid[key[0]][key[1]]=0
        if is_safe(grid,i,j,val):
            return True
        return False


        

    def find_unassigned_location(assignment):

        """
            Function to find an unassigned location in the grid
        """
        for i in range(9):
            for j in range(9):
                if type(assignment[(i,j)]) == list:
                    return i,j
        return -1,-1
    # Recursive function to solve the Sudoku grid using CSP
    def solve_csp(assignment):
        global csp_step
        i, j = find_unassigned_location(assignment)
        
        if i == -1 and j == -1:
            return True
        
        # Try assigning each possible value to the unassigned location
        for val in assignment[(i, j)]:
            csp_step+=1
            if is_valid_assignment(i, j, val, assignment):
                t1=assignment[(i, j)]
                t2=grid[i][j]
                assignment[(i, j)] = val
                grid[i][j]=val
                
                if solve_csp(assignment):
                    domains = create_domains(grid)
                    assignment = {(i, j): val for (i, j), val in domains.items()}
                    return True
                assignment[(i, j)]=t1
                grid[i][j]=t2
        
        return False
    global csp_step
    csp_step=0
    # Create initial domains for each cell in the grid
    domains = create_domains(grid)
    assignment = {(i, j): val for (i, j), val in domains.items()}
    # Solve the Sudoku grid using CSP
    if solve_csp(assignment):
        solved_grid = [[assignment[(i, j)] for j in range(9)] for i in range(9)]
        return solved_grid
    else:
        return None


"Show Result"

# Function to initialize the Sudoku grid
def initializing_grid():
    print("\nInitial Sudoku\n")
    sudoku_grid = generate_sudoku()
    for i in sudoku_grid:
        for j in i:
            print(j, end=' ')
        print("")
    print(end='\n')
    return sudoku_grid


# Function to solve the Sudoku grid using backtracking
def backtracking_answer(sudoku_grid):
    print("\n\nBack Tracking Answer:\n")

    if solve_sudoku(sudoku_grid):
        print("Sudoku solved successfully:")
        display_grid(sudoku_grid)
        print("BackTraking Step:",backtracking_step)
    else:
        print("No solution exists for the given Sudoku.")
    print(end='\n\n\n')


# Function to solve the Sudoku grid using CSP
def csp_answer(sudoku_grid):
    print("CSP Answer:\n")
    solved_grid = solve_sudoku_csp(sudoku_grid)
    if solved_grid is not None:
        print("Sudoku solved successfully:")
        for row in solved_grid:
            for r1 in row:
                for r2 in [r1]:
                    print(r2, end=' ')
            print()
        print("CSP Step:",csp_step)
    else:
        print("No solution exists for the given Sudoku.")
    # print(end='\n\n\n')




# # Generate and display the initial Sudoku grid
generate_sudoko = initializing_grid()

# # Solve the Sudoku grid using backtracking
generate_sudoko1=copy.deepcopy(generate_sudoko)
generate_sudoko2=copy.deepcopy(generate_sudoko)


backtracking_answer(generate_sudoko2)

csp_answer(generate_sudoko1)
# Solve the Sudoku grid using CSP

