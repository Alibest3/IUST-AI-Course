class GameSolution:
    """
        A class for solving the Water Sort game and finding solutions(normal, optimal).

        Attributes:
            ws_game (Game): An instance of the Water Sort game which implemented in game.py file.
            moves (List[Tuple[int, int]]): A list of tuples representing moves between source and destination tubes.
            solution_found (bool): True if a solution is found, False otherwise.

        Methods:
            solve(self, current_state):
                Find a solution to the Water Sort game from the current state.
                After finding solution, please set (self.solution_found) to True and fill (self.moves) list.

            optimal_solve(self, current_state):
                Find an optimal solution to the Water Sort game from the current state.
                After finding solution, please set (self.solution_found) to True and fill (self.moves) list.
    """
    def __init__(self, game):
        """
            Initialize a GameSolution instance.
            Args:
                game (Game): An instance of the Water Sort game.
        """
        self.ws_game = game  # An instance of the Water Sort game.
        self.moves = []  # A list of tuples representing moves between source and destination tubes.
        self.tube_numbers = game.NEmptyTubes + game.NColor  # Number of tubes in the game.
        self.solution_found = False  # True if a solution is found, False otherwise.
        self.visited_tubes = set()  # A set of visited tubes.

    def solve(self, current_state):
        """
            Find a solution to the Water Sort game from the current state.

            Args:
                current_state (List[List[int]]): A list of lists representing the colors in each tube.

            This method attempts to find a solution to the Water Sort game by iteratively exploring
            different moves and configurations starting from the current state.
        """
        if self.solution_found:
            return

        current_state_tuple = tuple(tuple(glass) for glass in current_state)

        if current_state_tuple in self.visited_tubes:
            return

        if self.ws_game.check_victory(current_state):
            self.solution_found = True
            return

        for source_idx, source_glass in enumerate(current_state):
            if source_glass:
                for dest_idx, dest_glass in enumerate(current_state):
                    if source_idx != dest_idx:
                        if  not dest_glass or (source_glass[-1] == dest_glass[-1] and len(dest_glass) < self.ws_game.NColorInTube):
                            #if len(dest_glass) + len(source_glass) <= self.ws_game.NColorInTube:
                                water_color = source_glass[-1]
                                new_state = [list(glass) for glass in current_state]

                                while len(source_glass) > 0 and len(new_state[dest_idx]) < self.ws_game.NColorInTube:
                                    try:
                                        if source_glass[-1] == dest_glass[-1] :
                                            new_state[dest_idx].append(source_glass.pop())
                                            dest_glass = new_state[dest_idx]
                                            new_state[source_idx] = source_glass
                                            current_state = new_state
                                    except:
                                            new_state[dest_idx].append(source_glass.pop())
                                            dest_glass = new_state[dest_idx]
                                            new_state[source_idx] = source_glass
                                            current_state = new_state
                                    else:
                                        break

                                self.visited_tubes.add(current_state_tuple)

                                self.moves.append((source_idx, dest_idx))

                                self.solve(new_state)

                                if self.solution_found:
                                    return

                                self.moves.pop()
        self.solve(current_state)
        """self.ws_game.NColor
        i = 0
        if not self.ws_game.check_victory(current_state):
            for tube in current_state:
                #for color in tube:
               if len(current_state[-1]) == 0:
                    current_state[-1].append(tube[-1])
                    current_state[i] = tube[0:len(tube)-1]
                    i += 1
        
        else:
            self.solution_found = True"""
            
        

    def optimal_solve(self, current_state):
        """
            Find an optimal solution to the Water Sort game from the current state.

            Args:
                current_state (List[List[int]]): A list of lists representing the colors in each tube.

            This method attempts to find an optimal solution to the Water Sort game by minimizing
            the number of moves required to complete the game, starting from the current state.
        """
        pass
