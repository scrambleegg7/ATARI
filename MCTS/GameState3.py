import copy
import numpy as np

from ATARI.MCTS import get_module_logger

mylogger = get_module_logger(__name__)



class GameState(object):
    """
    Represents a Tic Tac Toe game.
    The state consists of a 3x3 game board with each position occupied by:
      ' ' (empty square)
      'X' (X mark)
      'O' (O mark)
    as well as the following terminal states:
      X won
      O won
      Tie
    """
    def __init__(self):
        # Begin with an empty game board

        self.board = np.array([" "] * 10)
        self.NUM_MOVES = 9

    # GameState needs to be hashable so that it can be used as a unique graph
    # node in NetworkX
    def __key(self):
        return self.__str__()

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        """
        Returns a string that is a visual representation of the game
        state. Can be used to print the current game state of a game:
          print(game.state)
        will print a game board:
          ~X~
          O~~
          ~~X
        """
        output = ''
        for r in range(10)[1:]:
            contents = self.board[r]
            if r % 3 == 0:
                output += '{}\n'.format(contents)
            else:
                output += '{}'.format(contents)

        output = output.replace(' ', '~')

        return output

    def turn(self):
        """
        Returns the player whose turn it is: 'X' or 'O'
        """
        num_X = 0
        num_O = 0
        for r in range(10)[1:]:
            if self.board[r] == 'X':
                num_X += 1
            elif self.board[r] == 'O':
                num_O += 1

        if num_X != num_O:
            return 'X'
        else:
            return 'O'

    def move(self, r):
        """
        Places a marker at the position (row, col). The marker placed is
        determined by whose turn it is, either 'X' or 'O'.
        """
        if self.board[r] != " ":
            print("%d is occupied number.. please use another." % r)
            return

        #print('Move: {} moves to ({})'.format(self.turn(), r))
        self.board[r] = self.turn()
        #print('{}'.format(self))

    def legal_moves(self):
        """
        Returns a list of the legal actions from the current state,
        where an action is the placement of a marker 'X' or 'O' on a board
        position, represented as a (row, col) tuple, for example:
          [(2, 1), (0, 0)]
        would indicate that the positions (2, 1) and (0, 0) are available to
        place a marker on. If the game is in a terminal state, returns an
        empty list.
        """
        # Check if terminal state
        if self.winner() is not None:
            return []

        possible_moves = []
        for row in range(10)[1:]:
            if self.board[row] == ' ':
                possible_moves.append(row)

        return possible_moves


    def transition_function(self, r):
        """
        Applies the specified action to the current state and returns the new
        state that would result. Can be used to simulate the effect of
        different actions. The action is applied to the player whose turn
        it currently is.
        :return: The resulting new state that would occur
        """
        # Verify that the specified action is legal
        assert (r) in self.legal_moves()

        # First, make a copy of the current state
        new_state = copy.deepcopy(self)

        # Then, apply the action to produce the new state
        new_state.move(r)

        return new_state

    def winner(self):
        """
        Checks if the game state is a terminal state.
        :return: If it is not, returns None; if it is, returns 'X' or 'O'
        indicating who is the winner; if it is a tie, returns 'Tie'
        """
        for player in ['X', 'O']:
            # Check for winning vertical lines
            for cols in [ [1,4,7], [2,5,8], [3,6,9]      ]:
                accum = 0
                for col in cols:
                    if self.board[col] == player:
                        accum += 1
                if accum == 3:
                    return player

            # Check for winning horizontal lines
            for rows in [ [1,2,3], [4,5,6], [7,8,9] ] :
                accum = 0
                for row in rows:
                    if self.board[row] == player:
                        accum += 1
                if accum == 3:
                    return player

            # Check for winning diagonal lines (there are 2 possibilities)
            option1 = [self.board[1],
                       self.board[5],
                       self.board[9]]
            option2 = [self.board[3],
                       self.board[5],
                       self.board[7]]
            if all(marker == player for marker in option1) \
                    or all(marker == player for marker in option2):
                return player

        # Check for ties, defined as a board arrangement in which there are no
        # open board positions left and there are no winners (note that the
        # tie is not being detected ahead of time, as could potentially be
        # done)
        accum = 0
        for row in range(10)[1:]:
            if self.board[row] == ' ':
                accum += 1
        if accum == 0:
            return 'Tie'

        return None


def main():

    gs = GameState()

    for i in range(10)[1:]:
        gs.move(i)

if __name__ == "__main__":
    main()
