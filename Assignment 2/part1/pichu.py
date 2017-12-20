#!/usr/bin/python
"""
Authors : Ninaad Joshi and Aishwarya Dhage

Abstraction:

1.Initial State-Board provided by command line.(We have used one-dimensional
list to represent board)
2.Successors-all the valid moves of opponent for each action of a particular
player.
3.Evaluation Function-It considers mobility score, material score and parakeet
positions.
4.Max player - It considers initial player as max player and plays first. Both
max and min alternate actions
5.Terminal State-The state where only one king of any side(black or white)
remains.

There are three main parts of this problem- Successors, evaluation function and
alpha-beta pruning algorithm
1)Successors-The successors were generated using the property of indices for a
matrix where the indices of the pieces are considered for their valid moves.
Division and Mod operators have been liberally used to achieve the required
values for the successors. The moves for every piece on the pichu board are
calculated by  functions designed for every piece. These functions take the
current board, the piece and its position as input and return the next valid
position for the successors.

2)Evaluation Function[1][4]-

We have used Material Score that is we have assigned weights for each piece.
Weights are-Kingfisher=200, for quetzal=9, for robin=5, for nighthawk and
blue jay= 3, for parakeet=1.Weights are assigned according to the importance of
each piece.Sum of all the pieces on board associated with its weights is
material score.Total material score is the difference of material score for
black player and white player.
To enhance its efficiency we have used Mobility Score, which gives the sum of
all valid moves of each piece on its given position.It basically tells us that
if it has more moves, its position is better.Total mobility score is the
difference of mobility score for black player and white player.
Also, evaluation of parakeets helps in increasing efficiency of evaluation
function.We have considered three types of parakeet positions which can make a
side(black or white) weak-
-Doubled Parakeet-A parakeet in the same file of another parakeet, blocks that
parakeet.
-Isolated Parakeet-A parakeet with no parakeets on its either side of file.
-Blocked Parakeet-A parakeet which cannot move in any direction is a blocked
parakeet.
All above mentioned parakeets are bad parakeet positions.And so we have reduced
value of evaluation function, if such pieces are on the board.
The main part of this algorithm is evaluation function.In above evaluation
function below are some ways we tried to implement to enhance its efficiency
more but had difficulties-
-We tried to use tables for each piece.These tables had positional weights
assigned.Eg-a king in the corner is always better than near centre.[2]
-Also to remove redundant board orientation we tired to implement
transpositional tables.[3]

3)Alpha beta pruning algorithm-

Our algorithm assumes that max player plays first(we), followed by min player
(opponent).Also we have considered a depth upto which our algorithm will expand
our search tree.Our algorithm considers maximum depth of 10.
There are three functions in our algorithm-
-alphabeta()-which gets all the beta values from its successors and stores in
max-heap.And to find the next best move we take maximum of all beta values from
heap.This function calls min_value() which calculates beta value for a state.
-maxvalue()-for each successor of given board it recursively calls max-value and
 applies pruning.If it reaches terminating level, evaluation_function() is
 called.It returns maximum from all the min_value()
-minvalue()-for each successor of given board it recursively calls invalid and
applies pruning.If it reaches terminating level, evaluation_function() is
called.It returns minimum from all the max_value()

References-

[1] https://chessprogramming.wikispaces.com/Evaluation
[2]http://chessprogramming.wikispaces.com/Simplified+evaluation+function
[3]https://en.wikipedia.org/wiki/Transposition_table
[4]Discussed evaluation function with ameya angal

#piece_square tables
pawn = [0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0]

rook = [0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0]

queen = [-20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20]

king = [-30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20]

bishop = [-20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20]

knight = [50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50]
"""
import sys
import heapq
import time


# valid moves for the parakeet
def parakeet(board, piece, position):
    next_positions = []
    if piece.isupper() and 8 <= position <= 63:
        if abs((position + 7) % 8 - position % 8) < 2 \
                and board[position + 7].isalpha() \
                and piece.isupper() == board[position + 7].islower():
            # attack moves for white
            next_positions.append(position + 7)
        if abs((position + 9) % 8 - position % 8) < 2 \
                and board[position + 9].isalpha() \
                and piece.isupper() == board[position + 9].islower():
            # attack moves for white
            next_positions.append(position + 9)
        if position / 8 == 1:
            if board[position + 8] == ".":
                next_positions.append(position + 8)
                if board[position + 16] == ".":
                    next_positions.append(position + 16)
        else:
            if board[position + 8] == ".":
                next_positions.append(position + 8)
    elif piece.islower() and 0 <= position <= 55:
        if abs((position - 7) % 8 - position % 8) < 2 \
                and board[position - 7].isalpha() \
                and piece.isupper() == board[position - 7].islower():
            # attack moves for black
            next_positions.append(position - 7)
        if abs((position - 9) % 8 - position % 8) < 2 \
                and board[position - 9].isalpha() \
                and piece.isupper() == board[position - 9].islower():
            # attack moves for black
            next_positions.append(position - 9)
        if position / 8 == 6:
            if board[position - 8] == ".":
                next_positions.append(position - 8)
                if board[position - 16] == ".":
                    next_positions.append(position - 16)
        else:
            if board[position - 8] == ".":
                next_positions.append(position - 8)
    return next_positions


# valid moves for the night hawk
def nighthawk(board, piece, position):
    next_positions = []
    for index in [17, 15, 10, 6]:
        if 0 <= position + index <= 63 \
                and (board[position + index].islower() == piece.isupper()
                     or board[position + index] == ".") \
                and abs(position % 8 - abs(position + index) % 8) < 3:
            next_positions.append(position + index)
        if 0 <= position - index <= 63 \
                and (board[position - index].islower() == piece.isupper()
                     or board[position - index] == ".") \
                and abs(position % 8 - abs(position - index) % 8) < 3:
            next_positions.append(position - index)
    return next_positions


# valid moves for the blue jay
def blue_jay(board, piece, position):
    next_positions = []
    for index in range(9, 64, 9):
        if (position + index) % 8 == position % 8:
            break
        if 0 <= position + index <= 63:
            if (board[position + index].islower() == piece.isupper()
                or board[position + index] == ".") \
                    and ((position + index) % 8 > position % 8):
                next_positions.append(position + index)
            if board[position + index].isalpha():
                break
    for index in range(9, 64, 9):
        if (position - index) % 8 == position % 8:
            break
        if 0 <= position - index <= 63:
            if (board[position - index].islower() == piece.isupper()
                or board[position - index] == ".") \
                    and ((position - index) % 8 < position % 8):
                next_positions.append(position - index)
            if board[position - index].isalpha():
                break
    for index in range(7, 64, 7):
        if (position + index) % 8 == position % 8:
            break
        if 0 <= position + index <= 63:
            if (board[position + index].islower() == piece.isupper()
                or board[position + index] == ".") \
                    and ((position + index) % 8 < position % 8):
                next_positions.append(position + index)
            if board[position + index].isalpha():
                break
    for index in range(7, 64, 7):
        if (position - index) % 8 == position % 8:
            break
        if 0 <= position - index <= 63:
            if (board[position - index].islower() == piece.isupper()
                or board[position - index] == ".") \
                    and ((position - index) % 8 > position % 8):
                next_positions.append(position - index)
            if board[position - index].isalpha():
                break
    return next_positions


# valid moves for the robin
def robin(board, piece, position):
    next_positions = []
    for index in range(8, 64, 8):
        if 0 <= position - index <= 63:
            if (board[position - index].islower() == piece.isupper()
                or board[position - index] == ".") \
                    and ((position - index) % 8 == position % 8):
                next_positions.append(position - index)
            if board[position - index].isalpha():
                break
    for index in range(8, 64, 8):
        if 0 <= position + index <= 63:
            if (board[position + index].islower() == piece.isupper()
                or board[position + index] == ".") \
                    and ((position + index) % 8 == position % 8):
                next_positions.append(position + index)
            if board[position + index].isalpha():
                break
    for index in range(1, 8):
        if 0 <= position + index <= 63:
            if (board[position + index].islower() == piece.isupper()
                or board[position + index] == ".") \
                    and ((position + index) / 8 == position / 8):
                next_positions.append(position + index)
            if board[position + index].isalpha():
                break
    for index in range(1, 8):
        if 0 <= position - index <= 63:
            if (board[position - index].islower() == piece.isupper()
                or board[position - index] == ".") \
                    and ((position - index) / 8 == position / 8):
                next_positions.append(position - index)
            if board[position - index].isalpha():
                break
    return next_positions


# valid moves for the quetzal
def quetzal(board, piece, position):
    return robin(board, piece, position) + blue_jay(board, piece, position)


# valid moves for the kingfisher
def kingfisher(board, piece, position):
    next_positions = []
    for index in [1, 7, 8, 9]:
        if 0 <= position - index <= 63:
            if (board[position - index].islower() == piece.isupper()
                or board[position - index] == ".") \
                    and abs((position - index) % 8 - position % 8) < 2:
                next_positions.append(position - index)
        if 0 <= position + index <= 63:
            if (board[position + index].islower() == piece.isupper()
                or board[position + index] == ".") \
                    and abs((position + index) % 8 - position % 8) < 2:
                next_positions.append(position + index)
    return next_positions


def successors(current_board, max_player):
    if max_player:
        new_states = []
        for pos in [pos for pos, piece in enumerate(current_board)]:
            if current_board[pos] == "P":
                for new_max_position in parakeet(current_board, "P", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[
                        new_max_position] = "Q" if new_max_position / 8 == 7 \
                        else "P"
                    new_states.append(temp_board)
            if current_board[pos] == "R":
                for new_max_position in robin(current_board, "R", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "R"
                    new_states.append(temp_board)
            if current_board[pos] == "N":
                for new_max_position in nighthawk(current_board, "N", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "N"
                    new_states.append(temp_board)
            if current_board[pos] == "B":
                for new_max_position in blue_jay(current_board, "B", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "B"
                    new_states.append(temp_board)
            if current_board[pos] == "Q":
                for new_max_position in quetzal(current_board, "Q", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "Q"
                    new_states.append(temp_board)
            if current_board[pos] == "K":
                for new_max_position in kingfisher(current_board, "K", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "K"
                    new_states.append(temp_board)
        return new_states
    else:
        new_states = []
        for pos in [pos for pos, piece in enumerate(current_board)]:
            if current_board[pos] == "p":
                for new_max_position in parakeet(current_board, "p", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[
                        new_max_position] = "q" if new_max_position / 8 == 0 \
                        else "p"
                    new_states.append(temp_board)
            if current_board[pos] == "r":
                for new_max_position in robin(current_board, "r", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "r"
                    new_states.append(temp_board)
            if current_board[pos] == "n":
                for new_max_position in nighthawk(current_board, "n", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "n"
                    new_states.append(temp_board)
            if current_board[pos] == "b":
                for new_max_position in blue_jay(current_board, "b", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "b"
                    new_states.append(temp_board)
            if current_board[pos] == "q":
                for new_max_position in quetzal(current_board, "q", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "q"
                    new_states.append(temp_board)
            if current_board[pos] == "k":
                for new_max_position in kingfisher(current_board, "k", pos):
                    temp_board = list(current_board)
                    temp_board[pos] = "."
                    temp_board[new_max_position] = "k"
                    new_states.append(temp_board)
        return new_states


# evaluation function
def evaluation_function(board, current_player):
    blocked_max = 0
    blocked_min = 0
    m_max = 0
    m_min = 0
    parakeet_position_max = [0] * 8
    parakeet_position_min = [0] * 8
    for position in range(0, 64):
        # blocked parakeets and mobility for parakeets for white
        if board[position] == "P":
            parakeet_position_max[position % 8] += 1
            m_max += len(parakeet(board, "P", position))
            if board[position + 8].isalpha():
                if position % 8 == 7 \
                        and (board[position + 7].isupper()
                             or board[position + 7] == "."):
                    blocked_max += 1
                elif position % 8 == 0 \
                        and (board[position + 9].isupper()
                             or board[position + 9] == "."):
                    blocked_max += 1
                elif 1 <= position % 8 <= 6 \
                        and ((board[position + 7].isupper()
                              and board[position + 9].isupper())
                             or (board[position + 7] == "."
                                 and board[position + 9] == ".")):
                    blocked_max += 1
        # blocked parakeets and mobility for parakeets for black
        if board[position] == "p":
            parakeet_position_min[position % 8] += 1
            m_min += len(parakeet(board, "p", position))
            if board[position - 8].isalpha():
                if position % 8 == 7 \
                        and (board[position - 7].islower()
                             or board[position - 7] == "."):
                    blocked_min += 1
                elif position % 8 == 0 \
                        and (board[position - 9].islower()
                             or board[position - 9] == "."):
                    blocked_min += 1
                elif 1 <= position % 8 <= 6 \
                        and ((board[position - 7].islower()
                              and board[position - 9].islower())
                             or (board[position - 7] == "."
                                 and board[position - 9] == ".")):
                    blocked_min += 1
        # mobility of the pieces for white
        if board[position] == "R":
            m_max += len(robin(board, "R", position))
        elif board[position] == "N":
            m_max += len(nighthawk(board, "N", position))
        elif board[position] == "B":
            m_max += len(blue_jay(board, "B", position))
        elif board[position] == "Q":
            m_max += len(quetzal(board, "Q", position))
        
        # mobility of the pieces for black
        if board[position] == "r":
            m_min += len(robin(board, "r", position))
        elif board[position] == "n":
            m_min += len(nighthawk(board, "n", position))
        elif board[position] == "b":
            m_min += len(blue_jay(board, "b", position))
        elif board[position] == "q":
            m_min += len(quetzal(board, "q", position))
    
    # double parakeets and isolated parakeets for white
    double_max = 0
    isolated_max = 0
    for index, x in enumerate(parakeet_position_max):
        if x == 1:
            if 1 <= index <= 6 and parakeet_position_max[index - 1] == 0 and \
                            parakeet_position_max[index + 1] == 0:
                isolated_max += 1
            if index == 0 and parakeet_position_max[index + 1] == 0:
                isolated_max += 1
            if index == 7 and parakeet_position_max[index - 1] == 0:
                isolated_max += 1
        
        if x > 1:
            double_max += 1
    # double parakeets and isolated parakeets for black
    double_min = 0
    isolated_min = 0
    for index, x in enumerate(parakeet_position_min):
        if x == 1:
            if 1 <= index <= 6 and parakeet_position_min[index - 1] == 0 and \
                            parakeet_position_min[index + 1] == 0:
                isolated_min += 1
            if index == 0 and parakeet_position_min[index + 1] == 0:
                isolated_min += 1
            if index == 7 and parakeet_position_min[index - 1] == 0:
                isolated_min += 1
        if x > 1:
            double_min += 1
    # value contains the evaluated value according to the used evaluation
    # function
    value = 200 * (board.count("K") - board.count("k")) \
            + 9 * (board.count("Q") - board.count("q")) \
            + 5 * (board.count("R") - board.count("r")) \
            + 3 * (board.count("B") - board.count("b")
                   + board.count("N") - board.count("n")) \
            + (board.count("P") - board.count("p")) \
            - 0.5 * (double_max - double_min + blocked_max
                     - blocked_min + isolated_max - isolated_min) \
            + 0.1 * (m_max - m_min)
    
    if current_player:
        return value
    else:
        return -value


# terminal condition where either of the kings is absent on the board
def is_terminal(board):
    return True if "K" not in board or "k" not in board else False


# min-max algorithm
def alpha_beta(board, depth, max_player):
    moves = []
    for successor in successors(board, max_player):
        b = min_value(successor, depth - 1, -100000, 100000, max_player)
        heapq.heappush(moves, [-b, successor])
    return heapq.heappop(moves)


# the max function which maximizes the result alpha values
def max_value(board, depth, alpha, beta, max_player):
    global start_time
    global time_limit
    if time.time() - start_time >= time_limit:
        # timer expired
        exit(0)
    if is_terminal(board) or depth == 0:
        return evaluation_function(board, max_player)
    for s in successors(board, max_player):
        alpha = max(alpha, min_value(s, depth - 1, alpha, beta, max_player))
        if alpha >= beta:
            return alpha
    return alpha


# the min function which minimizes the result beta values
def min_value(board, depth, alpha, beta, max_player):
    global start_time
    global time_limit
    if time.time() - start_time >= time_limit:
        # timer expired
        exit(0)
    if is_terminal(board) or depth == 0:
        return evaluation_function(board, max_player)
    for s in successors(board, not max_player):
        beta = min(beta, max_value(s, depth - 1, alpha, beta, max_player))
        if alpha >= beta:
            return beta
    return beta


try:
    player = str(sys.argv[1])
    input_board_config = list(str(sys.argv[2]))
    time_limit = int(str(sys.argv[3]))
    initial_board = list(input_board_config)
    start_time = time.time()
    for i in range(2, 10):
        if time.time() - start_time <= time_limit:
            a = alpha_beta(input_board_config, i,
                           True if player == "w" else False)
            print "".join(a[1])
        else:
            exit(0)

except ValueError:
    print "Invalid Input. :("
    exit(0)
