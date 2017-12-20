# !/usr/bin/env python
# solver16.py -15 puzzle problem using astar:
#
# Abstraction:
# Initial State - Board provided by command line,can be any board in a file as
# 1 2 3 4
# 5 6 7 8
# 9 10 12 0
# 13 14 11 15.
# Goal State - [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]].Board tiles in correct sequence.
# Successors - 1,2 or 3 moves in left, right, up or down direction at a time.It will give 6 successors for each state.
# Cost Function - for each move ie. 1,2 or 3 moves, cost is 1
# Heuristic Function - Manhattan distance/3 + sum(linear conflicts(in each row and column))
#
# There are three main parts in this problem-generating successors,heuristic calculation and algorithm.
# 1)Successors-
# To generate successors,first check what is the position of empty tile.
# Generate new board with moves of tiles ie 1,2 or 3 moves left,right,up and down of that empty tile.
# While generating successors each state is saved as a list containing elemens -heuristic function(f(s)=g(s)+h(s),cost(g(s)),path it takes,new_successor_board.
# Heuristic function calculates manhattan distance with linear conflict.
#
# 2)Heuristic-Used (Manhattan distance)/3 with Linear Conflict algorithm.
# This is admissible and consistent.Since we are considering that there can be 1, 2 or 3 tiles
# that can move at a time,in any 4 directions,and if we assume that each tile is independent of others then,
# manhattan distance will give distance for each tile to goal position.
# This distance will never over estimate the actual cost considering it can move max 3 tiles at a time,
# therefore we can use - manhattan distance/(max no. tiles that can move).
# Linear Conflict-For each row and column if there is a pair of tiles which belongs to same
# row or column and position of tile 1 is left of tile 2 but its goal position is to right
# of tile 2 then we say that this pair of tiles is in linear conflict and it takes more than number of moves calculated from
# manhattan distance, to remove this conflict,we add 2 to actual manhattan distance,considering each tile will take two extra moves to remove this conflict.
# This function ie manhattan distance/3 +linear conflict will never overestimate the actual number of moves taken to reach goal state.[1]
#
# 3)Algorithm-
#
# 1]There are two lists maintained.One is visited nodes-fringe_closed and second is not-visited nodes-fringe_open
# 2]Fringe_open is a priority queue which uses heapq,to maintain priority according to first index ie heuristic function(f(s)=g(s)+h(s)).
# 3]State consists of a list which includes-heuristic function(f(s)=g(s)+h(s),cost(g(s),path it takes,its board(list of list)).[2]
# 4]Once board from file is retrieved we check if it is solvable or not.If it is, then we do following steps.
# 5]First,it takes initial_board from a file and checks if it is goal state.
# 6]If not,it adds this state in fringe_open.
# 7]Repeat till fringe_open is not empty
# 	-Pop from heap state with min heuristic function
# 	-checks if this node is goal node or not
# 	-checks if its not in fringe_closed,then adds in fringe_closed, ie our visited nodes list or else ignores this state and moves to next state in fringe_open.
# 	-Then generate all the successors of this state and push in fringe_open which is priority queue(heapq).
#
# Difficulties faced in algorithm-
# 1]According to Algorithm 3-We check for every successors that if it is already in fringe_open and its heuristic function is less than that in fringe_open,
# remove from fringe_open and append this successor in fringe_open.We have removed this check as it was reducing our code's time efficiency without affecting its optimality.
# This is because heapify is used to remove any element from heap if above condition is satisfied, this consumes more time than expected.
# 2]According to Algorithm 3-We check if any generated successor is already present in fringe_closed(visited state).
# This check is done in our algorithm when we pop a state from fringe_open.This has made our algorithm more efficient in time,
# as we won't need to check this condition for every successors of a state.
# 3]Earlier we were using sort() function to pop a state with minimum heuristic function(f(s)=h(s)+g(s)) from the fringe,
# using heapq(priority queue) significantly reduced our computation time as it does push and pop according to priority.
#
# References-
# [1]We have referred github and blog for understanding linear conflict algorithm.
# -https://algorithmsinsight.wordpress.com/graph-theory-2/a-star-in-general/implementing-a-star-to-solve-n-puzzle/
# -https://github.com/jDramaix/SlidingPuzzle/blob/master/src/be/dramaix/ai/slidingpuzzle/server/search/heuristic/LinearConflict.java
#  Also had a discussion with Akshay Naik about the vertical conflict condition of checking condition,if tile is in same column.
# [2]In solve() we are not checking if a successor is already in fringe_open.This check is put after we pop from fringe_open a state rather than after we get successor.
# This reduced our computation time significantly.Had a discussion with Akshay Naik about how this change has reduce computation time and increased.

import sys
import copy
import heapq
import math

#calculate heuristic f(s)=g(s)+h(s)
def calculate_heuristic(board,cost):
    mandist = 0
    for r in range(0,4):
        for c in range(0,4):
            if board[r][c]==0:continue
            mandist=mandist+(abs(r-(board[r][c]-1)/4)+abs(c-(board[r][c]-1)%4))
    #returns manhattan distance/3+linearconflict+cost
    return int(math.ceil(float(mandist)/3))+verticalconflict(board)+horizontalconflict(board)+cost

# horizontal linear conflicts
def horizontalconflict(board):
    linearconflict = 0
    for r in range(0, 4):
        max = -1
        for c in range(0, 4):
            #checks if that tile is not zero, belongs to same row and then checks for linear conflict in each row
            if board[r][c] != 0 and (((board[r][c] - 1) / 4) == r):
                if board[r][c] > max:
                    max = board[r][c]
                else:
                    linearconflict += 2
    return linearconflict

#We have referred github and blog for understanding linear conflict algorithm.
# -https://algorithmsinsight.wordpress.com/graph-theory-2/a-star-in-general/implementing-a-star-to-solve-n-puzzle/
# -https://github.com/jDramaix/SlidingPuzzle/blob/master/src/be/dramaix/ai/slidingpuzzle/server/search/heuristic/LinearConflict.java
#Also had a discussion with Akshay Naik about the vertical conflict condition of checking condition,if tile is in same column.
# vertical linear conflicts
def verticalconflict(board):
    linearconflict = 0
    for c in range(0, 4):
        max = -1
        for r in range(0, 4):
            #checks if that tile is not zero, belongs to same column and then checks for linear conflict in each column
            if board[r][c] != 0 and ((board[r][c]) % 4) == (c + 1) % 4:
                if board[r][c] > max:
                    max = board[r][c]
                else:
                    linearconflict += 2
    return linearconflict

#get all the succesors of a state in row for move in left side
def get_statesrowleft(state1,r,c,row):
    state=copy.deepcopy(state1)
    new_state= state[0:r] +[state[r][0:c] + [0, ] + state[r][c + 1:]] + state[r + 1:]
    temp=new_state[r][c]
    while c>row:
        new_state[r][c]=new_state[r][c-1]
        c-=1
    new_state[r][row]=temp
    return new_state

#get all the succesors of a state in row for move in right side
def get_statesrowright(state1,r,c,row):
    state=copy.deepcopy(state1)
    new_state= state[0:r] +[state[r][0:c] + [0, ] + state[r][c + 1:]] + state[r + 1:]
    temp=new_state[r][c]
    while c<row:
        new_state[r][c]=new_state[r][c+1]
        c+=1
    new_state[r][row]=temp
    return new_state

#get all the succesors of a state in column for move up
def get_statescolumnup(board,r,c,col):
    board = copy.deepcopy(board)
    new_state= board[0:r] +[board[r][0:c] + [0, ] + board[r][c + 1:]] + board[r + 1:]
    temp = new_state[r][c]
    while r>col:
        new_state[r][c] = new_state[r-1][c]
        r -= 1
    new_state[col][c] = temp
    return new_state

#get all the succesors of a state in column for move down
def get_statescolumndown(board,r,c,col):
    board = copy.deepcopy(board)
    new_state= board[0:r] +[board[r][0:c] + [0, ] + board[r][c + 1:]] + board[r + 1:]
    temp=new_state[r][c]
    while r<col:
        new_state[r][c]=new_state[r+1][c]
        r+=1
    new_state[col][c]=temp
    return new_state

#generates successors states for up down left right moves,and stores in list of list
#each state has heuristic(h(s)) and cost (g(s) , move it takes associated with it.
def pathstate(board,r,c,cost,path):
    succ=[]
    for row in range(0, 4):
        #for move to the right of empty tile
        if row < c:
            state = get_statesrowleft(board, r, c, row)
            s = ["R" + str(abs(c - row)) + str(r + 1)]  # row
            succ.append([calculate_heuristic(state,cost),cost,path+s,state])
        #for move to the left of empty tile
        elif row > c:
            state = get_statesrowright(board, r, c, row)
            s = ["L" + str(abs(row - c)) + str(r + 1)]
            succ.append([calculate_heuristic(state, cost), cost,path+s ,state])
        #for move down of empty tile
        if row < r:
            state1 = get_statescolumnup(board, r, c, row)
            s = ["D" + str(abs(r - row)) + str(c + 1)]  # col
            succ.append([calculate_heuristic(state1, cost), cost,path+s,state1])
        #for move up of empty tile
        elif row > r:
            state1 = get_statescolumndown(board, r, c, row)
            s = ["U" + str(abs(row - r)) + str(c + 1)]
            succ.append([calculate_heuristic(state1, cost), cost,path+s,state1])
    return succ

# this function finds where empty tile is i.e. where 0 is and calls pathstate which generates all the successor states
def successors(board,cost,path):
    for r in range(0,4):
        for c in range(0,4):
            if board[r][c]==0:
                return pathstate(board,r,c,cost,path)

#checks if a state is already in fringe or not
def in_fringeclose(s,fringe):
    for state in fringe:
        if state[3] == s[3]:
            return True
    return False

#check if given board is solvable or not
def is_solvable(board):
    parity=0
    blankrow=0
    for r in range(0,4):
        for c in range(0,4):
            if board[r][c]==0:
                blankrow=3-r
    list = [item for sublist in board for item in sublist]
    for i in range(0,len(list)):
        for j in range(i+1,len(list)):
            if list[j]<list[i] and list[j]!=0:
                parity+=1
    if blankrow%2==0:
        return parity%2==0
    else:
        return parity%2 !=0

#In solve() we are not checking if a successor is already in fringe_open.This check is put after we pop from fringe_open a state rather than after we get successor.
# This reduced our computation time significantly.Had a discussion with Akshay Naik about how this change has reduce computation time and increased.
#algorithm to solve 15 puzzle problem
def solve(board):
    fringe_open=[]
    fringe_closed=[]
    if board==goal:
        return board
    #push in the fringe_open which has all not visited list
    heapq.heappush(fringe_open,[0,0,[],board])
    while len(fringe_open)>0:
        #pops a state with least heuristic value
        state = heapq.heappop(fringe_open)
        #if this popped state is already in closed fringe ie is already visited then we wont add it in fringe
        if in_fringeclose(state, fringe_closed):
            continue
        #add to list fringe_closed which has all the visited nodes
        if state[3]==goal:
            return state
        fringe_closed.append(state)
        #generate all the successors and pass arguments-path,cost and board to find its successors
        for s in successors(state[3], state[1] + 1,state[2]):
            #push all the successors generated in fringe_open
            heapq.heappush(fringe_open,s)

goal=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
#gets the board from the file through command line argument and converts to list of list
board=sys.argv[1]
initial_board = [map(int,line.split()) for line in open(board)]
#solve 15 puzzle problem.
#checks if board is 4X4 board
try:
    if is_solvable(initial_board):
        succ=solve(initial_board)
        # print path initial board takes to reach goal
        for r in range(0,len(succ[2])):
            print (succ[2])[r],
    else:print "Board is not solvable"
except IndexError:
    print "Input Format is not proper"
