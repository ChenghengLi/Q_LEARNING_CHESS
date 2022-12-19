#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022
@author: ignasi
"""
import copy
import math
import chess
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations
from state import State

class Aichess():
    """
    A class to represent the game of chess.
    ...
    Attributes:
    -----------
    chess : Chess
        represents the chess game
    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece
    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.depthMax = 4
        self.checkMate = False

    def getTurn(self):
        return self.chess.turn

    def setTurn(self, turn):
        self.chess.setTurn(turn)

    def getCurrentStateW(self):
        return self.chess.boardSim.currentStateW

    def getCurrentStateB(self):
        return self.chess.boardSim.currentStateB

    def getListNextStates(self, stateW, stateB, player, verbose = False):
        if player:
            assert sorted(stateW) == sorted(self.getCurrentStateW())
            if verbose:
                print(stateW)
                print(self.getCurrentStateW())
            self.chess.boardSim.getListNextStatesW(stateW)
            self.listNextStates = self.chess.boardSim.listNextStates.copy()
            a = self.listNextStates
            return self.listNextStates

        else:
            assert sorted(stateB) == sorted (self.getCurrentStateB())
            self.chess.boardSim.getListNextStatesB(stateB)
            self.listNextStates = self.chess.boardSim.listNextStates.copy()
            a = self.listNextStates
            return self.listNextStates



    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):
        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))
            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedStates)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True
            return isVisited
        else:
            return False

    # Funció que determina si un estat és Check
    def isCheck_1(self, state, player = True):

        stateB = state.stateB
        stateW = state.stateW

        def checkPosition(child):
            positions = set()
            for piece in child:
                (x, y) = piece[0:2]
                if (x, y) in positions:
                    return False
                positions.add((x, y))
            return True

        oponent_king = self.getKing(stateB if player else stateW)

        children = self.getListNextStates(stateW, stateB, player)
        # Per totes les posibles jugades, s'ha de poder arribar al rey rival
        for child in children:
            if not checkPosition(child):
                continue
            mymoves = set((x[0],x[1]) for x in child)
            if oponent_king in mymoves:
                return True

        return False

    def getKing(self, state):
        for i in state:
            if i[2] == 6 or i[2] == 12:
                return tuple(i[0:2])

    def isCheckMate(self, state, player=True):

        if not self.isCheck_1(state, player):
            return False

        stateW = state.stateW
        stateB = state.stateB

        o_children = self.getListNextStates(stateW, stateB, not player)

        # Tots els possibles moviments del rival

        for o_child in o_children:
            if not self.checkPositions(stateW, stateB, o_child, not player):
                continue
            o_kill, o_nState = self.moveSim(stateW, stateB, o_child, not player)
            if o_kill:
                if self.checkKing(o_nState, not player):
                    self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                    continue

                sW = o_nState if player else stateW
                sB = stateB if player else o_nState

            else:
                sW = stateW if player else o_child
                sB = stateB if not player else o_child

            newState = State(sW, sB, None, -1, player)
            if self.isCheck_1(newState, player):
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                continue
            else:
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                return False
        return True


    def getKingSurrondings(self, king_y, king_x):
        king_surrondings = {(king_y - 1, king_x - 1), (king_y - 1, king_x), (king_y - 1, king_x + 1),
                            (king_y, king_x - 1), (king_y, king_x + 1), (king_y + 1, king_x - 1),
                            (king_y + 1, king_x), (king_y + 1, king_x + 1)}
        return king_surrondings

    def evaluate(self, state, player = True):
        player = self.player

        stateW = state.stateW
        stateB = state.stateB

        oponent_state = stateB if player else stateW
        state_1 = stateW if player else stateB

        value = 0
        o_rook_y, o_rook_x, rook_y, rook_x = None, None, None, None
        # Material count
        for i in state_1:
            if i[2] == 2 or i[2] == 8:
                value += 100
                rook_y, rook_x = i[0:2]
            elif i[2] == 6 or i[2] == 12:
                value += 1000
                king_y, king_x = i[0:2]

        for i in oponent_state:
            if i[2] == 2 or i[2] == 8:
                value -= 100
                o_rook_y, o_rook_x = i[0:2]
            elif i[2] == 6 or i[2] == 12:
                value -= 1000
                o_king_y, o_king_x = i[0:2]

        # Mobility
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
        value += len(children)

        children = self.getListNextStates(stateW, stateB, not player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
        value -= len(children)


        # Checks
        if self.isCheck_1(state, player):
            value += 10

        if self.isCheck_1(state, not player):
            value -= 10

        def manhattan_distance(piece_1, piece_2):
            return abs(piece_1[0] - piece_2[0]) + abs(piece_1[1] - piece_2[1])

        def get_dist_to_center(piece):
            center = {(3, 3), (3, 4), (4, 3), (4, 4)}
            return max([manhattan_distance(piece, x) for x in center])

        value += get_dist_to_center((o_king_y, o_king_x))

        value -= get_dist_to_center((king_y, king_x))

        if rook_x == None and o_rook_x == None:
            return 0

        if rook_x == None and o_rook_y != None:
            # get far from the the opponent king
            value -= manhattan_distance((o_king_y, o_king_x), (king_y, king_x))
            # get close to the rook
            value += manhattan_distance((o_king_y, o_king_x), (o_rook_y, o_rook_x))

        if rook_x != None and o_rook_x == None:
            #get close to the king our king
            value += manhattan_distance((king_y, king_x), (king_y, king_x))
            # get far from the king our rook
            value -= manhattan_distance((king_y, king_x), (rook_y, rook_x))


        return value


    # In order to eliminate invalid states
    def checkPositions(self, stateW, stateB = None, child = None, player = True):

        king_y, king_x, o_king_y, o_king_x = None, None, None, None

        if player:
            for i in child:
                if i[2] == 6:
                    king_y, king_x = i[0:2]
            for i in stateB:
                if i[2] == 12:
                    o_king_y, o_king_x = i[0:2]
            kS = self.getKingSurrondings(o_king_y, o_king_x)
            if (king_y, king_x) in kS:
                return False

        else:
            for i in child:
                if i[2] == 12:
                    king_y, king_x = i[0:2]
            for i in stateW:
                if i[2] == 6:
                    o_king_y, o_king_x = i[0:2]
            kS = self.getKingSurrondings(o_king_y, o_king_x)
            if (king_y, king_x) in kS:
                return False

        positions = [(o_king_y, o_king_x)]
        for piece in child:
            (x, y) = piece[0:2]
            if (x, y) in positions:
                return False
            positions.append((x, y))
        return True

    def tupleSort(self, stateW, stateB):
        l = stateW + stateB
        return tuple(tuple(i) for i in sorted(l))

    def listSort(self, state, player):
        if player:
            return list(list(i) for i in sorted(state) if i[2] >= 0 and i[2] <= 6)
        else:
            return list(list(i) for i in sorted(state) if i[2] > 6 and i[2] < 13)

    def moveSim(self, stateW, stateB, nextState, player = True):
        currentState = stateW if player else stateB

        (x_1, y_1) = [e for e in currentState if e not in nextState][0][0:2]
        (x_2, y_2) = [e for e in nextState if e not in currentState][0][0:2]

        kill = False
        nS = None

        if player:
            kill, nS = self.chess.moveSimW([x_1, y_1], [x_2, y_2])
        else:
            kill, nS = self.chess.moveSimB([x_1, y_1], [x_2, y_2])

        return kill, nS

    def move(self, stateW, stateB, nextState, player = True):
        currentState = stateW if player else stateB

        (x_1, y_1) = [e for e in currentState if e not in nextState][0][0:2]
        (x_2, y_2) = [e for e in nextState if e not in currentState][0][0:2]

        self.chess.move([x_1, y_1], [x_2, y_2])


    def undoMovement(self, stateW, stateB, child, kill, player = True):
        if kill:
            self.chess.updateTable(stateW, stateB)
            return
        if player:
            self.moveSim(child, stateB, stateW, player)
            return
        else:
            self.moveSim(stateW, child, stateB, player)
            return

    def checkKing(self, state, player):
        king_code = 12 if player else 6
        return king_code not in set(x[2] for x in state)

    # Q-Learning
    def q_learning(self, state, player):
        stateW = state.stateW
        stateB = state.stateB
        self.resetTable(self.chess.board)

        #print(stateW)
        #print(self.getCurrentStateW())
        board = copy.deepcopy(self.chess.boardSim)
        # Constantes
        gamma = 0.9  # Constante de Disminución
        alpha = 0.05  # Constante de Aprendizaje
        delta_list = []

        # Inicialización de la tabla de Q-values
        Q = dict()
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

        Q[self.tupleSort(stateW, stateB)] = dict()

        for child in children:
            Q[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB if player else stateW)] = 0


        iter = 1

        while True:

            # Per anar reduint l'aleatorietat a mesura que va aprenent
            iter += 1
            ct = 1/iter

            children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
            a = random.random()

            # Elegimos el hijo de mayor Q-value
            if a > ct:
                v, child = self.get_maxStates(Q, stateW, stateB, player)
            # Elegimos un hijo al azar
            else:
                child = random.choice(children)

            kill, nState = self.moveSim(stateW, stateB, child, player)
            if kill:  # Ha matat al rei
                self.undoMovement(stateW, stateB, child, kill, player)
                continue

            # self.chess.boardSim.print_board()


            if player:
                newState = State(child, stateB, None, iter, player)
            else:
                newState = State(stateW, child, None, iter, player)

            # Recompensa del nuevo estado
            # print(newState)
            r = self.recompensa(newState, player)

            # Máximo de los Q-values des del hijo
            if player:
                maxim, fill = self.get_maxStates(Q, child, stateB, player)
                q = Q[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB)]
            else:
                maxim, fill = self.get_maxStates(Q, stateW, child, player)
                q = Q[self.tupleSort(stateW, stateB)][self.tupleSort(stateW, child)]

            delta = r + gamma*maxim - q
            delta_list.append(delta)

            Q[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB if player else stateW)] = q + alpha * delta

            # Si hem fet checkmate, fem reset del tauler i dels estats
            if r == 100:
                # Calculem l'error del camí
                max_delta = max(delta_list)
                if max_delta < pow(10, -9):
                    break
                delta_list = []
                #self.chess.boardSim.print_board()
                stateW = state.stateW
                stateB = state.stateB
                self.resetTable(board)
                children = self.getListNextStates(stateW, stateB, player)
                continue

            stateW = newState.stateW
            stateB = newState.stateB
            children = self.getListNextStates(stateW, stateB, player)  # not player, per canviar de jugador
            # player = not player

        print("----END----")
        self.resetTable(board)
        return Q


    def resetTable(self, board):
        # Funció per fer reset al tauler i tornar a començar la iteració
        self.chess.boardSim = copy.deepcopy(board)


    def get_maxStates(self, Q, stateW, stateB, player = True):
        if self.tupleSort(stateW, stateB) in Q:
            m = max(Q[self.tupleSort(stateW, stateB)].items(), key = lambda x: x[1])
            maxim = m[1]
            fill = self.listSort(m[0], player)
            return maxim, fill
        else:
            children = self.getListNextStates(stateW, stateB, player)
            Q[self.tupleSort(stateW, stateB)] = dict()
            children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

            for child in children:
                Q[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB if player else stateW)] = 0

            return 0, random.choice(children)


    def recompensa(self, state, player):
        # Abierto a otras sugerencias
        if self.isCheckMate(state, player):
            return 100
        else:
            return -1

    #Hace falta que aprenda a la vez no turnandose (Ya lo acabo yo sino)
    '''
    Pseudocode
    
    import numpy as np
    
    # Define the environment
    state_space = [1, 2, 3, 4]  # Possible states
    action_space = [0, 1]  # Possible actions
    reward_function = {
        (1, 0): 0,
        (1, 1): 1,
        (2, 0): 1,
        (2, 1): 0,
        (3, 0): 0,
        (3, 1): 1,
        (4, 0): 1,
        (4, 1): 0
    }  # Rewards for (state, action) pairs
    
    # Initialize Q-tables for both agents
    Q1 = np.zeros((len(state_space), len(action_space)))
    Q2 = np.zeros((len(state_space), len(action_space)))
    
    # Set hyperparameters
    num_episodes = 1000  # Number of episodes to run
    learning_rate = 0.1  # Learning rate for Q-learning update
    discount_factor = 0.9  # Discount factor for future rewards
    epsilon = 0.1  # Probability of choosing a random action (exploration)
    
    # Loop through the episodes
    for episode in range(num_episodes):
        # Reset the environment for the start of the episode
        state1 = np.random.choice(state_space)  # Initial state for agent 1
        state2 = np.random.choice(state_space)  # Initial state for agent 2
    
        while True:  # Loop until the episode ends
            # Select actions for both agents
            if np.random.uniform(0, 1) < epsilon:  # Explore
                action1 = np.random.choice(action_space)
                action2 = np.random.choice(action_space)
            else:  # Exploit
                action1 = np.argmax(Q1[state1, :])
                action2 = np.argmax(Q2[state2, :])
    
            # Take actions and get rewards
            next_state1, reward1 = state2, reward_function[(state1, action1)]
            next_state2, reward2 = state1, reward_function[(state2, action2)]
    
            # Update Q-tables
            Q1[state1, action1] = (1 - learning_rate) * Q1[state1, action1] + learning_rate * (reward1 + discount_factor * np.max(Q1[next_state1, :]))
            Q2[state2, action2] = (1 - learning_rate) * Q2[state2, action2] + learning_rate * (reward2 + discount_factor * np.max(Q2[next_state2, :]))
    
            # Check if episode has ended
            if state1 == state2 and action1 == action2:  # Terminal state
                break
    
            # Update states
            state1, state2 = next_state1, next_state2
        
        '''
    def q_learning_2(self, state, player):
        stateW = state.stateW
        stateB = state.stateB
        depth = state.depth
        self.resetTable(self.chess.board)

        #print(stateW)
        #print(self.getCurrentStateW())
        board = copy.deepcopy(self.chess.boardSim)
        # Constantes
        gamma = 0.9  # Constante de Disminución
        alpha = 0.2  # Constante de Aprendizaje
        delta = 1  # Error

        # Inicialización de la tabla de Q-values
        Q_W = dict()
        Q_B = dict()
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

        Q_W[self.tupleSort(stateW, stateB)] = dict()

        for child in children:
            Q_W[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB if player else stateW)] = 0

        start = True

        while abs(delta) > pow(10, -7):

            children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

            a = random.random()

            # Elegimos el hijo de mayor Q-value
            if start:
                child = random.choice(children)
                start = False
            elif a > 0.2:
                v, child = self.get_maxStates(Q_W if player else Q_B, stateW, stateB, player)
            # Elegimos un hijo al azar
            else:
                child = random.choice(children)

            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:

                newStateW = child if player else nState
                newStateB = nState if player else child
                newState = State(newStateW, newStateB, state, depth, not player)

                if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

            # self.chess.boardSim.print_board()
            depth += 1

            # Recompensa del nuevo estado
            # print(newState)
            r = self.recompensa_2(newState, player)

            # Máximo de los Q-values des del hijo
            if player:
                maxim, fill = self.get_maxStates(Q_W, child, stateB, player)
                q = Q_W[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB)]
                delta = r + gamma * maxim - q
                Q_W[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB)] = q + alpha * delta
            else:
                maxim, fill = self.get_maxStates(Q_B, stateW, child, player)
                q = Q_B[self.tupleSort(stateW, stateB)][self.tupleSort(stateW, child)]
                delta = r + gamma * maxim - q
                Q_B[self.tupleSort(stateW, stateB)][self.tupleSort(stateW, child)] = q + alpha * delta


            # Si hem fet checkmate, fem reset del tauler i dels estats
            if r == 100:
                player = True
                start = True
                depth = 0
                #self.chess.boardSim.print_board()
                stateW = state.stateW
                stateB = state.stateB
                self.resetTable(board)
                children = self.getListNextStates(stateW, stateB, player)
                continue

            stateW = newState.stateW
            stateB = newState.stateB
            player = not player
            children = self.getListNextStates(stateW, stateB, player)


        print("----END----")
        self.resetTable(board)
        return Q_W, Q_B

    def recompensa_2(self, state, player):
        # Abierto a otras sugerencias
        if player:
            if self.isCheckMate(state, player):
                return 100
            else:
                return -1
        else:
            if self.isCheckMate(state, not player):
                return 100
            else:
                return -1



def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


import random


def getKingSurrondings(king_y, king_x):
    king_surrondings = {(king_y - 1, king_x - 1), (king_y - 1, king_x), (king_y - 1, king_x + 1),
                        (king_y, king_x - 1), (king_y, king_x + 1), (king_y + 1, king_x - 1),
                        (king_y + 1, king_x), (king_y + 1, king_x + 1)}
    return king_surrondings

def generateBoard():
    """
    Generates a random board
    """
    board = np.zeros((8, 8))
    # Generate random pieces
    kS = set()
    for i in (6,2,12,8):
        x = random.randint(0, 7)
        y = random.randint(0, 7)
        if i == 6:
            kS = getKingSurrondings(y, x)
        while board[x][y] != 0 or (y, x) in kS:
            x = random.randint(0, 7)
            y = random.randint(0, 7)
            if i == 6:
                kS = getKingSurrondings(y, x)
        board[x][y] = i
    return board


if __name__ == "__main__":

    print("stating AI chess... ")
    TA = np.zeros((8, 8))

    TA[0][5] = 12
    TA[7][5] = 6
    TA[7][0] = 2

    aichess = Aichess(TA, True)



    aichess.chess.board.print_board()
    # initialize board
    '''
    for i in range(10000):
        
        # initialise board

        currentStateW = copy.deepcopy(aichess.chess.board.currentStateW)
        currentStateB = copy.deepcopy(aichess.chess.board.currentStateB)
        
        oK = list()
        for i in currentStateB:
            if i[2] == 12:
                oK = i[0:2]
        oK = tuple(oK)

        a = State(currentStateW, currentStateB, None, 0, True)
        cmW = aichess.isCheckMate(a, True)
        if cmW:
            print("white checkmate")
            aichess.chess.board.print_board()
        cm = aichess.isCheckMate(a, not True)
        if cm:
            print("black checkmate")
            aichess.chess.board.print_board()
        '''
    currentStateW = copy.deepcopy(aichess.chess.board.currentStateW)
    currentStateB = copy.deepcopy(aichess.chess.board.currentStateB)
    state = State(currentStateW, currentStateB, None, 0, True)
    depth = 0


    # Exercici 1
    Q = aichess.q_learning(state, True)

    # PARTIDA:
    
    while not aichess.isCheckMate(state, True):
        currentStateW = state.stateW
        currentStateB = state.stateB
        pare = aichess.tupleSort(currentStateW, currentStateB)
        maxim = -float("inf")
        child = aichess.listSort(max(Q[pare].items(), key = lambda x: x[1])[0], True)
        aichess.moveSim(currentStateW, currentStateB, child, True)
        aichess.move(currentStateW, currentStateB, child, True)
        depth += 1
        aichess.chess.board.print_board()
        state = State(child, currentStateB, state, depth, True)
        print(state, aichess.getCurrentStateW())


    '''
    # Exercici 2
    Q_W, Q_B = aichess.q_learning_2(state, True)

    # PARTIDA:

    player = True

    while not aichess.isCheckMate(state, player):
        Q = Q_W if player else Q_B

        currentStateW = state.stateW
        currentStateB = state.stateB
        pare = aichess.tupleSort(currentStateW, currentStateB)
        maxim = -float("inf")
        child = aichess.listSort(max(Q[pare].items(), key = lambda x: x[1])[0], player)
        aichess.moveSim(currentStateW, currentStateB, child, player)
        aichess.move(currentStateW, currentStateB, child, player)
        depth += 1
        aichess.chess.board.print_board()
        state = State(child, currentStateB, state, depth, player)
        player = not player
        print(state)

    '''

