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
from tqdm import tqdm
from matplotlib import pyplot as plt

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

    def getListNextStates(self, stateW, stateB, player):
        if player:
            assert sorted(stateW) == sorted(self.getCurrentStateW())
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

            newState = State(sW, sB, player)
            if self.isCheck_1(newState, player):
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                continue
            else:
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                return False
        state.terminal = True
        return True


    def getKingSurrondings(self, king_y, king_x):
        king_surrondings = {(king_y - 1, king_x - 1), (king_y - 1, king_x), (king_y - 1, king_x + 1),
                            (king_y, king_x - 1), (king_y, king_x + 1), (king_y + 1, king_x - 1),
                            (king_y + 1, king_x), (king_y + 1, king_x + 1)}
        return king_surrondings

    # In order to eliminate invalid states
    def checkPositions(self, stateW, stateB = None, child = None, player = True):


        state = State(child if player else stateW, stateB if player else child, player)

        if self.isCheck_1(state, not player):
            return False


        king_y, king_x, o_king_y, o_king_x = None, None, None, None

        if player:
            for i in child:
                if i[2] == 6:
                    king_y, king_x = i[0:2]
            for i in stateB:
                if i[2] == 12:
                    o_king_y, o_king_x = i[0:2]
        else:
            for i in child:
                if i[2] == 12:
                    king_y, king_x = i[0:2]
            for i in stateW:
                if i[2] == 6:
                    o_king_y, o_king_x = i[0:2]



        positions = self.getKingSurrondings(o_king_y, o_king_x)
        positions.add((o_king_y, o_king_x))

        for piece in child:
            (x, y) = piece[0:2]
            if (x, y) in positions:
                return False
            positions.add((x, y))
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

    def isTerminalState(self, state):
        return state.terminal
    # Q-Learning
    def q_learning(self, state, player):
        stateW = state.stateW
        stateB = state.stateB
        convergency = False
        board = copy.deepcopy(self.chess.boardSim)
        # Constantes
        gamma = 0.9  # Constante de Disminución
        alpha = 0.05  # Constante de Aprendizaje
        Q = dict()
        macthes= 100000

        # Inicialización de Q
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

        Q[self.tupleSort(stateW, stateB)] = dict()
        list_epochs = []
        list_error= []
        for child in children:
            Q[self.tupleSort(stateW, stateB)][self.tupleSort(child, stateB if player else stateW)] = 0

        for match in tqdm(range(1,macthes + 1)):
            newState = state
            delta_list = list()
            self.resetTable(board)
            moves = 0

            while not self.isTerminalState(newState):
                moves += 1
                ct = 1/(match*0.01)
                stateW = newState.stateW
                stateB = newState.stateB
                children = self.getListNextStates(stateW, stateB, player)
                children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

                # Epsilon-Greedy
                a = random.random()
                if a > ct:
                    v, child = self.get_maxStates(Q, stateW, stateB, player)
                else:
                    child = random.choice(children)

                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:  # Ha matat al rei
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                # self.chess.boardSim.print_board()

                if player:
                    newState = State(child, stateB, player)
                else:
                    newState = State(stateW, child, player)

                r = self.recompensa(newState, moves, player)

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

            error = max(abs(max(delta_list)), abs(min(delta_list)))
            list_error.append(error)
            list_epochs.append(match)

            if error < pow(10, -2):
                convergency =  True
                break

        print("----END----")
        print("Convergency: ", convergency)
        print("Delta: ", error)
        print("Matches: ", match)
        print("--------------------------")
        # Graficar
        plt.style.use('seaborn')
        plt.title("Gráfica de los deltas")
        plt.scatter(list_epochs, list_error, s=1)
        plt.xticks(np.arange(min(list_epochs)-1, max(list_epochs) + 1, int(match/15) ))
        plt.yticks(np.arange(0, max(list_error) + 1, 10))
        plt.show()
        self.resetTable(board)
        return Q


    def resetTable(self, board):
        # Funció per fer reset al tauler i tornar a començar la iteració
        self.chess.boardSim = copy.deepcopy(board)


    def get_maxStates(self, Q, stateW, stateB, player):

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


    def recompensa(self, state, moves,  player):
        if self.isCheckMate(state, player):
            return 1000/moves
        else:
            return -1

    def q_learning_1(self, state, player):
        stateW = state.stateW
        stateB = state.stateB
        self.resetTable(self.chess.board)

        # print(stateW)
        # print(self.getCurrentStateW())
        board = copy.deepcopy(self.chess.boardSim)
        # Constantes
        gamma = 0.9  # Constante de Disminución
        alpha_W = 0.05  # Constante de Aprendizaje
        epoches = 5000

        list_epochs = []
        list_error= []
        convergency = False
        # Inicialización de la tabla de Q-values
        Q= dict()
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
        Q[self.tupleSort(stateW, stateB)] = dict()
        Q[self.tupleSort(stateW, stateB)][player] = dict()

        for child in children:
            Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(child, stateB if player else stateW)] = 0


        for epoch in tqdm(range(1, epoches + 1)):
            newState = state
            player = True
            delta_list = list()
            self.resetTable(board)
            moves = 0
            visited = set()
            explore_random = False
            while not self.isTerminalState(newState):


                moves += 1

                ct = 1 / (epoch * 0.01)
                stateW = newState.stateW
                stateB = newState.stateB
                children = self.getListNextStates(stateW, stateB, player)
                children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
                visited.add(self.tupleSort(stateW, stateB))


                a = random.random()

                # Elegimos el hijo de mayor Q-value
                if explore_random:
                    child = random.choice(children)
                    explore_random = False
                elif a > ct:
                    if not player:
                        v, child = self.get_maxStates_2(Q, stateW, stateB, player)
                    if player:
                        v, child = self.get_minStates_1(Q, stateW, stateB, player)
                # Elegimos un hijo al azar
                else:
                    child = random.choice(children)


                if self.tupleSort(child if player else stateW, stateB if player else child) in visited:
                    break

                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:
                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, not player)

                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        explore_random = True
                        newState.stateW = stateW
                        newState.stateB = stateB
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(newStateW, newStateB)] = 0


                else:

                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, not player)

                # Recompensa del nuevo estado
                r_1 = self.recompensa_1(newState, moves, player)

                if r_1 == 0:
                    break

                # Máximo de los Q-values des del hijo
                if player:
                    maxim, fill = self.get_minStates_1(Q, newStateW, newStateB, not player)
                if not player:
                    maxim, fill = self.get_maxStates_2(Q, newStateW, newStateB, not player)

                q = Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(newStateW, newStateB)]
                delta_w = r_1 + gamma * maxim - q

                Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(newStateW, newStateB)] = q + alpha_W * delta_w

                delta_list.append(delta_w)
                # Si hem fet checkmate, fem reset del tauler i dels estats

                player = not player

            error = max(abs(max(delta_list)), abs(min(delta_list)))
            list_error.append(error)
            list_epochs.append(epoch)

            if error < pow(10, -5):
                convergency = True
                break

        print("----END----")
        print("Convergency: ", convergency)
        print("Delta: ", error)
        print("Epochs: ", epoch)
        print("--------------------------")
        # Graficar
        plt.style.use('seaborn')
        plt.title("Gráfica de los deltas")
        plt.scatter(list_epochs, list_error, s=1)
        plt.xticks(np.arange(min(list_epochs) - 1, max(list_epochs) + 1, int(epoch / 15)))
        plt.yticks(np.arange(0, max(list_error) + 1, 10))
        plt.show()
        self.resetTable(board)

        return Q

    def get_minStates_1(self, Q, stateW, stateB, player):

        if self.tupleSort(stateW, stateB) in Q and player in Q[self.tupleSort(stateW, stateB)]:
            m = min(Q[self.tupleSort(stateW, stateB)][player].items(), key=lambda x: x[1])
            minim = m[1]
            fill = self.listSort(m[0], player)
            return minim, fill
        else:
            children = self.getListNextStates(stateW, stateB, player)

            if self.tupleSort(stateW, stateB) not in Q:
                Q[self.tupleSort(stateW, stateB)] = dict()
            Q[self.tupleSort(stateW, stateB)][player] = dict()
            children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

            for child in children:
                Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(child, stateB if player else stateW)] = 0

            return 0, random.choice(children)

    def recompensa_1(self, state, moves, player):
        stateW = state.stateW
        stateB = state.stateB

        children = self.getListNextStates(stateW, stateB, not player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, not player)]

        if len(children) == 0:
            state.terminal = True
            return 0


        if self.onlyKings(state):
            state.terminal = True
            return -1000

        value = 0

        if self.isCheckMate(state, player):
            print("Checkmate", player)
            return 1000

        if player:
            for i in stateW:
                if i[2] == 2:
                    value += 10


            for i in stateB:
                if i[2] == 8:
                    value -= 10
        else:
            for i in stateW:
                if i[2] == 2:
                    value -= 10

            for i in stateB:
                if i[2] == 8:
                    value += 10

        return value - 1

    def get_maxStates_2(self, Q, stateW, stateB, player):

        if self.tupleSort(stateW, stateB) in Q and player in Q[self.tupleSort(stateW, stateB)]:
            m = max(Q[self.tupleSort(stateW, stateB)][player].items(), key=lambda x: x[1])
            maxim = m[1]
            fill = self.listSort(m[0], player)
            return maxim, fill
        else:
            children = self.getListNextStates(stateW, stateB, player)

            if self.tupleSort(stateW, stateB) not in Q:
                Q[self.tupleSort(stateW, stateB)] = dict()

            Q[self.tupleSort(stateW, stateB)][player] = dict()
            children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]

            for child in children:
                Q[self.tupleSort(stateW, stateB)][player][self.tupleSort(child, stateB if player else stateW)] = 0

            return 0, random.choice(children)

    def onlyKings(self, state):
        stateW = state.stateW
        stateB = state.stateB

        for i in stateW:
            if i[2] != 6:
                return False
        for i in stateB:
            if i[2] != 12:
                return False
        state.terminal = True
        return True

    def recompensa_2(self, state, player):
        stateW = state.stateW
        stateB = state.stateB

        # Recompensa orientada per a que les negres es deixin guanyar
        value = 0
        # Material count

        if self.onlyKings(state):
            value -= 10

        if player:
            for i in stateW:
                if i[2] == 2:
                    value += 1


            for i in stateB:
                if i[2] == 8:
                    value -= 1

        else:
            for i in stateW:
                if i[2] == 2:
                    value -= 1

            for i in stateB:
                if i[2] == 8:
                    value += 1


        '''
        # Mobility
        children = self.getListNextStates(stateW, stateB, True)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, True)]
        value += len(children)

        children = self.getListNextStates(stateW, stateB, False)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, False)]
        value -= len(children)

        # Checks
        if self.isCheck_1(state, True):
            value += 10
        if self.isCheck_1(state, False):
            value -= 10
        '''

        if self.isCheckMate(state, player):
            value += 10
        else:
            value -= 1

        if self.isCheckMate(state, not player):
            value -= 10
        else:
            value += 1

        return value



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

def eliminar_estat(stateW, stateB, child, player):
    if player:
        for i in stateB:
            for j in child:
                if i[0:2] == j[0:2]:
                    stateB.remove(i)
    else:
        for i in stateW:
            for j in child:
                if i[0:2] == j[0:2]:
                    stateW.remove(i)



if __name__ == "__main__":

    print("stating AI chess... ")
    TA = np.zeros((8, 8))

    TA[0][5] = 12
    TA[7][5] = 6
    TA[7][0] = 2
    TA[0][0] = 8

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
    state = State(currentStateW, currentStateB, True)
    print(aichess.isCheck_1(state, True))
    depth = 0

    '''
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
    Q = aichess.q_learning_1(state, True)
    # Exercici 2
    #Q_W, Q_B = aichess.q_learning_2(state, True)

    # Exercici 3
    # PARTIDA:

    player = True
    move = 0
    while not aichess.isCheckMate(state, player):
        move += 1
        #Q = Q_W if player else Q_B
        currentStateW = state.stateW
        currentStateB = state.stateB
        pare = aichess.tupleSort(currentStateW, currentStateB)
        maxim = -float("inf")
        if player:
            value, child = aichess.get_maxStates_2(Q, currentStateW, currentStateB, player)
        else:
            value, child = aichess.get_minStates_1(Q, currentStateW, currentStateB, player)

        aichess.moveSim(currentStateW, currentStateB, child, player)
        aichess.move(currentStateW, currentStateB, child, player)
        eliminar_estat(currentStateW, currentStateB, child, player)
        depth += 1
        aichess.chess.board.print_board()
        if player:
            state = State(child, currentStateB, player)
        else:
            state = State(currentStateW, child, player)
        player = not player
        if aichess.onlyKings(state) or move >= 80:
            print("Draw")
            break
        print(currentStateW, aichess.getCurrentStateW())

