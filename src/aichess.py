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

        positions = set()
        for piece in child:
            (x, y) = piece[0:2]
            if (x, y) in positions:
                return False
            positions.add((x, y))
        return True

    def tupleSort(self, stateW, stateB):
        l = stateW + stateB
        return tuple(tuple(i) for i in sorted(l))


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
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())
    print("stating AI chess... ")
    # intiialize board
    for i in range(10000):
        TA = generateBoard()

        # initialise board
        aichess = Aichess(TA, True)

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