#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy

import chess
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


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
        self.myCurrentStateW = self.chess.boardSim.currentStateW
        self.myCurrentStateB = self.chess.boardSim.currentStateB
        self.depthMax = 8
        self.checkMate = False

    def getCurrentStateW(self):
        return self.myCurrentStateW

    def getCurrentStateB(self):
        return self.myCurrentStateB

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):

        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

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

    def isCheckMate(self, mystate, mycolor=True):

        def checkStates(myState):
            # White checkmate
            rook_y, rook_x, king_y, king_x, o_rook_y, o_rook_x, o_king_y, o_king_x = (
            None, None, None, None, None, None, None, None)
            if mycolor:
                for i in myState:
                    if i[2] == 2:
                        rook_y, rook_y = i[0:2]
                    elif i[2] == 6:
                        king_y, king_x = i[0:2]
                opponent_state = self.getCurrentStateB()
                for i in opponent_state:
                    if i[2] == 8:
                        o_rook_y, o_rook_x = i[0:2]
                    elif i[2] == 12:
                        o_king_y, o_king_x = i[0:2]
            # Black checkmate
            else:
                for i in myState:
                    if i[2] == 8:
                        rook_y, rook_x = i[0:2]
                    elif i[2] == 12:
                        king_y, king_x = i[0:2]
                opponent_state = self.getCurrentStateW()
                for i in opponent_state:
                    if i[2] == 2:
                        o_rook_y, o_rook_x = i[0:2]
                    elif i[2] == 6:
                        o_king_y, o_king_x = i[0:2]
            king_surrondings = {(king_y - 1, king_x - 1), (king_y - 1, king_x), (king_y - 1, king_x + 1),
                                (king_y, king_x - 1), (king_y, king_x + 1), (king_y + 1, king_x - 1),
                                (king_y + 1, king_x), (king_y + 1, king_x + 1)}
            if (o_king_y, o_king_x) in king_surrondings or o_rook_x == king_x or o_rook_y == king_y:
                return True
            return False

        # children = self.getListNextStatesW(self.getCurrentStateW()) if mycolor else self.getListNextStatesB(self.getCurrentStateB())
        if mycolor:
            assert self.getCurrentStateW() == mystate, "State not match!!"
            children = self.getListNextStatesW(mystate)
            for i in children:
                if checkStates(i) == False:
                    return False
        else:
            assert self.getCurrentStateB() == mystate, "State not match!!"
            children = self.getListNextStatesB(mystate)
            for i in children:
                if checkStates(i) == False:
                    return False
        return True

    def evaluate(self, state, color = True):
        if self.isCheckMate(state, color):
            return 999999
        oponent_state = self.getCurrentStateB() if color else self.getCurrentStateW()

        value = 0
        for i in state:
            if i[2] == 2 or i[2] == 8:
                value += 50
            elif i[2] == 6 or i[2] == 12:
                value += 900

        for i in oponent_state:
            if i[2] == 2 or i[2] == 8:
                value -= 50
            elif i[2] == 6 or i[2] == 12:
                value -= 900

        return value

    # In order to eliminate invalid states
    def checkPositions(self, state):
        positions = set()
        for piece in state:
            (x, y) = piece[0:2]
            if (x, y) in positions:
                return False
            positions.add((x, y))
        return True

    def tupleSort(self, state):
        return tuple(tuple(i) for i in sorted(state))


    def moveSim(self, currentState, nextState):
        (x_1, y_1) = [e for e in currentState if e not in nextState][0][0:2]
        (x_2, y_2) = [e for e in nextState if e not in currentState][0][0:2]
        self.chess.moveSim([x_1, y_1], [x_2, y_2])
        return

    def minimax_decision(self, stateW, stateB, color=True):
        """Given a state in a game, calculate the best move by searching
        forward all the way to the terminal states. [Fig. 6.4]"""
        visited = set()
        def max_value(stateW, stateB, depth, color):
            if depth == self.depthMax:
                return self.evaluate(state, color)
            v = -float('inf')
            children = list()
            if color:
                assert self.getCurrentStateW() == stateW, "State not match!!"
                children = self.getListNextStatesW(stateW)
                tuple_stateW = self.tupleSort(stateW)
                visited.add(tuple_stateW)
                for child in children:
                    print(child)
                    tuple_child = self.tupleSort(child)
                    if self.checkPositions(child) == False or tuple_child in visited:
                        continue
                    self.moveSim(stateW, child)
                    v = max(v, min_value(child, stateB, depth + 1, not color))
                    self.moveSim(child, stateW)
            else:
                assert self.getCurrentStateB() == stateB, "State not match!!"
                children = self.getListNextStatesB(stateB)
                tuple_stateB = self.tupleSort(stateB)
                visited.add(tuple_stateB)
                for child in children:
                    tuple_child = self.tupleSort(child)
                    print(child)
                    if self.checkPositions(child) == False or tuple_child in visited:
                        continue
                    self.moveSim(stateB, child)
                    v = max(v, min_value(stateW, child, depth + 1, not color))
                    self.moveSim(child, stateB)
            return v

        def min_value(stateW, stateB, depth, color):
            if depth == self.depthMax:
                return self.evaluate(state, color)
            v = float('inf')
            children = list()
            if color:
                assert self.getCurrentStateW() == stateW, "State not match!!"
                children = self.getListNextStatesW(stateW)
                tuple_stateW = self.tupleSort(stateW)
                visited.add(tuple_stateW)
                for child in children:
                    tuple_child = self.tupleSort(child)
                    print(child)
                    if self.checkPositions(child) == False or tuple_child in visited:
                        continue
                    self.moveSim(stateW, child)
                    v = min(v, max_value(child, stateB, depth + 1, not color))
                    self.moveSim(child, stateW)
            else:
                print(self.getCurrentStateB(), stateB)
                print(self.getCurrentStateW(), stateW)
                assert self.getCurrentStateB() == stateB, "State not match!!"
                children = self.getListNextStatesB(stateB)
                tuple_stateB = self.tupleSort(stateB)
                visited.add(tuple_stateB)
                for child in children:
                    tuple_child = self.tupleSort(child)
                    print(child)
                    if self.checkPositions(child) == False or tuple_child in visited:
                        continue
                    self.moveSim(stateB, child)
                    v = min(v, max_value(stateW, child, depth + 1, not color))
                    self.moveSim(child, stateB)
            return v

        # Body of minimax_decision starts here:
        children = list()
        if color:
            print(self.getCurrentStateB(), stateB)
            print(self.getCurrentStateW(), stateW)
            assert self.getCurrentStateW() == stateW, "State not match!!"
            children = self.getListNextStatesW(stateW)
            tuple_stateW = self.tupleSort(stateW)
            visited.add(tuple_stateW)
            v = -float('inf')
            next_move = list()
            for child in children:
                print(child)
                if self.checkPositions(child) == False:
                    continue
                self.moveSim(stateW, child)
                print("AAA", currentStateW, currentStateB, child)
                value = min_value(child, stateB, 0, not color)
                if value > v:
                    v = value
                    next_move = child
                self.moveSim(child, stateW)
        else:
            assert self.getCurrentStateB() == stateB, "State not match!!"
            children = self.getListNextStatesB(stateB)
            tuple_stateB = self.tupleSort(stateB)
            visited.add(tuple_stateB)
            v = -float('inf')
            next_move = list()
            for child in children:
                print(child)
                if self.checkPositions(child) == False:
                    continue
                self.moveSim(stateB, child)
                value = min_value(stateW, child, 0, not color)
                if value > v:
                    v = value
                    next_move = child
                self.moveSim(child, stateB)

        return next_move


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


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    # TA[0][0] = 2
    # TA[2][4] = 6
    # # black pieces
    # TA[0][4] = 12

    TA[0][0] = 2
    TA[0][5] = 6
    TA[2][4] = 12
    TA[0][7] = 8

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentStateW = aichess.chess.board.currentStateW.copy()
    currentStateB = aichess.chess.board.currentStateB.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current StateW", currentStateW)
    print("current StateB", currentStateB)

    # it uses board to get them... careful 
    aichess.getListNextStatesW(currentStateW)
    #   aichess.getListNextStatesW([[7,4,2],[7,4,6]])
    print("list next states ", aichess.isCheckMate(aichess.getCurrentStateW()))

    print("next move", aichess.minimax_decision(aichess.getCurrentStateW(), aichess.getCurrentStateB(), True))
