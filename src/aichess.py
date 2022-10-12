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
        self.myCurrentStateW = self.chess.boardSim.currentStateW;
        self.myCurrentStateB = self.chess.boardSim.currentStateB;
        self.depthMax = 8;
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

    def isCheckMate(self, mystate, mycolor = True):

        def checkStates(myState):
            # White checkmate
            rook_y, rook_x, king_y, king_x, o_rook_y, o_rook_x, o_king_y, o_king_x = (None, None, None, None, None, None, None, None)
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
            king_surrondings = {(king_y-1, king_x-1), (king_y-1, king_x), (king_y-1, king_x+1), (king_y, king_x-1), (king_y, king_x+1), (king_y+1, king_x-1), (king_y+1, king_x), (king_y+1, king_x+1)}
            if (o_king_y, o_king_x) in king_surrondings or o_rook_x == king_x or o_rook_y == king_y:
                return True
            return False
        if mycolor:
            assert self.getCurrentStateW() == mystate,  "State not match!!"
            children = self.getListNextStatesW(mystate)
            for i in children:
                if checkStates(i) == False:
                    return False
        else:
            assert self.getCurrentStateB() == mystate,  "State not match!!"
            children = self.getListNextStatesB(mystate)
            for i in children:
                if checkStates(i) == False:
                    return False
        return True
        

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
    TA[0][4] = 6
    TA[2][4] = 12
    TA[0][7] = 8

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State", currentState)

    # it uses board to get them... careful 
    aichess.getListNextStatesW(currentState)
    #   aichess.getListNextStatesW([[7,4,2],[7,4,6]])
    print("list next states ", aichess.isCheckMate(aichess.getCurrentStateW()))

