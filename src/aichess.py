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

    def isCheckMate(self, stateW, stateB, player=True):

        o_children = self.getListNextStates(stateW, stateB, not player)
        # Tots els possibles moviments del rival
        for o_child in o_children:
            next = False
            if not self.checkPositions(stateW, stateB, o_child, not player):
                continue
            o_kill, o_nState = self.moveSim(stateW, stateB, o_child, not player)
            if o_kill:
                if self.checkKing(o_nState, not player):
                    self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                    continue
                if player:
                    sW = o_nState
                else:
                    sB = o_nState
            else:
                sW = stateW
                sB = stateB

            for i in o_child:
                if i[2] == 6 or i[2] == 12:
                    o_king_y, o_king_x = i[0:2]

            # En cada possible moviment s'ha de poder matar al rei en una jugada
            if player:
                children = self.getListNextStates(sW, o_child, player)
                for child in children:
                    next = False
                    # Checkpositions
                    if not self.checkPositions(sW, o_child, child, player):
                        continue
                    # Fem el moviment
                    kill, nState = self.moveSim(sW, o_child, child, player)
                    # Revisem si podem arribar a la casella del rei rival
                    for piece in child:
                        if piece[0:2] == [o_king_y, o_king_x]:
                            next = True
                            break
                    self.undoMovement(sW, o_child, child, kill, player)
                    if next:
                        break
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                if next:
                    continue
                return False

            else:
                children = self.getListNextStates(o_child, sB, player)
                for child in children:
                    next = False
                    if not self.checkPositions(o_child, sB, child, player):
                        continue
                    kill, nState = self.moveSim(o_child, sB, child, player)
                    for piece in child:
                        if piece[0:2] == [o_king_y, o_king_x]:
                            next = True
                            break
                    self.undoMovement(o_child, sB, child, kill, player)
                    if next:
                        break
                self.undoMovement(stateW, stateB, o_child, o_kill, not player)
                if next:
                    continue
                return False

        return True

    def getKingSurrondings(self, king_y, king_x):
        king_surrondings = {(king_y - 1, king_x - 1), (king_y - 1, king_x), (king_y - 1, king_x + 1),
                            (king_y, king_x - 1), (king_y, king_x + 1), (king_y + 1, king_x - 1),
                            (king_y + 1, king_x), (king_y + 1, king_x + 1)}
        return king_surrondings

    def evaluate(self, stateW, stateB, player = True):
        if self.isCheckMate(stateW, stateB, player):
            return 999999

        oponent_state = stateB if player else stateW
        state = stateW if player else stateB

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
    def checkPositions(self, stateW, stateB, child, player):

        if player:
            #self.chess.boardSim.print_board()
            #print(self.checkKing(child, player))
            #print(stateW, stateB)
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
        #print(king_code)
        return king_code not in set(x[2] for x in state)


    def minimax_decision(self, stateW, stateB, player=True):

        visited = set()

        def max_value(stateW, stateB, child, depth, player):

            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, player):
                print(stateW, stateB, player)
                print("White checkmate")
                self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                #print(a)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:
                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  min_value(stateW, nState, child, depth + 1, not player)
                    else:
                        value =  min_value(nState, stateB, child, depth + 1, not player)
                else:
                    value =  min_value(stateW, stateB, child, depth + 1, not player)
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        def min_value(stateW, stateB, child, depth, player):
            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, player):
                print("Black checkmate")
                print(stateW, stateB, player)
                self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                print(a)
                return a
            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)
            for child in children:
                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  max_value(stateW, nState, child, depth + 1, not player)
                    else:
                        value =  max_value(nState, stateB, child, depth + 1, not player)
                else:
                    value =  max_value(stateW, stateB, child, depth + 1, not player)
                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)

        for child in children:
            if not self.checkPositions(stateW, stateB, child, player):
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)
            if kill:
                if self.checkKing(nState, player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                if player:
                    value = min_value(stateW, nState, child, 1, not player)
                else:
                    value = min_value(nState, stateB, child, 1, not player)
            else:
                value = min_value(stateW, stateB, child, 1, not player)
            #print(value, v, child)
            if value > v:
                #print(child)
                #self.chess.boardSim.print_board()
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

        print("Next move:", next_move)
        return next_move


    def alphabeta(self, stateW, stateB, player):

        visited = set()

        def max_value(stateW, stateB, child, depth, alpha, beta, player):
            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, player):
                print(stateW, stateB, player)
                print("max_value checkmate")
                self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                # print(a)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:
                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value = min_value(stateW, nState, child, depth + 1, alpha, beta, not player)
                    else:
                        value = min_value(nState, stateB, child, depth + 1, alpha, beta, not player)
                else:
                    value = min_value(stateW, stateB, child, depth + 1, alpha, beta, not player)
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

                if v >= beta:
                    return v
                alpha = max(alpha, v)

            return v

        def min_value(stateW, stateB, child, depth, alpha, beta, player):
            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, player):
                print("min_value checkmate")
                print(stateW, stateB, player)
                self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                # print(a)
                return a
            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)
            for child in children:
                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if not self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value = max_value(stateW, nState, child, depth + 1, alpha, beta, not player)
                    else:
                        value = max_value(nState, stateB, child, depth + 1, alpha, beta, not player)
                else:
                    value = max_value(stateW, stateB, child, depth + 1, alpha, beta, not player)
                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)
        alpha = -float("inf")
        beta = float("inf")

        for child in children:
            if not self.checkPositions(stateW, stateB, child, player):
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)
            if kill:
                if self.checkKing(nState, player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                if player:
                    value = min_value(stateW, nState, child, 1, alpha, beta, not player)
                else:
                    value = min_value(nState, stateB, child, 1, alpha, beta, not player)
            else:
                value = min_value(stateW, stateB, child, 1, alpha, beta, not player)
            # print(value, v, child)
            if value > v:
                #print(child)
                #self.chess.boardSim.print_board()
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

            if v >= beta:
                return next_move
            alpha = max(alpha, v)

        print(next_move)
        return next_move


    def expectiminimax(self, stateW, stateB, player):

        pass


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

    TA[0][7] = 2
    TA[0][5] = 6
    TA[2][5] = 12
    TA[2][0] = 8

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentStateW = copy.deepcopy(aichess.chess.board.currentStateW)
    currentStateB = copy.deepcopy(aichess.chess.board.currentStateB)

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current StateW", currentStateW)
    print("current StateB", currentStateB)
    #print(aichess.isCheckMate(currentStateW, currentStateB, True))


    #sW = [[6,0,2], [7,5,6]]
    #sB = [[0,5,12], [0,7,8]]

    #aichess.chess.boardSim.updateState(sW, sB)
    #aichess.chess.boardSim.print_board()
    #print(aichess.getListNextStates(currentStateW, currentStateB, True))
    aichess.minimax_decision(currentStateW, currentStateB, True)