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
    def isCheck(self, stateW, stateB, oponent_king, player = True):

        def checkPosition(child):
            positions = set()
            for piece in child:
                (x, y) = piece[0:2]
                if (x, y) in positions:
                    return False
                positions.add((x, y))
            return True

        children = self.getListNextStates(stateW, stateB, player)
        for child in children:
            if not checkPosition(child):
                continue
            mymoves = set((x[0],x[1]) for x in child)
            if oponent_king in mymoves:
                return True

        return False



    def isCheckMate(self, stateW, stateB, player=True):

        state = stateW if not player else stateB

        for i in state:
            if i[2] == 6 or i[2] == 12:
                o_king_y, o_king_x = i[0:2]

        if not self.isCheck(stateW, stateB, (o_king_y, o_king_x), player):
            return False

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
                if player:
                    sW = o_nState
                    sB = stateB
                else:
                    sB = o_nState
                    sW = stateW
            else:
                sW = stateW if player else o_child
                sB = stateB if not player else o_child

            for i in o_child:
                if i[2] == 6 or i[2] == 12:
                    o_king_y, o_king_x = i[0:2]

            if self.isCheck(sW, sB, (o_king_y, o_king_x), player):
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

    def evaluate(self, stateW, stateB, player = True):


        oponent_state = stateB if player else stateW
        state = stateW if player else stateB

        value = 0
        o_rook_y, o_rook_x, rook_y, rook_x = None, None, None, None
        # Material count
        for i in state:
            if i[2] == 2 or i[2] == 8:
                value += 5
                rook_y, rook_x = i[0:2]
            elif i[2] == 6 or i[2] == 12:
                value += 100
                king_y, king_x = i[0:2]

        for i in oponent_state:
            if i[2] == 2 or i[2] == 8:
                value -= 5
                o_rook_y, o_rook_x = i[0:2]
            elif i[2] == 6 or i[2] == 12:
                value -= 100
                o_king_y, o_king_x = i[0:2]

        # Mobility
        children = self.getListNextStates(stateW, stateB, player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
        value += len(children)
        children = self.getListNextStates(stateW, stateB, not player)
        children = [x for x in children if self.checkPositions(stateW, stateB, x, player)]
        value -= len(children)

        # King safety
        '''
        king_surrondings = self.getKingSurrondings(king_y, king_x)
        for i in oponent_state:
            if tuple(i[0:2]) in king_surrondings:
                value += 1

        o_king_surrondings = self.getKingSurrondings(o_king_y, o_king_x)
        for i in state:
            if tuple(i[0:2]) in o_king_surrondings:
                value -= 1
        '''


        # Checks
        if self.isCheck(stateW, stateB, (o_king_y, o_king_x), player):
            value += 10

        if self.isCheck(stateW, stateB, (king_y, king_x), not player):
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
            # get far from the the oppponent king
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
        #print(king_code)
        return king_code not in set(x[2] for x in state)



    def minimax_decision(self, stateW, stateB, player=True):

        visited = set()

        def max_value(stateW, stateB, child, depth, player):

            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, not player):
                return -999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:
                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player): #or self.tupleSort(child if player else stateW, child if not player else stateB) in visited:
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  min_value(stateW, nState, child, depth + 1, not player)
                    else:
                        value =  min_value(nState, stateB, child, depth + 1, not player)
                else:
                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    value =  min_value(stateW, stateB, child, depth + 1, not player)
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        def min_value(stateW, stateB, child, depth, player):
            if player:
                stateB = child
            else:
                stateW = child

            visited.add(self.tupleSort(stateW, stateB))

            if self.isCheckMate(stateW, stateB, not player):
                #print("max_value checkmate")
                #print(stateW, stateB, not player)
                #self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                return a

            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:
                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player): #or self.tupleSort(child if player else stateW, child if not player else stateB) in visited:
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  max_value(stateW, nState, child, depth + 1, not player)
                    else:
                        value =  max_value(nState, stateB, child, depth + 1, not player)

                else:
                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    value =  max_value(stateW, stateB, child, depth + 1, not player)


                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        if self.isCheckMate(stateW, stateB, not player):
            #print("max_value checkmate")
            #print(stateW, stateB, not player)
            # self.chess.boardSim.print_board()
            if player:
                state = stateW
            else:
                state = stateB
            print("Checkmate")
            return state

        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)

        for child in children:
            for i in child:
                if i[2] == 6 or i[2] == 12:
                    king_y, king_x = i[0:2]
            king = (king_y, king_x)

            if not self.checkPositions(stateW, stateB, child, player): #or self.tupleSort(child if player else stateW, child if not player else stateB) in visited:
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:
                if player:
                    if self.isCheck(child, nState, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(nState, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                if self.checkKing(nState, player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                if player:
                    value = min_value(stateW, nState, child, 1, not player)
                else:
                    value = min_value(nState, stateB, child, 1, not player)
            else:
                if player:
                    if self.isCheck(child, stateB, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(stateW, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                value = min_value(stateW, stateB, child, 1, not player)

            if value > v:
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


            if self.isCheckMate(stateW, stateB, not player):
                return -999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                # print(a)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:

                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value = min_value(stateW, nState, child, depth + 1, alpha, beta, not player)
                    else:
                        value = min_value(nState, stateB, child, depth + 1, alpha, beta, not player)
                else:

                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

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

            if self.isCheckMate(stateW, stateB, not player):
                #print("max_value checkmate")
                #print(stateW, stateB, not player)
                #self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, not player)
                # print(a)
                return a
            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:

                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value = max_value(stateW, nState, child, depth + 1, alpha, beta, not player)
                    else:
                        value = max_value(nState, stateB, child, depth + 1, alpha, beta, not player)
                else:

                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    value = max_value(stateW, stateB, child, depth + 1, alpha, beta, not player)

                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        if self.isCheckMate(stateW, stateB, player):
            #print("max_value checkmate")
            # print(stateW, stateB, not player)
            # self.chess.boardSim.print_board()
            if player:
                state = stateW
            else:
                state = stateB
            print("Checkmate")
            return state

        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)
        alpha = -float("inf")
        beta = float("inf")

        for child in children:

            for i in child:
                if i[2] == 6 or i[2] == 12:
                    king_y, king_x = i[0:2]
            king = (king_y, king_x)
            if not self.checkPositions(stateW, stateB, child, player):
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:

                if player:
                    if self.isCheck(child, nState, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(nState, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                if self.checkKing(nState, player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                if player:
                    value = min_value(stateW, nState, child, 1, alpha, beta, not player)
                else:
                    value = min_value(nState, stateB, child, 1, alpha, beta, not player)
            else:

                if player:
                    if self.isCheck(child, stateB, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(stateW, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                value = min_value(stateW, stateB, child, 1, alpha, beta, not player)

            #print(value, v, child)
            if value > v:
                #self.chess.boardSim.print_board()
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

            if v >= beta:
                print("Next move:", next_move)
                return next_move
            alpha = max(alpha, v)

        print("Next move:", next_move)
        return next_move


    def expectimax(self, stateW, stateB, player):

        visited = set()
        def max_value(stateW, stateB, child, depth, player):

            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, not player):
                #print(stateW, stateB, not player)
                #print("chance_value checkmate")
                #self.chess.boardSim.print_board()
                return -999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, player)
                #print(a)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:
                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  chance_value(stateW, nState, child, depth + 1, not player)
                    else:
                        value =  chance_value(nState, stateB, child, depth + 1, not player)
                else:
                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    value =  chance_value(stateW, stateB, child, depth + 1, not player)
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        def chance_value(stateW, stateB, child, depth, player):
            if player:
                stateB = child
            else:
                stateW = child

            if self.isCheckMate(stateW, stateB, not player):
                #print("max_value checkmate")
                #print(stateW, stateB, not player)
                #self.chess.boardSim.print_board()
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(stateW, stateB, not player)
                #print(a)
                return a

            children = self.getListNextStates(stateW, stateB, player)

            sum = 0
            plays = {}
            for child in children:
                for i in child:
                    if i[2] == 6 or i[2] == 12:
                        king_y, king_x = i[0:2]
                king = (king_y, king_x)

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    if player:
                        if self.isCheck(child, nState, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(nState, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    if self.checkKing(nState, player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                    if player:
                        value =  max_value(stateW, nState, child, depth + 1, not player)
                        fill = child.copy()
                        i = 0
                        for piece in fill:
                            fill[i] = tuple(piece)
                            i += 1
                        plays[tuple(fill)] = value
                        if value > 0:
                            sum += -1/value
                        elif value == 0:
                            sum += -1
                        else:
                            sum += value
                    else:
                        value =  max_value(nState, stateB, child, depth + 1, not player)
                        fill = child.copy()
                        i = 0
                        for piece in fill:
                            fill[i] = tuple(piece)
                            i += 1
                        plays[tuple(fill)] = value
                        if value > 0:
                            sum += -1/value
                        elif value == 0:
                            sum += -1
                        else:
                            sum += value
                else:
                    if player:
                        if self.isCheck(child, stateB, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue
                    else:
                        if self.isCheck(stateW, child, king, not player):
                            self.undoMovement(stateW, stateB, child, kill, player)
                            continue

                    value =  max_value(stateW, stateB, child, depth + 1, not player)
                    i = 0
                    fill = child.copy()
                    for piece in fill:
                        fill[i] = tuple(piece)
                        i += 1
                    plays[tuple(fill)] = value
                    if value > 0:
                        sum += -1/value
                    elif value == 0:
                        sum += -1
                    else:
                        sum += value

                self.undoMovement(stateW, stateB, child, kill, player)

            prob = {}
            v = 0
            if sum == 0:
                return 0
            sum = abs(sum)
            for child in plays.keys():
                if plays[child] > 0:
                    prob[child] = (1/plays[child]) / sum
                else:
                    prob[child] = -plays[child]/sum
                v += prob[child] * plays[child]
            return v

        if self.isCheckMate(stateW, stateB, not player):
            #print("max_value checkmate")
            # print(stateW, stateB, not player)
            # self.chess.boardSim.print_board()
            if player:
                state = stateW
            else:
                state = stateB
            print("Checkmate")
            return state

        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)

        for child in children:
            for i in child:
                if i[2] == 6 or i[2] == 12:
                    king_y, king_x = i[0:2]
            king = (king_y, king_x)

            if not self.checkPositions(stateW, stateB, child, player):
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)
            if kill:
                if player:
                    if self.isCheck(child, nState, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(nState, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                if self.checkKing(nState, player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                if player:
                    value = chance_value(stateW, nState, child, 1, not player)
                else:
                    value = chance_value(nState, stateB, child, 1, not player)
            else:
                if player:
                    if self.isCheck(child, stateB, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue
                else:
                    if self.isCheck(stateW, child, king, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                value = chance_value(stateW, stateB, child, 1, not player)
            if value > v:
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

        print("Next move:", next_move)
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
        cmW = aichess.isCheckMate(currentStateW, currentStateB, True)
        if cmW:
            print("white checkmate")
            aichess.chess.board.print_board()
        cm = aichess.isCheckMate(currentStateW, currentStateB, not True)
        if cm:
            print("black checkmate")
            aichess.chess.board.print_board()

