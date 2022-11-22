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



    def minimax_decision(self, stateW, stateB, player=True):
        self.player = player

        def max_value(state, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)

            # En primer lloc, mirem si el estat és un checkmate
            if self.isCheckMate(state, not player):
                return -999999

            # Després mirem si ens trobem en la profunditat demanada
            if depth == self.depthMax:
                a = self.evaluate(state, player)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            # Tot seguit passem a mirar els fills del estat del jugador corresponent
            for child in children:

                # Descartem els fills si són estats "erronis"
                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                # Sinó, feim el moviment
                kill, nState = self.moveSim(stateW, stateB, child, player)

                # Revisem si després de fer-lo és un Check del rival, en aquest cas no el feim
                # Si matem una peça, creem el nou estat llevant la part de la peça morta corresponent
                if kill:

                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    # També mirem si no han matat al rey
                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  min_value(newState, not player)

                else:
                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  min_value(newState, not player)

                # Al final feim el màxim i desfem el moviment
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

            return v

        # Equivalent al max_value però fent el mínim al final
        def min_value(state, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)

            if self.isCheckMate(state, not player):
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(state, player)
                return a

            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                if not self.checkPositions(stateW, stateB, child, player):
                    continue

                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:

                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)


                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  max_value(newState, not player)

                else:

                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  max_value(newState, not player)

                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        # Part principal
        initialState = State(stateW, stateB, None, 0, player)

        # En primer lloc mirem si és un checkmate
        if self.isCheckMate(initialState, not player):
            if player:
                state = stateW
            else:
                state = stateB
            print("Checkmate")
            return state

        # Sinó, comencem a aplicar el minimax
        next_move = None
        v = -float('inf')
        children = self.getListNextStates(stateW, stateB, player)

        for child in children:

            if not self.checkPositions(stateW, stateB, child, player):
                continue

            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:

                newStateW = child if player else nState
                newStateB = nState if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                value = min_value(newState, not player)

            else:

                newStateW = child if player else stateW
                newStateB = stateB if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                value = min_value(newState, not player)

            if value > v:
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

        print("Next move:", next_move)
        return next_move


    # Mètode alphabeta molt similar però afegint la part de les alpha i les beta tal i com s'ha explicat a classe
    def alphabeta(self, stateW, stateB, player):
        self.player = player

        def max_value(state, alpha, beta, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)

            if self.isCheckMate(state, not player):
                return -999999

            if depth == self.depthMax:
                a = self.evaluate(state, player)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                if not self.checkPositions(stateW, stateB, child, player):
                    continue

                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:

                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value = min_value(newState, alpha, beta, not player)
                else:

                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value = min_value(newState, alpha, beta, not player)

                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

                if v >= beta:
                    return v
                alpha = max(alpha, v)


            return v

        def min_value(state, alpha, beta, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)

            if self.isCheckMate(state, not player):
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(state, not player)
                return a

            v = float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                if not self.checkPositions(stateW, stateB, child, player):
                    continue

                kill, nState = self.moveSim(stateW, stateB, child, player)

                if kill:

                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value = max_value(newState, alpha, beta, not player)
                else:

                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value = max_value(newState, alpha, beta, not player)

                v = min(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)

                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        initialState = State(stateW, stateB, None, 0, player)

        if self.isCheckMate(initialState, player):
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


            if not self.checkPositions(stateW, stateB, child, player):
                continue
            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:

                newStateW = child if player else nState
                newStateB = nState if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                value = min_value(newState, alpha, beta, not player)
            else:

                newStateW = child if player else stateW
                newStateB = stateB if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue
                value = min_value(newState, alpha, beta, not player)

            if value > v:
                v = value
                next_move = child
            self.undoMovement(stateW, stateB, child, kill, player)

            if v >= beta:
                print("Next move:", next_move)
                return next_move
            alpha = max(alpha, v)

        print("Next move:", next_move)
        return next_move

    # Mètode expectimax, ara el min_value passa a ser un chance_value
    def expectimax(self, stateW, stateB, player):
        self.player = player

        visited = set()
        def max_value(state, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)

            if self.isCheckMate(state, not player):
                return -999999

            if depth == self.depthMax:
                a = self.evaluate(state, player)
                return a

            v = -float('inf')
            children = self.getListNextStates(stateW, stateB, player)

            for child in children:

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:
                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  chance_value(newState, not player)

                else:
                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  chance_value(newState, not player)
                v = max(v, value)
                self.undoMovement(stateW, stateB, child, kill, player)
            return v

        # Tupelitzar estats
        def tupleState(state):
            return tuple(tuple(row) for row in state)

        def chance_value(state, player):

            stateB = state.stateB
            stateW = state.stateW
            depth = state.depth + 1
            state.check = self.isCheck_1(state, not player)


            if self.isCheckMate(state, not player):
                return 999999

            if depth == self.depthMax:
                a = self.evaluate(state, not player)
                return a

            children = self.getListNextStates(stateW, stateB, player)

            sum = 0
            plays = {}
            for child in children:

                if not self.checkPositions(stateW, stateB, child, player):
                    continue
                kill, nState = self.moveSim(stateW, stateB, child, player)
                if kill:

                    newStateW = child if player else nState
                    newStateB = nState if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  max_value(newState, not player)

                    # Anem guardant els values de les jugades d'aquesta manera en un diccionari
                    # MILLOR EXPLICAT A LA MEMÒRIA
                    plays[tupleState(child)] = value

                    if value > 0:
                        sum += -1/float(value)
                    elif value == 0:
                        sum += -1
                    else:
                        sum += value

                else:
                    newStateW = child if player else stateW
                    newStateB = stateB if player else child
                    newState = State(newStateW, newStateB, state, depth, not player)

                    if self.isCheck_1(newState, not player):
                        self.undoMovement(stateW, stateB, child, kill, player)
                        continue

                    value =  max_value(newState, not player)

                    plays[tupleState(child)] = value
                    if value > 0:
                        sum += -1/float(value)
                    elif value == 0:
                        sum += -1
                    else:
                        sum += value
                self.undoMovement(stateW, stateB, child, kill, player)

            # Al final, apliquem la distribució de probabilitat a cada jugada i la multipliquem pel propi valor de la jugada
            # Així calculem l'esperança

            prob = {}
            v = 0
            if sum == 0:
                return 0
            sum = math.fabs(sum)
            for child in plays.keys():
                if plays[child] > 0:
                    prob[child] = (1/float(plays[child])) / sum
                else:
                    prob[child] = -plays[child]/sum
                v += prob[child] * plays[child]

            return v

        # Part principal
        initialState = State(stateW, stateB, None, 0, player)

        if self.isCheckMate(initialState, not player):
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

            if not self.checkPositions(stateW, stateB, child, player):
                continue

            kill, nState = self.moveSim(stateW, stateB, child, player)

            if kill:
                newStateW = child if player else nState
                newStateB = nState if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.checkKing(nState, player) or self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                value = chance_value(newState, not player)
            else:
                newStateW = child if player else stateW
                newStateB = stateB if player else child
                newState = State(newStateW, newStateB, initialState, 0, not player)

                if self.isCheck_1(newState, not player):
                    self.undoMovement(stateW, stateB, child, kill, player)
                    continue

                value = chance_value(newState, not player)

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

        a = State(currentStateW, currentStateB, None, 0, True)
        cmW = aichess.isCheckMate(a, True)
        if cmW:
            print("white checkmate")
            aichess.chess.board.print_board()
        cm = aichess.isCheckMate(a, not True)
        if cm:
            print("black checkmate")
            aichess.chess.board.print_board()