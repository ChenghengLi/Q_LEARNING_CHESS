from collections import defaultdict

import numpy as np

import aichess
import copy

from src.state import State


class Simulate:

    def __init__(self):
        self._reset()
        print("INNITIAL BOARD")
        self.aichess.chess.board.print_board()
        print("STARTING PLAYING")


    def _reset(self):
        TA = np.zeros((8, 8))
        TA[7][0] = 2
        TA[7][5] = 6
        TA[0][5] = 12
        TA[0][0] = 8

        self.aichess = aichess.Aichess(TA, True)
        self.currentStateW = copy.deepcopy(self.aichess.chess.board.currentStateW)
        self.currentStateB = copy.deepcopy(self.aichess.chess.board.currentStateB)
        self.player = True

    def isCheckmate(self):
        state = State(self.currentStateW, self.currentStateB, None, -1, self.player)
        return self.aichess.isCheckMate(state, self.player)

    def _getNextMove(self, algorithm):
        if algorithm == 1:
            return self.aichess.minimax_decision(self.currentStateW, self.currentStateB, self.player)
        elif algorithm == 2:
            return self.aichess.alphabeta(self.currentStateW, self.currentStateB, self.player)
        elif algorithm == 3:
            return self.aichess.expectimax(self.currentStateW, self.currentStateB, self.player)

    def _move(self, next_move):
        self.aichess.move(self.currentStateW, self.currentStateB, next_move, self.player)
        kill, nState = self.aichess.moveSim(self.currentStateW, self.currentStateB, next_move, self.player)
        if self.player:
            self.currentStateW = copy.deepcopy(next_move)
            if kill:
                self.currentStateB = copy.deepcopy(nState)
        else:
            self.currentStateB = copy.deepcopy(next_move)
            if kill:
                self.currentStateW = copy.deepcopy(nState)
        self.aichess.chess.board.print_board()

    def _isStaleman(self):
        l = self.currentStateW + self.currentStateB
        l = tuple(tuple(x) for x in sorted(l))
        self.prev_moves[l] += 1
        if self.prev_moves[l] == 4:
            return True
        return False




    def simulate(self, depth, times, algorithmW, algorithmB):
        self.aichess.depthMax = depth
        black_counter = 0
        white_counter = 0
        draw_counter = 0
        for _ in range(times):
            print(_ + 1," simulation")
            self.prev_moves = defaultdict(int)
            while True:
                next_move = self._getNextMove(algorithmW)
                self._move(next_move)


                if self.isCheckmate():
                    white_counter += 1
                    break

                if self._isStaleman():
                    draw_counter += 1
                    break

                self.player = not self.player
                next_move = self._getNextMove(algorithmB)
                self._move(next_move)
                if self.isCheckmate():
                    black_counter += 1
                    break
                self.player = not self.player

                if self._isStaleman():
                    draw_counter += 1
                    break

            self._reset()


        print("--RESULTS--")
        print("White wins:", white_counter, "->", 100*white_counter/times, "%")
        print("Black wins:", black_counter, "->", 100*black_counter/times, "%")
        print("Draws:", draw_counter, "->", 100*draw_counter/times, "%")
        print("|| Depth =", depth, "||")

if __name__ == "__main__":
    print("Simulation innitialized")
    simulator = Simulate()
    algorithm = {"minmax":1, "alphabeta":2, "expectimax":3}
    depth = 4
    aW = algorithm["expectimax"]
    aB = algorithm["alphabeta"]
    times = 5
    simulator.simulate(depth, times, aW, aB)