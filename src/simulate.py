import numpy as np

import aichess
import copy

class Simulate:

    def __init__(self):
        self._reset()


    def _reset(self):
        TA = np.zeros((8, 8))
        TA[7][7] = 2
        TA[2][4] = 6
        TA[0][5] = 12
        TA[2][6] = 8

        self.aichess = aichess.Aichess(TA, True)
        self.currentStateW = copy.deepcopy(self.aichess.chess.board.currentStateW)
        self.currentStateB = copy.deepcopy(self.aichess.chess.board.currentStateB)
        print("INNITIAL BOARD")
        self.aichess.chess.board.print_board()
        print("STARTING PLAYING")
        self.player = True

    def isCheckmate(self):
        return self.aichess.isCheckMate(self.currentStateW, self.currentStateB, self.player)

    def _getNextMove(self, algorithm):
        if algorithm == 1:
            return self.aichess.minimax_decision(self.currentStateW, self.currentStateB, self.player)
        elif algorithm == 2:
            return self.aichess.alphabeta(self.currentStateW, self.currentStateB, self.player)
        elif algorithm == 3:
            return self.aichess.expectimax(self.currentStateW, self.currentStateB, self.player)

    def _move(self, next_move):
        self.aichess.move(self.currentStateW, self.currentStateB, next_move, self.player)
        self.aichess.moveSim(self.currentStateW, self.currentStateB, next_move, self.player)
        if self.player:
            self.currentStateW =copy.deepcopy(next_move)
        else:
            self.currentStateB = copy.deepcopy(next_move)
        self.aichess.chess.board.print_board()

    def simulate(self, depth, times, algorithmW, algorithmB):
        self.aichess.depthMax = depth
        black_counter = 0
        white_counter = 0
        for _ in range(times):
            while True:
                next_move = self._getNextMove(algorithmW)
                self._move(next_move)
                if self.isCheckmate():
                    white_counter += 1
                    break
                self.player = not self.player
                next_move = self._getNextMove(algorithmB)
                self._move(next_move)
                if self.isCheckmate():
                    black_counter += 1
                    break
                self.player = not self.player
            self._reset()

if __name__ == "__main__":
    print("Simulation innitialized")
    simulator = Simulate()
    algorithm = {"minmax":1, "alphabeta":2, "expectimax":3}
    depth = 4
    aW = algorithm["minmax"]
    aB = algorithm["minmax"]
    times = 10
    simulator.simulate(depth, times, aW, aB)

