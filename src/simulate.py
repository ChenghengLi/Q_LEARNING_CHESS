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

    def simulate(self, depth, times, algorithmW, algorithmB):
        self.aichess.depthMax = depth
        black_counter = 0
        white_counter = 0
        draw_counter = 0
        for _ in range(times):
            move_counter = 0
            while True and move_counter < 40:
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
                move_counter += 1
            self._reset()
            if move_counter == 40:
                draw_counter += 1

        print("--RESULTS--")
        print("White wins:", white_counter, "->", 100*white_counter/times, "%")
        print("Black wins:", black_counter, "->", 100*black_counter/times, "%")
        print("Draws:", draw_counter, "->", 100*draw_counter/times, "%")
        print("|| Depth =", depth, "||")

if __name__ == "__main__":
    print("Simulation innitialized")
    simulator = Simulate()
    algorithm = {"minmax":1, "alphabeta":2, "expectimax":3}
    depth = 2
    aW = algorithm["alphabeta"]
    aB = algorithm["alphabeta"]
    times = 1
    simulator.simulate(depth, times, aW, aB)