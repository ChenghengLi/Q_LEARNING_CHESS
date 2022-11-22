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
        TA[7][4] = 2
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




    def simulate(self, depth_1, depth_2, times, algorithmW, algorithmB):
        self.aichess.depthMax = depth
        black_counter = 0
        white_counter = 0
        draw_counter = 0
        for _ in range(times):
            print(_ + 1," simulation")
            self.prev_moves = defaultdict(int)
            while True:

                self.aichess.depthMax = depth_1
                next_move = self._getNextMove(algorithmW)

                if next_move == None:
                    draw_counter += 1
                    break

                self._move(next_move)


                if self.isCheckmate():
                    white_counter += 1
                    break

                if self._isStaleman():
                    draw_counter += 1
                    break

                self.player = not self.player
                self.aichess.depthMax = depth_2
                next_move = self._getNextMove(algorithmB)
                self._move(next_move)

                if next_move == None:
                    draw_counter += 1
                    break

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
        print("|| Depth Whites =", depth_1, "||")
        print("|| Depth Blacks =", depth_2, "||")

        return white_counter, black_counter, draw_counter


import matplotlib.pyplot as plt
if __name__ == "__main__":
    print("Simulation innitialized")
    simulator = Simulate()
    algorithm = {"minmax":1, "alphabeta":2, "expectimax":3}
    depth = 4
    aW = algorithm["minmax"]
    aB = algorithm["minmax"]
    times = 10
    #simulator.simulate(depth, depth, times, aW, aB)

    for i in range(2, 5):
        for j in range(1,6):
            aW = algorithm["alphabeta"]
            aB = algorithm["alphabeta"]
            sizes = list(simulator.simulate(i, j, times, aW, aB))
            labels = ['White ' + str(round(100*sizes[0]/float(times), 2)) + "%", 'Black ' + str(round(100*sizes[1]/float(times), 2)) + "%", 'Draw '+ str(round(100*sizes[2]/float(times), 2)) + "%"]
            colors = ['gold', 'yellowgreen', 'lightcoral']
            patches, texts = plt.pie(sizes, colors=colors, startangle=90)
            plt.legend(patches, labels, loc="best")
            plt.title("Depth Whites = " + str(i) + " | Depth Blacks = " + str(j))
            plt.axis('equal')
            plt.tight_layout()
            plt.show()


