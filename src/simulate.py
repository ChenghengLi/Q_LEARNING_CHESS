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

        self.aichess = Aichess(TA, True)
        self.currentStateW = copy.deepcopy(aichess.chess.boardSim.currentStateW)
        self.currentStateB = copy.deepcopy(aichess.chess.boardSim.currentStateB)



    def _getNextMoveB(self, algorithm):
        if algorithm == "minmax":
            return self.aichess.minimax_decision(self.currentStateW, self.currentStateB, False)

    def _getNextMoveW(self, algorithm):
        if algorithm == "minmax":
            return self.aichess.minimax_decision(self.currentStateW, self.currentStateB, True)

    def _move(self, next_move, player):

        currentState = self.currentStateW if player else self.currentStateB

        (x_1, y_1) = [e for e in currentState if e not in nextState][0][0:2]
        (x_2, y_2) = [e for e in nextState if e not in currentState][0][0:2]

        kill = False
        nS = None

        if player:
            kill, nS = self.chess.move([x_1, y_1], [x_2, y_2])
            self.aichess.chess.boardSim = copy.deepcopy(self.aichess.chess.board)
            self.currentStateW = copy.deepcopy(next_move)
        else:
            kill, nS = self.chess.move([x_1, y_1], [x_2, y_2])
            self.currentStateB = copy.deepcopy(next_move)
            self.aichess.chess.boardSim = copy.deepcopy(self.aichess.chess.board)
        return kill, nS



    def simulate(self, depth, times, algorithmW, algorithmB):
        self.aichess.depthMax = depth
        black_counter = 0
        white_counter = 0
        for _ in range(times):
            while True:
                next_move = self._getNextMoveW(algorithmW)
                self._move(next_move, True)
                #si jaque mate blanco: white_counter += 1 ; break;
                next_move = self._getNextMoveB(algorithmB)
                self._move(next_move, False)
                # si jaque mate negro: black_counter += 1 ; break;

            self._reset()

if __name__ == "__main__":
    simulator = Simulate()
    algorithm = ["minmax", "alphabeta"]
    depth = 4
    aW = algorithm[0]
    aB = algorithm[0]
    times = 10
    simulator.simulate(depth, times, aW, aB)

