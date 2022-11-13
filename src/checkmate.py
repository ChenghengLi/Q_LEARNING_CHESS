
import copy

class Checkmate:
    def __init__(self, board, player = True):
        self.innital_stateW = copy.deepcopy(board.currentStateW)
        self.innital_stateB = copy.deepcopy(board.currentStateB)
        self.setBoard(board)
        self.setPlayer(player)

    def setBoard(self, board):
        self.board = board

    def setPlayer(self, player):
        self.player = player

    def _getKing(self):
        current_state = self.board.currentStateW if not self.player else self.board.currentStateB
        for piece in current_state:
            if piece[2] == 6 or piece[2] == 12:
                return piece[0:2]

    def _reset(self):
        self.board.updateState(self.innital_stateW, self.innital_stateB)

    def _checkPositions(self, state):
        positions = set()
        for piece in state:
            (x, y) = piece[0:2]
            if (x, y) in positions:
                return False
            positions.add((x, y))
        return True

    def _getChildren(self, state, player):
        if player:
            return self.board.getListNextStatesW(state)
        else:
            return self.board.getListNextStatesB(state)



    def _move(self, stateW, stateB):
        self.board.updateState(stateW, stateB)

    def isCheck(self, state):
        o_king_y, o_king_x = self._getKing()
        children = self._getChildren(state, self.player)
        for child in children:
            if not self._checkPositions(child):
                continue
            my_moves = set([(e[0], e[1]) for e in child])
            if (o_king_x, o_king_y) in my_moves:
                return True
        return False


    def isCheckmate(self):
        if not self.isCheck(self.board.currentStateW):
            return False
        children = self._getChildren(self.board.currentStateW, not self.player)
        for child in children:
            if not self._checkPositions(child):
                continue
            if self.player:
                self._move(child, self.board.currentStateB)
            else:
                self._move(self.board.currentStateW, child)
            if not self.isCheck(self.board.currentStateW):
                self._reset()
                return False
        self._reset()
        return True
