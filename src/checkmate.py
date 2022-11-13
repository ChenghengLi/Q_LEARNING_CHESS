
import copy

class Checkmate:
    def __init__(self, board, player = True):
        self.setBoard(board)
        self.setPlayer(player)

    def setBoard(self, board):
        self.board = board

    def setPlayer(self, player):
        self.player = player



    def _getKing(self, state):
        for piece in state:
            if piece[2] == 6 or piece[2] == 12:
                return piece[0:2]

    def _reset(self, stateW, stateB):
        self.board.updateState(stateW, stateB)

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

    def isCheck(self, stateW, stateB):
        o_king_y, o_king_x = self._getKing(stateW if not self.player else stateB)
        children = self._getChildren(stateW if self.player else stateB, self.player)
        for child in children:
            if not self._checkPositions(child):
                continue
            my_moves = set([(e[0], e[1]) for e in child])
            if (o_king_x, o_king_y) in my_moves:
                return True
        return False

    def _checkKing(self, state):
        pieces = set(x[2] for x in state)
        return 6 in pieces or 12 in pieces


    def isCheckmate(self, stateW, stateB):

        if not self._checkKing(stateW) or not self._checkKing(stateB):
            return False

        assert  stateW == self.board.currentStateW and stateB == self.board.currentStateB
        innitialW = copy.deepcopy(stateW)
        innitialB = copy.deepcopy(stateB)
        if not self.isCheck(stateW, stateB):
            self._reset(innitialW, innitialB)
            return False
        children = self._getChildren(stateW if not self.player else stateB, not self.player)
        for child in children:
            if not self._checkPositions(child):
                continue
            if self.player:
                self._move(child, stateB)
            else:
                self._move(stateW, child)
            if not self.isCheck(child):
                self._reset(innitialW, innitialB)
                return False
        self._reset(innitialW, innitialB)
        return True
