import copy


class State:
    def __init__(self, stateW, stateB, player):
        self.stateW = stateW.copy()
        self.stateB = stateB.copy()
        self.player = player
        self.check = False
        self.terminal = False
        self.checkmate = False

    def __str__(self):
        return "State: " + str(self.stateW) + " " + str(self.stateB)
