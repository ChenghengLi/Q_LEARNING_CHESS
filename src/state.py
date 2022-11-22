import copy


class State:
    def __init__(self, stateW, stateB, parent, depth, player):
        self.stateW = stateW.copy()
        self.stateB = stateB.copy()
        self.parent = parent
        self.depth = depth
        self.player = player
        self.check = False

    def __str__(self):
        return "State: " + str(self.stateW) + " " + str(self.stateB)
