import pandas as pd
import numpy as np

class Decision:
    def __init__(self, data, weights, method):
        self.data = pd.read_csv(data).values
        self.weights = pd.read_csv(weights, header=None).values[0]
        self.method = method

        self.min_or_max = [0] * len(self.data)  # 1 = maximiser et 0 = minimiser, par défaut on minimise

        print(self.data)
        print(self.weights)

    def change_to_max(self, index: list[int]):
        for i in index:
            self.min_or_max[i] = 1

    def aggregate_criteria(self):
        pass

    def weighted_sum(self):
        pass

    def prometheeI(self):
        pass

    def prometheeII(self):
        pass

    def electreIV(self):
        pass

    def electreIs(self):
        pass

    def execute(self):
        if self.method == 'Weighted Sum':
            self.weighted_sum()
        elif self.method == 'Promethee I':
            self.prometheeI()
        elif self.method == 'Promethee II':
            self.prometheeII()
        elif self.method == 'Electre IV':
            self.electreIV()
        elif self.method == 'Electre IS':
            self.electreIs()
        else:
            raise ValueError('Méthode non reconnue')


if __name__ == '__main__':
    decision = Decision('data/donnees.csv', 'data/poids.csv', 'Weighted Sum')

    decision.execute()
