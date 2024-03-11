import pandas as pd
import numpy as np

class Decision:
    def __init__(self, data, weights, method):
        self.data = pd.read_csv(data, header=None).values
        self.weights = pd.read_csv(weights, header=None).values[0]
        self.method = method

        self.min_or_max = [0] * len(self.data)  # 1 = maximiser et 0 = minimiser, par défaut on minimise

        # print(self.data)
        # print(self.weights)

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
        
    
    def dominates(self, candidat_1, candidat_2):
        best_count_c1 = 0
        best_count_c2 = 0
        print(self.data[:3, :])
        for index, row in enumerate(self.data[:3, :]):
            if row[candidat_1] > row[candidat_2]:
                if self.min_or_max[index] == 1 : 
                    best_count_c1 += 1
                else :
                    best_count_c2 += 1   
            elif row[candidat_1] < row[candidat_2]:
                if self.min_or_max[index] == 0 : 
                    best_count_c1 += 1
                else :
                    best_count_c2 += 1  

        print(best_count_c1)
        print(best_count_c2)
            
        if best_count_c1 > best_count_c2:
            return candidat_1
        elif best_count_c1 < best_count_c2:
            return candidat_2
        else:
            return None
        

if __name__ == '__main__':
    decision = Decision('data/donnees.csv', 'data/poids.csv', 'Weighted Sum')

    decision.execute()

    print(decision.dominates(2, 1))