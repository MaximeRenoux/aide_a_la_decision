import pandas as pd
import numpy as np
import argparse

class Decision:
    def __init__(self, data, weights, method, dataset_name):
        self.divide_weights = 18
        self.data = pd.read_csv(data, header=None).values
        self.weights = pd.read_csv(weights, header=None).values[0]
        self.veto_matrix = [0]*len(self.data)
        if dataset_name == 'waste':
            self.veto_matrix = [3]*len(self.data)
            self.weights = self.weights/self.divide_weights

        self.method = method
        
        self.min_or_max = [0] * len(self.data)  # 1 = maximiser et 0 = minimiser, par défaut on minimise

        # print(self.data)
        # print(self.weights)

    def set_discordance_matrix(self, value):
        for i in enumerate(self.discordance_matrix):
            self.discordance_matrix[i] = value

    def set_discordance(self, index, value):
        self.discordance_matrix[index] = value

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

        concordance_matrix = [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]
        non_discordance_matrix =  [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]

        # build concordance matrix
        for row in decision.data:
            for i, candidat_1 in enumerate(row):
                for j, candidat_2 in enumerate(row):
                    concordance_matrix[i][j] = decision.concordance(i, j, self.weights)
        print(concordance_matrix)

        # build discordance matrix
        for row in decision.data:
            for i, candidat_1 in enumerate(row):
                for j, candidat_2 in enumerate(row):
                    non_discordance_matrix[i][j] = decision.non_discordance(i, j)
        print(non_discordance_matrix)

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
        
    
    def concordance(self, candidat_1_index, candidat_2_index, weights):
        if candidat_1_index == candidat_2_index:
            return None
        concordance = 0
        for index, criteria in enumerate(self.data):
            if criteria[candidat_1_index] >= criteria[candidat_2_index] and self.min_or_max[index] == 1 : 
                concordance += weights[index]
            if criteria[candidat_1_index] <= criteria[candidat_2_index] and self.min_or_max[index] == 0 : 
                concordance += weights[index]

        return concordance
        
    def non_discordance(self, candidat_1_index, candidat_2_index):
        if candidat_1_index == candidat_2_index:
            return None
        for index, criteria in enumerate(self.data):
            if self.min_or_max[index] == 0 : 
                if criteria[candidat_1_index] > criteria[candidat_2_index] and criteria[candidat_1_index]-criteria[candidat_2_index] >= self.veto_matrix[index]:
                    return 0
            elif self.min_or_max[index] == 1 :
                if criteria[candidat_1_index] < criteria[candidat_2_index] and criteria[candidat_2_index]-criteria[candidat_1_index] >= self.veto_matrix[index]:
                    return 0
        return 1
    

if __name__ == '__main__':


    decision = Decision('data/donnees.csv', 'data/poids.csv', 'Weighted Sum', 'waste')

    decision.electreIV()
