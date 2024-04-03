import pandas as pd
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt


class Decision:
    def __init__(self, method, dataset_name):
        self.divide_weights = 18

        match dataset_name:
            case 'waste':
                self.data = pd.read_csv('data/waste_management/donnees.csv', header=None).values
                self.weights = pd.read_csv('data/waste_management/poids.csv', header=None).values[0]
                self.min_or_max = [0] * len(self.data) 
                self.veto_matrix = [3]*len(self.data)
                self.weights = self.weights/self.divide_weights
            case 'td3':
                self.data = pd.read_csv('data/td3/donnees.csv', header=None).values
                self.weights = pd.read_csv('data/td3/poids.csv', header=None).values[0]
                self.veto_matrix = [45, 29, 550, 6, 4.5, 4.5]
                self.min_or_max = [0] * len(self.data)
                self.change_to_max([1, 5])

        print(self.min_or_max)

        self.method = method

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

        indice = 0.6
        print(len(concordance_matrix))
        surclasse = np.zeros((len(concordance_matrix), len(concordance_matrix)))

        for i in range(len(concordance_matrix)):
            for j in range(len(concordance_matrix)):
                if i == j:
                    surclasse[i][j] = None
                elif concordance_matrix[i][j] > indice and non_discordance_matrix[i][j] == 1:
                    surclasse[i][j] = 1
                else:
                    surclasse[i][j] = 0
        print('\n')
        print(surclasse)

        G = nx.DiGraph()

        for i in range(len(surclasse)):
            G.add_node(i, weight=surclasse[i])

        for i in range(len(surclasse)):
            for j in range(len(surclasse)):
                if surclasse[i][j] == 1:
                    G.add_edge(i, j)

        nx.draw(G, with_labels=True)
        plt.title('salut')
        plt.show()

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

    #Possible values for DATASET : waste, td3
    DATASET = 'td3'

    decision = Decision('Weighted Sum', DATASET)

    decision.electreIV()
