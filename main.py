import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
import matplotlib.pyplot as plt


class Decision:
    def __init__(self, data, weights, method):
        self.data = pd.read_csv(data, header=None).values
        self.weights = pd.read_csv(weights, header=None).values[0]
        self.method = method
        self.weights_for_criteria = [1] * len(self.data[1])

        self.min_or_max = [0] * len(self.data)  # 1 = maximiser et 0 = minimiser, par défaut on minimise

    def change_to_max(self, index: list[int]):
        for i in index:
            self.min_or_max[i] = 1

    def aggregate_criteria(self):
        pass

    def assign_weight(self, values: list[int]):
        for i, value in enumerate(values):
            self.weights_for_criteria[i] = value

    def weighted_sum(self):
        normalized_data = StandardScaler().fit_transform(self.data)
        print(normalized_data)
        # implementing of a weighted sum method with the normalized data, each line of the data is a candidate and
        # each column is a criterion
        result = np.dot(self.weights, normalized_data)
        print(f'result : {result}')
        G = nx.DiGraph()
        # Ajout des nœuds au graphe - on suppose que chaque candidat est représenté par son indice dans la liste des result
        for i in range(len(result)):
            G.add_node(i, weight=result[i])

        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if (result[i] > result[j]):
                    G.add_edge(i, j)
                elif (result[j] > result[i]):
                    G.add_edge(j, i)

        # Utilisez matplotlib pour afficher le graphe
        nx.draw(G, with_labels=True)
        plt.title('Graphe de dominance, somme pondérée')
        plt.show()
        plt.savefig('figs/graph.png')

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
