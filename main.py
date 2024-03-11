import pandas as pd
import numpy as np

class Decision:
    def __init__(self, data, weights, method):
        self.data = pd.read_csv(data, header=None).values
        self.weights = pd.read_csv(weights, header=None).values[0]
        self.method = method

        self.min_or_max = [0] * len(self.data)  # 1 = maximiser et 0 = minimiser, par défaut on minimise

        print(self.data)
        print(self.weights)
        
        #pour test td3 :
        self.change_to_max([1,5])

    def change_to_max(self, index: list[int]):
        for i in index:
            self.min_or_max[i] = 1

    def aggregate_criteria(self):
        pass

    def weighted_sum(self):
        pass


    def print_matrix_prometheeI(self, matrix):
        print("   ", end='  ')
        for i in range(len(matrix[0])-1):
            print("A"+str(i+1), end='  ')
        print("phi+")
        
        for i in range(len(matrix[0])):
            if i < len(matrix[0])-1:
                print("A"+str(i+1), end='   ')
            else :
                print("phi-", end= ' ')
            for j in range(len(matrix)):
                print("{:.1f}".format(matrix[i][j]), end=' ')
            print("\n")



    def prometheeI(self):
        #version où le gagnant prend toujours tout le poids correspondant
        
        nbCandidates = len(self.data[0])
        nbCriteria = len(self.data)
        results = np.zeros((nbCandidates+1,nbCandidates+1))
        for i in range(nbCandidates):
            for j in range(nbCandidates):
                if i == j : #diagonale
                    continue
                else :
                    for k in range(nbCriteria):
                        #print(self.data[k][i], self.data[k][j])
                        if self.data[k][i] < self.data[k][j] :
                            if self.min_or_max[k] == 1 :
                                results[j][i] = results[j][i] + self.weights[k]
                            else :
                                results[i][j] = results[i][j] + self.weights[k]
        
        fluxPlus = []
        fluxMoins = []
        
        for i in range(len(results)):
            lineSum = 0
            for j in range(len(results)):
                lineSum = lineSum + results[i][j]
            results[i][len(results)-1] = lineSum
            fluxPlus.append(lineSum)

        for j in range(len(results)-1):
            columnSum = 0
            for i in range(len(results)):
                columnSum = columnSum + results[i][j]
            results[len(results)-1][j] = columnSum
            fluxMoins.append(columnSum)
        
        #todo : les classement pour les différents flux de manière automatique
        
        
        return results       
        
        
        
        
        

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
            results = self.prometheeI()
            self.print_matrix_prometheeI(results)
        elif self.method == 'Promethee II':
            self.prometheeII()
        elif self.method == 'Electre IV':
            self.electreIV()
        elif self.method == 'Electre IS':
            self.electreIs()
        else:
            raise ValueError('Méthode non reconnue')


if __name__ == '__main__':
    #decision = Decision('data/donnees.csv', 'data/poids.csv', 'Promethee I')
    decision = Decision('data/td3 - data.csv', 'data/td3 - weight.csv', 'Promethee I')


    decision.execute()
