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


    def print_matrix_promethee(self, matrix, version):
        #version promethee I ou promethee II
        
        nbCandidates = len(self.data[0])
        
        print("   ", end='  ')
        for i in range(nbCandidates):
            print("A"+str(i+1), end='  ')
        if version == "I":
            print("phi+")
        elif version == "II" :
            print("phi+", end ='  ')
            print("phi")
        
        for i in range(nbCandidates+1):
            if i < nbCandidates:
                print("A"+str(i+1), end='   ')
            else :
                print("phi-", end= ' ')
            for j in range(len(matrix[0])):
                print("{:.1f}".format(matrix[i][j]), end=' ')
            print("\n")



    def promethee(self, version):
        #version "I", prometheeI, version "II", prometheeII, où le gagnant prend toujours tout le poids correspondant
        
        nbCandidates = len(self.data[0])
        nbCriteria = len(self.data)
        if version == "I" :
            results = np.zeros((nbCandidates+1,nbCandidates+1))
        elif version == "II":
            results = np.zeros((nbCandidates+1,nbCandidates+2))
            
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
        
        fluxPlus = {}
        fluxMoins = {}
        
        for i in range(nbCandidates):
            lineSum = 0
            for j in range(nbCandidates):
                lineSum = lineSum + results[i][j]
            results[i][nbCandidates] = lineSum
            fluxPlus[i+1] = lineSum

        for j in range(nbCandidates):
            columnSum = 0
            for i in range(nbCandidates):
                columnSum = columnSum + results[i][j]
            results[nbCandidates][j] = columnSum
            fluxMoins[j+1] = columnSum
            
        sortFluxPlus = sorted(fluxPlus, key=lambda x: fluxPlus[x], reverse=True)
        sortFluxMoins = sorted(fluxMoins, key=lambda x: fluxMoins[x])
        print(sortFluxPlus)
        print(sortFluxMoins)
        
        if version == "II":
            flux = {}
            for i  in range(nbCandidates):
                results[i][nbCandidates+1] = results[i][len(results)-1] - results[nbCandidates][i]
                flux[i+1] = results[i][nbCandidates+1]
            sortFlux = sorted(flux, key=lambda x: flux[x], reverse=True)
            print(sortFlux)
            
        #todo : les classement graphiquement
        #prendre en compte égalité dans classement graphiquement
        
        return results       
        
        


    def electreIV(self):
        pass

    def electreIs(self):
        pass

    def execute(self):
        if self.method == 'Weighted Sum':
            self.weighted_sum()
        elif self.method == 'Promethee I':
            results = self.promethee("I")
            self.print_matrix_promethee(results, "I")
        elif self.method == 'Promethee II':
            results = self.promethee("II")
            self.print_matrix_promethee(results, "II")
        elif self.method == 'Electre IV':
            self.electreIV()
        elif self.method == 'Electre IS':
            self.electreIs()
        else:
            raise ValueError('Méthode non reconnue')


if __name__ == '__main__':
    #decision = Decision('data/donnees.csv', 'data/poids.csv', 'Promethee I')
    decision = Decision('data/td3 - data.csv', 'data/td3 - weight.csv', 'Promethee II')


    decision.execute()
