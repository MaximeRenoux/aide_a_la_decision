import pandas as pd
import numpy as np

import networkx as nx
import math
import matplotlib.pyplot as plt

class Decision:
    def __init__(self, method, dataset_name):    
        case 'waste':
                self.data = pd.read_csv('data/waste_management/donnees.csv', header=None).values
                self.weights = pd.read_csv('data/waste_management/poids.csv', header=None).values[0]
                self.min_or_max = [0] * len(self.data) 
                self.veto_matrix = [3]*len(self.data)
                self.seuils_pref = [2]*len(self.data)
                self.weights = self.weights/self.divide_weights
            case 'td3':
                self.data = pd.read_csv('data/td3/donnees.csv', header=None).values
                self.weights = pd.read_csv('data/td3/poids.csv', header=None).values[0]
                self.veto_matrix = [45, 29, 550, 6, 4.5, 4.5]
                self.seuils_pref = [20,10,200,4,2,2]
                self.min_or_max = [0] * len(self.data)
                self.change_to_max([1, 5])
            case 'countries':
                self.data = pd.read_csv('data/countries/donnees.csv', header=None).values
                self.weights = pd.read_csv('data/countries/poids.csv', header=None).values[0]
                self.veto_matrix = [1000]*len(self.data)
                self.seuils_pref = [2]*len(self.data)
                self.min_or_max = [1] * len(self.data) #1 max, 0 min

        print(self.min_or_max)
        self.method = method
        self.weights_for_criteria = [1] * len(self.data[1])

        # print(self.data)
        # print(self.weights)


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
            
            
    def print_classement(self, flux, sortFlux):
        #flux : dictionnaire avec le numéro du candidat en clé et son résultat en valeur
        #sortFlux : les numéros des candidats triés selon leur résultat
        
        
        previousCandidate = 0
        for i in sortFlux :
            if previousCandidate == 0:
                print(i, end='')
            else :
                if round(flux[i], 2) == round(flux[previousCandidate], 2):
                    print(" = ",i, end='')
                else :
                    print(" -> ", i, end='')
            previousCandidate = i
                
        print("\n")
                        
        
    def gen_graph_classement(self, flux, sortFlux, name):
        G = nx.DiGraph()
        for node in sortFlux :
            G.add_node(node)
            
        previousCandidate = 0
        for i in sortFlux :
            if previousCandidate != 0:
                if round(flux[i], 2) == round(flux[previousCandidate], 2):
                    G.add_edge(previousCandidate, i)
                    G.add_edge(i, previousCandidate)
                else :
                    G.add_edge(previousCandidate, i)
            previousCandidate = i    
            
        nx.draw(G, with_labels=True)
        plt.title(name)
        plt.savefig(name+" "+self.name)
        plt.show()
        


    def promethee(self, version, seuils=None):
        #version "I", prometheeI, version "II", prometheeII
        #l arg seuils est une liste avec les seuils de préférence pour chaque critère, facultatif
        
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
                        
                        if seuils != None :
                            ecart = self.data[k][j] - self.data[k][i]
                            #if valeur absolue supérieure au seuil correspondant, le gagnant prend tout
                            #sinon ecart/seuil*weight pour le gagnant
                            if abs(ecart) > self.seuils_pref[k]:
                                if self.data[k][i] < self.data[k][j] :
                                    if self.min_or_max[k] == 1 :
                                        results[j][i] = results[j][i] + self.weights[k]
                                            
                                    else :
                                        results[i][j] = results[i][j] + self.weights[k]
                            else:
                                if self.data[k][i] < self.data[k][j] :
                                    if self.min_or_max[k] == 1 :
                                        results[j][i] = results[j][i] + (ecart/self.seuils_pref[k])*self.weights[k]
                                            
                                    else :
                                        results[i][j] = results[i][j] + (ecart/self.seuils_pref[k])*self.weights[k]
                            
                        
                        elif self.data[k][i] < self.data[k][j] :
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
        print("classement phi+ :")
        self.print_classement(fluxPlus, sortFluxPlus)
        self.gen_graph_classement(fluxPlus, sortFluxPlus, "Graphe classement phi +")
        print("classement phi- :")
        self.print_classement(fluxMoins, sortFluxMoins)
        self.gen_graph_classement(fluxMoins, sortFluxMoins, "Graphe classement phi -")

        
        if version == "II":
            flux = {}
            for i  in range(nbCandidates):
                results[i][nbCandidates+1] = results[i][len(results)-1] - results[nbCandidates][i]
                flux[i+1] = results[i][nbCandidates+1]
            sortFlux = sorted(flux, key=lambda x: flux[x], reverse=True)
            print("classement phi :")
            self.print_classement(flux, sortFlux)
            self.gen_graph_classement(flux, sortFlux, "Graphe classement phi")
            
            
            

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
            
        elif self.method == 'Promethee I seuilsPréférence':
            results = self.promethee("I", self.seuils_pref)
            self.print_matrix_promethee(results, "I")
            
        elif self.method == 'Promethee II':
            results = self.promethee("II")
            self.print_matrix_promethee(results, "II")
            
        elif self.method == 'Promethee II seuilsPréférence':
            results = self.promethee("II", self.seuils_pref)
            self.print_matrix_promethee(results, "II")
            
        elif self.method == 'Electre IV':
            self.electreIV()
        elif self.method == 'Electre IS':
            self.electreIs()
        else:
            raise ValueError('Méthode non reconnue')


if __name__ == '__main__':
    #Possible values for DATASET : waste, td3
    DATASET = 'countries'

    decision = Decision('Weighted Sum', DATASET)

    decision.weighted_sum()
    #
    # decision.electreIV()
    # decision.electreIs()
