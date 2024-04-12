import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse
import networkx as nx
import matplotlib.pyplot as plt


import networkx as nx
import math
import matplotlib.pyplot as plt

class Decision:
    def __init__(self, method, dataset_name):
        self.divide_weights = 18
        self.name = dataset_name
        match dataset_name:
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

    def assign_weight(self, values: list[int]):
        for i, value in enumerate(values):
            self.weights_for_criteria[i] = value

    def weighted_sum(self):
        normalized_data = StandardScaler().fit_transform(self.data)
        print(normalized_data)
        # implementing of a weighted sum method with the normalized data, each line of the data is a candidate and
        # each column is a criterion
        self.weights = self.weights / self.divide_weights
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
        plt.savefig(f'weighted_sum/{self.name}_graph.png')
        plt.clf()
        plt.cla()
        plt.close()


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
        
        


    def electreIV(self, indice_surclassement=0.6):
        with open('electre4/electre4_'+self.name+'.txt', 'w') as f:
            f.write("\nElectre IV : \n")

            concordance_matrix = [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]
            non_discordance_matrix =  [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]

            # build concordance matrix
            for row in decision.data:
                for i, candidat_1 in enumerate(row):
                    for j, candidat_2 in enumerate(row):
                        concordance_matrix[i][j] = decision.concordance(i, j, self.weights)
            f.write('Concordance matrix : \n')
            f.write(str(concordance_matrix))

            # build discordance matrix
            for row in decision.data:
                for i, candidat_1 in enumerate(row):
                    for j, candidat_2 in enumerate(row):
                        non_discordance_matrix[i][j] = decision.non_discordance(i, j)
            f.write(str(non_discordance_matrix)+'\n')
            f.write('Non Discordance matrix : \n')
            f.write(str(concordance_matrix))

            surclasse = np.zeros((len(concordance_matrix), len(concordance_matrix)))

            for indice in range(10):
                f.write("Seuil : ")
                f.write(str(indice))
                f.write(' ')
                for i in range(len(concordance_matrix)):
                    for j in range(len(concordance_matrix)):
                        if i == j:
                            surclasse[i][j] = None
                        elif concordance_matrix[i][j] > indice/10 and non_discordance_matrix[i][j] == 1:
                            surclasse[i][j] = 1
                        else:
                            surclasse[i][j] = 0

                #Noyaux
                non_surclasse = [True]*len(concordance_matrix)
                for i in range(len(concordance_matrix)):
                    for j in range(len(concordance_matrix)):
                        if surclasse[j][i] == 1:
                            non_surclasse[i] = False

                f.write("Noyau : ")
                for i in range(len(concordance_matrix)):
                    if non_surclasse[i] == True:
                        f.write(str(i)+' ')
                f.write('\n')

                G = nx.DiGraph()

                for i in range(len(surclasse)):
                    G.add_node(i, weight=surclasse[i])

                for i in range(len(surclasse)):
                    for j in range(len(surclasse)):
                        if surclasse[i][j] == 1:
                            G.add_edge(i, j)

                nx.draw(G, with_labels=True)
                plt.savefig('electre4/'+self.name+'_seuil_'+str(indice)+'.png')
                plt.clf()
                plt.cla()
                plt.close()

    def electreIs(self):
        with open('electre1s/electre1s_'+self.name+'.txt', 'w') as f:
            f.write("\nElectre Is : \n")
            concordance_matrix = [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]
            non_discordance_matrix =  [[0]*decision.data.shape[1] for i in range(decision.data.shape[1])]

            # build concordance matrix
            for row in decision.data:
                for i, candidat_1 in enumerate(row):
                    for j, candidat_2 in enumerate(row):
                        concordance_matrix[i][j] = decision.concordance_electreIs(i, j, self.weights)
            f.write('Concordance matrix : \n')
            f.write(str(concordance_matrix))

            # build discordance matrix
            for row in decision.data:
                for i, candidat_1 in enumerate(row):
                    for j, candidat_2 in enumerate(row):
                        non_discordance_matrix[i][j] = decision.non_discordance(i, j)
            f.write('Non Discordance matrix : \n')
            f.write(str(concordance_matrix))

            surclasse = np.zeros((len(concordance_matrix), len(concordance_matrix)))

            for indice in range(10):
                f.write("Seuil : ")
                f.write(str(indice))
                f.write(' ')
                for i in range(len(concordance_matrix)):
                    for j in range(len(concordance_matrix)):
                        if i == j:
                            surclasse[i][j] = None
                        elif concordance_matrix[i][j] > indice/10 and non_discordance_matrix[i][j] == 1:
                            surclasse[i][j] = 1
                        else:
                            surclasse[i][j] = 0

                #Noyaux
                non_surclasse = [True]*len(concordance_matrix)
                for i in range(len(concordance_matrix)):
                    for j in range(len(concordance_matrix)):
                        if surclasse[j][i] == 1:
                            non_surclasse[i] = False

                f.write("Noyau : ")
                for i in range(len(concordance_matrix)):
                    if non_surclasse[i] == True:
                        f.write(str(i)+' ')
                f.write('\n')

                G = nx.DiGraph()

                for i in range(len(surclasse)):
                    G.add_node(i, weight=surclasse[i])

                for i in range(len(surclasse)):
                    for j in range(len(surclasse)):
                        if surclasse[i][j] == 1:
                            G.add_edge(i, j)
                
                nx.draw(G, with_labels=True)
                plt.savefig('electre1s/'+self.name+'_seuil_'+str(indice)+'.png')
                plt.clf()
                plt.cla()
                plt.close()

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
    
    def concordance_electreIs(self, candidat_1_index, candidat_2_index, weights):
        if candidat_1_index == candidat_2_index:
            return None
        concordance = 0
        for index, criteria in enumerate(self.data):
            if self.min_or_max[index] == 1:
                if criteria[candidat_1_index] >= criteria[candidat_2_index]:
                    concordance += weights[index]
                else:
                    difference = criteria[candidat_2_index] - criteria[candidat_1_index]
                    if difference < self.seuils_pref[index] :
                        concordance += (1 - (difference / self.seuils_pref[index])) * weights[index]
            elif self.min_or_max[index] == 0 :
                if criteria[candidat_1_index] <= criteria[candidat_2_index] :
                    concordance += weights[index]
                else:
                    difference = criteria[candidat_1_index] - criteria[candidat_2_index]
                    if difference < self.seuils_pref[index] :
                        concordance += (1 - (difference / self.seuils_pref[index])) * weights[index]

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
    DATASET = 'countries'

    decision = Decision('Weighted Sum', DATASET)

    decision.weighted_sum()
    #
    # decision.electreIV()
    # decision.electreIs()
