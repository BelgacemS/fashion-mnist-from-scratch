# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2026

# Import de packages externes
import numpy as np
import pandas as pd

import copy

from abc import ABC, abstractmethod

# ---------------------------


# ------------------------ A COMPLETER :
# importation de la librairie abc:
from abc import ABC, abstractmethod

class Classifier(ABC):
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    __nombre_crees: int = 0  # Variable de classe pour compter le nombre de classifiers créés
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__nombre_crees += 1
        self.__ident = Classifier.__nombre_crees  # identifiant du classifieur (unique)
        self.__dimension = input_dimension
        
    def get_dimension(self):
        """ Accesseur de la variable __dimension 
        """
        return self.__dimension
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f'Classifier #{self.__ident} (d{self.__dimension})'
        
    @abstractmethod
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """      
        pass


    @abstractmethod
    def score(self,x) -> float:
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
    
    @abstractmethod
    def predict(self, x) -> int:
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass

    def accuracy(self, desc_set, label_set) -> float:
        """ rend le taux d'exemples bien classés dans le dataset
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        s = 0
        for i in range(len(desc_set)):

            if self.predict(desc_set[i]) == label_set[i]:
                s+=1
        
        return (s/len(desc_set))

# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__k= k
        # les 2 variables suivantes seront utilisées dans la méthode train()
        self.__desc_set= None   
        self.__label_set = None

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        
        return f'ClassifierKNN #{self._Classifier__ident}(d{self.get_dimension()}) k={self.__k}'

    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.__desc_set = desc_set
        self.__label_set =label_set

    def score(self,x) -> float:
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un array
        """
        distances = np.linalg.norm(self.__desc_set - x, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_labels = self.__label_set[sorted_indices[:self.__k]]
        p = np.sum(k_nearest_labels == 1) / self.__k
        return 2 * (p - 0.5)
    
    def predict(self, x) -> int:
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un array
        """
        return 1 if self.score(x) >= 0 else -1
    

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        v = np.random.uniform(-1, 1, input_dimension)
        self.__w = v / np.linalg.norm(v)
        
    def __str__(self) -> str:
        return f'Classifier #{self._Classifier__ident} (d{self.get_dimension()}) - LinAleatoire w= [{", ".join([f" {wi:.6g}" for wi in self.__w[:5]])}]'
     
    def train(self, desc_set, label_set) -> None:
        print("Pas d'apprentissage pour ce classifier !")

    def score(self,x) -> float:
        return np.dot(x, self.__w)
    
    def predict(self, x) -> int:
        return 1 if self.score(x) >= 0 else -1
    

    # ------------------------ A COMPLETER :
class ClassifierKNN_MC(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self,input_dimension,k,nc):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
                - nc (int): nombre de classes
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        ############### A COMPLETER
        raise NotImplementedError("Please Implement this method")
    
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        raise NotImplementedError("Please Implement this method")
    
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x) -> float:
        """ rend un vecteur avec le nombre de voisin de chaque classe
        """
        # Calculer le tableau des distances entre x et les points du set
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x) -> int:
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un array
        """
        raise NotImplementedError("Please Implement this method")
        



class ClassifierPerceptronTME3(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        super().__init__(input_dimension)
        self.__learning_rate = learning_rate
        if init:
            self.__w = np.zeros(input_dimension)
        else:
            self.__w = (2 * np.random.rand(input_dimension) - 1) * 0.001
        if verbose:
            print(f"{super().__str__()}: initialisation (learning rate= {self.__learning_rate}) w= {self.__w}")

    def train_step(self, desc_set, label_set):
        indices = list(range(len(desc_set)))
        np.random.shuffle(indices)
        for i in indices:
            x = desc_set[i]
            y = label_set[i]
            if y * np.dot(x, self.__w) <= 0:
                self.__w += self.__learning_rate * y * x

    def __str__(self) -> str:
        return f"[TME3]{super().__str__()} - Perceptron w= {self.__w}"

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, verbose=False):
        liste_diffs = []
        nb_iter = 0
        for i in range(nb_max):
            w_old = self.__w.copy()
            self.train_step(desc_set, label_set)
            diff = np.linalg.norm(self.__w - w_old)
            liste_diffs.append(diff)
            nb_iter += 1
            if diff < seuil:
                break
        if verbose:
            print(f"train {self} : {nb_iter} appels à train_step")
        return liste_diffs

    def score(self, x):
        return np.dot(x, self.__w)

    def predict(self, x):
        return 1 if self.score(x) > 0 else -1


class ClassifierPerceptron(Classifier):
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        super().__init__(input_dimension)
        self.__learning_rate = learning_rate
        if init:
            self.__w = np.zeros(input_dimension + 1)
        else:
            self.__w = (2 * np.random.rand(input_dimension) - 1) * 0.001
            self.__w = np.append(self.__w, np.random.rand())
        self.__allw = [self.__w.copy()]
        if verbose:
            print(f"{super().__str__()}: initialisation (learning rate= {self.__learning_rate}) w= {self.__w}")

    def train_step(self, desc_set, label_set, stabilised=False):
        indices = list(range(len(desc_set)))
        np.random.shuffle(indices)
        for i in indices:
            x = ClassifierPerceptron._augmente(desc_set[i])
            y = label_set[i]
            if y * np.dot(x, self.__w) <= 0:
                self.__w += self.__learning_rate * y * x
                self.__allw.append(self.__w.copy())

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, stabilised=False, verbose=False):
        liste_diffs = []
        nb_iter = 0
        for i in range(nb_max):
            w_old = self.__w.copy()
            self.train_step(desc_set, label_set)  # jamais stabilised ici
            diff = np.linalg.norm(self.__w - w_old)
            liste_diffs.append(diff)
            nb_iter += 1
            if diff < seuil:
                break
        
        # Stabilisation à la fin
        if stabilised:
            best_w = min(self.__allw, key=lambda w: sum(
                1 for xi, yi in zip(desc_set, label_set)
                if yi * np.dot(ClassifierPerceptron._augmente(xi), w) <= 0
            ))
            self.__w[:] = best_w  # in-place
            self.__allw.append(best_w.copy())
        
        if verbose:
            print(f"train {self} : {nb_iter} appels à train_step")
        return liste_diffs

    def __str__(self):
        return f"{super().__str__()} - Perceptron w= {self.__w}"

    def score(self, x):
        x = ClassifierPerceptron._augmente(x)
        return np.dot(x, self.__w)

    def predict(self, x):
        return 1 if self.score(x) > 0 else -1

    @staticmethod
    def _augmente(x):
        if x.ndim == 1:
            return np.append(x, -1)
        else:
            col = -np.ones((x.shape[0], 1))
            return np.hstack([x, col])

    def get_allw(self):
        return self.__allw


# nouveau (la cellule du TME-04 etait un stub) : classifieur multi-classes un-contre-tous
class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes par strategie un-contre-tous (One-vs-All) """
    def __init__(self, cl_bin):
        # cl_bin : un classifieur binaire NON entraine (sert de modele a copier)
        super().__init__(cl_bin.get_dimension())
        self.cl_bin = cl_bin
        self.classifieurs = []
        self.classes = []

    def train(self, desc_set, label_set):
        self.classes = np.unique(label_set)
        self.classifieurs = []
        for c in self.classes:
            # classe c contre toutes les autres : c -> +1, le reste -> -1
            y_c = np.where(label_set == c, 1, -1)
            clf = copy.deepcopy(self.cl_bin)   # copie independante pour chaque classe
            clf.train(desc_set, y_c)
            self.classifieurs.append(clf)

    def score(self, x):
        # un score par classe
        return np.array([clf.score(x) for clf in self.classifieurs])

    def predict(self, x):
        # on choisit la classe dont le classifieur binaire est le plus confiant
        return self.classes[np.argmax(self.score(x))]


# Fonctions pour les arbres de decision

def shannon(P, k=2):
    # entropie de Shannon en base k (k = nb de classes -> valeur dans [0,1])
    if k <= 1:
        return 0.0
    h = 0.0
    for p in P:
        if p > 0:
            h -= p * np.log2(p) / np.log2(k)
    return h

def classe_majoritaire(Y):
    valeurs, nb = np.unique(Y, return_counts=True)
    return valeurs[np.argmax(nb)]

def entropie(Y, k=None):
    valeurs, nb = np.unique(Y, return_counts=True)
    # par defaut on normalise par le nombre de classes present (ok en multi-classes)
    if k is None:
        k = len(valeurs)
    P = nb / len(Y)
    return shannon(list(P), k)


class NoeudCategoriel:
    def __init__(self, num_att=-1, nom=''):
        self.__attribut = num_att
        if (nom == ''):
            self.__nom_attribut = 'att_'+str(num_att)
        else:
            self.__nom_attribut = nom
        self.__Les_fils = None
        self.__classe = None

    def est_feuille(self):
        return self.__Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__Les_fils[valeur] = Fils

    def ajoute_feuille(self, classe):
        self.__classe = classe
        self.__Les_fils = None

    def classifie(self, exemple):
        if self.est_feuille():
            return self.__classe
        if exemple[self.__attribut] in self.__Les_fils:
            return self.__Les_fils[exemple[self.__attribut]].classifie(exemple)
        else:
            print('\t*** Warning: attribut ',self.__nom_attribut,' -> Valeur inconnue: ',exemple[self.__attribut])
            return None

    def compte_feuilles(self):
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.__Les_fils:
            total += self.__Les_fils[noeud].compte_feuilles()
        return total

    def to_graph(self, g, prefixe='A'):
        if self.est_feuille():
            g.node(prefixe, str(self.__classe), shape='box')
        else:
            g.node(prefixe, self.__nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.__Les_fils.items():
                sous_arbre.to_graph(g, prefixe+str(i))
                g.edge(prefixe, prefixe+str(i), str(valeur))
                i = i+1
        return g


def construit_AD(X, Y, epsilon, LNoms=[], verbose=False):
    entropie_ens = entropie(Y)
    if verbose:
        print(f"Construction: entropie classe {entropie_ens:1.5f}")
    if (entropie_ens <= epsilon):
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        if verbose:
            print(f"\tajout d'une feuille avec la classe {classe_majoritaire(Y)}")
    else:
        min_entropie = float('inf')
        i_best = -1
        Xbest_valeurs = None

        for j in range(X.shape[1]):
            valeurs_j = np.unique(X[:, j])
            h_cond = 0.0
            for v in valeurs_j:
                Y_v = Y[X[:, j] == v]
                h_cond += len(Y_v) / len(Y) * entropie(Y_v)

            if verbose:
                nom = str(j)
                if LNoms != []:
                    nom = LNoms[j]
                print(f"\tattribut {nom}: entropie= {h_cond:1.5f}")

            if h_cond < min_entropie:
                min_entropie = h_cond
                i_best = j
                Xbest_valeurs = valeurs_j

        if verbose:
            nom = str(i_best)
            if LNoms != []:
                nom = LNoms[i_best]
            print(f"\tMeilleur: {nom}: entropie= {min_entropie:1.5f}")
        if len(LNoms) > 0:
            noeud = NoeudCategoriel(i_best, LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            if verbose:
                print(f"\tdescente pour {v}")
            noeud.ajoute_fils(v, construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v], epsilon, LNoms, verbose))
    return noeud


# repris du TME-08 : arbres de decision numeriques (attributs continus)

def discretise(m_desc, m_class, num_col, verbose=False):
    """ cherche le seuil de coupure sur la colonne num_col qui minimise l'entropie
        rend ((seuil, entropie), (liste_coupures, liste_entropies))
    """
    # liste triee des valeurs differentes de la colonne
    l_valeurs = np.unique(m_desc[:,num_col])
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    best_seuil = None
    best_entropie = float('Inf')
    liste_entropies = []
    liste_coupures = []
    nb_exemples = len(m_class)
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        # entropie de la coupure = moyenne ponderee des deux cotes
        val_entropie_inf = entropie(cl_inf)
        val_entropie_sup = entropie(cl_sup)
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        if verbose:
            print(f"Discretise: coupure en {v} -> entropie de {val_entropie:1.4f}")
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    if verbose:
        print(f"Discretise: *** meilleure coupure en {best_seuil} avec une entropie de {best_entropie:1.4f} ***")
    return (best_seuil, best_entropie), (liste_coupures, liste_entropies)


def partitionne(m_desc, m_class, n, s):
    """ partage le dataset selon la colonne n et le seuil s : (<= s) puis (> s) """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))


class NoeudNumerique:
    """ Noeud d'un arbre de decision pour attributs numeriques (coupure par seuil) """
    def __init__(self, num_att=-1, nom=''):
        self.__attribut = num_att
        if (nom == ''):
            self.__nom_attribut = 'att_'+str(num_att)
        else:
            self.__nom_attribut = nom
        self.__seuil = None
        self.__Les_fils = None
        self.__classe = None

    def est_feuille(self):
        return self.__Les_fils == None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__seuil = val_seuil
        self.__Les_fils['inf'] = fils_inf
        self.__Les_fils['sup'] = fils_sup

    def ajoute_feuille(self, classe):
        self.__classe = classe
        self.__Les_fils = None

    def classifie(self, exemple):
        if self.est_feuille():
            return self.__classe
        if float(exemple[self.__attribut]) <= float(self.__seuil):
            return self.__Les_fils['inf'].classifie(exemple)
        return self.__Les_fils['sup'].classifie(exemple)

    def compte_feuilles(self):
        if self.est_feuille():
            return 1
        return self.__Les_fils['inf'].compte_feuilles() + self.__Les_fils['sup'].compte_feuilles()

    def to_graph(self, g, prefixe='A'):
        if self.est_feuille():
            g.node(prefixe, str(self.__classe), shape='box')
        else:
            g.node(prefixe, str(self.__nom_attribut))
            self.__Les_fils['inf'].to_graph(g, prefixe+"g")
            self.__Les_fils['sup'].to_graph(g, prefixe+"d")
            g.edge(prefixe, prefixe+"g", '<='+ str(self.__seuil))
            g.edge(prefixe, prefixe+"d", '>'+ str(self.__seuil))
        return g


def construit_AD_num(X, Y, epsilon, LNoms=[], verbose=False):
    (nb_lig, nb_col) = X.shape
    entropie_classe = entropie(Y)
    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        # noeud pur (ou plus assez d'exemples) -> feuille
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0
        i_best = -1
        Xbest_seuil = None
        Xbest_tuple = None
        # on cherche l'attribut et le seuil qui maximisent le gain d'information
        for i in range(nb_col):
            (seuil, entropie_cond), _ = discretise(X, Y, i)
            if seuil is not None:
                gain = entropie_classe - entropie_cond
                if gain > gain_max:
                    gain_max = gain
                    i_best = i
                    Xbest_seuil = seuil
                    Xbest_tuple = partitionne(X, Y, i, seuil)
        if i_best != -1:
            if len(LNoms) > 0:
                noeud = NoeudNumerique(i_best, LNoms[i_best])
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil,
                              construit_AD_num(left_data, left_class, epsilon, LNoms),
                              construit_AD_num(right_data, right_class, epsilon, LNoms))
        else:
            # aucun gain possible -> feuille
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
    return noeud


class ClassifierArbreNumerique(Classifier):
    """ Arbre de decision pour des attributs numeriques """
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        super().__init__(input_dimension)
        self.__epsilon = epsilon
        self.__LNoms = LNoms
        self.__racine = None

    def __str__(self):
        return super().__str__()+' - ArbreNumerique ['+str(super().get_dimension())+'] eps='+str(self.__epsilon)

    def train(self, desc_set, label_set, verbose=False):
        self.__racine = construit_AD_num(desc_set, label_set, self.__epsilon, self.__LNoms, verbose)

    def score(self, x):
        pass

    def predict(self, x):
        return self.__racine.classifie(x)

    def accuracy(self, desc_set, label_set):
        nb_ok = sum(1 for i in range(len(desc_set)) if self.predict(desc_set[i]) == label_set[i])
        return nb_ok / len(desc_set)

    def number_leaves(self):
        return self.__racine.compte_feuilles()

    def affiche(self, GTree):
        self.__racine.to_graph(GTree)

