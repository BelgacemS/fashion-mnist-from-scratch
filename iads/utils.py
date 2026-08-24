# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2026

# import externe
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[array, array]
        Hyp: n est pair
        d: nombre de dimensions de la description
        nc: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    # #####################
    # COMPLETER ICI (et enlever la ligne suivante qui lève une exception)
    # #####################

    x1=np.random.uniform(binf,bsup,(2*nc,d))

    x2=np.concatenate((-1*np.ones(nc,dtype=np.int_),np.ones(nc,dtype=np.int_)))
    
    return (x1,x2)

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nc):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    #####################################
    # COMPLETER ICI (puis enlever la ligne suivante)
    #####################################

    x1=np.vstack((np.random.multivariate_normal(negative_center,negative_sigma,nc),np.random.multivariate_normal(positive_center,positive_sigma,nc)))

    x2=np.concatenate((-1*np.ones(nc,dtype=np.int_),np.ones(nc,dtype=np.int_)))
    
    return (x1,x2)

def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """

    liste_des_indices_positifs = np.where(label_set == 1)[0]
    liste_des_indices_negatif = np.where(label_set == -1)[0]


    indices_pos_train = random.sample(list(liste_des_indices_positifs),n_pos)
    indices_neg_train = random.sample(list(liste_des_indices_negatif),n_neg)
    
    tous_les_indices = [i for i in range(len(desc_set))]
    indices_train = indices_pos_train + indices_neg_train
    indices_test = [i for i in tous_les_indices if i not in indices_train]  

    return (desc_set[indices_train], label_set[indices_train]), (desc_set[indices_test], label_set[indices_test])


def plot2DSet(desc,labels,nom_dataset= "Dataset", avec_grid=True):    
    """ array * array * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille, False sinon
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """

    #####################################
    # COMPLETER ICI (puis enlever la ligne suivante)
    #####################################
    desc_nega=desc[labels==-1]
    desc_posi=desc[labels==1]

    plt.scatter(desc_nega[:,0],desc_nega[:,1],marker='o', color="red", label='classe -1') # 'o' rouge pour la classe -1
    plt.scatter(desc_posi[:,0],desc_posi[:,1],marker='x', color="blue", label='classe +1') # 'x' bleu pour la classe +1

    plt.title("data2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(visible=avec_grid)  
    plt.show()

def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


def plot2DTrainTestSet(d_train,l_train, d_test,l_test, nom_dataset= "Dataset", avec_grid=True):    
    """ array * array array * array * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille, False sinon
        la fonction doit utiliser les couleurs suivantes:
        - pour les données d'apprentissage : la couleur 'red' pour la classe -1 et 'blue' pour la +1
        - pour les données de test : la couleur 'jaune' pour la classe -1 et 'verte' pour la +1
    """

    ###############################################
    #################### COMPLETER 
    ###############################################


    train_neg = d_train[l_train == -1]

    train_pos = d_train[l_train == 1]

    test_neg = d_test[l_test == -1]

    test_pos = d_test[l_test == 1]

    plt.scatter(train_neg[:,0],train_neg[:,1],marker='o', color="red", label='classe -1')
    plt.scatter(train_pos[:,0],train_pos[:,1],marker='o', color="blue", label='classe +1') 

    plt.scatter(test_neg [:,0],test_neg[:,1],marker='x', color="yellow", label='classe -1') 
    plt.scatter(test_pos[:,0],test_pos[:,1],marker='x', color="green", label='classe +1') 

    
    
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


# ------------------------ COMPLETER LES INSTRUCTIONS DANS CETTE BOITE 
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    sigma = np.sqrt(var)
    points_per_cluster =n 
    
    
    c1 =np.random.randn(points_per_cluster,2)*sigma+np.array([0,0])
    c2 =np.random.randn(points_per_cluster,2)*sigma+np.array([1,1])
    c3 =np.random.randn(points_per_cluster,2)*sigma+np.array([1,0])
    c4 =np.random.randn(points_per_cluster,2)*sigma+np.array([0,1])
    
    
    data_xor = np.vstack((c1,c2,c3,c4))
    
    labels_neg= -np.ones(points_per_cluster*2)
    labels_pos = np.ones(points_per_cluster*2)
    
    label_xor= np.hstack((labels_neg, labels_pos))
    
    return data_xor,label_xor


# nouveau (pas de version reutilisable dans les TME) : reduction de dimension par ACP
def acp(X, k):
    """ Analyse en composantes principales.
        Centre X, diagonalise la covariance, projette sur les k axes de plus forte variance.
        Renvoie (X_proj, infos) ; infos sert a reprojeter de nouvelles donnees.
    """
    X = np.asarray(X, dtype=float)
    moyenne = X.mean(axis=0)
    Xc = X - moyenne
    # matrice de covariance (une variable par colonne)
    C = np.cov(Xc, rowvar=False)
    # eigh : matrice symetrique -> valeurs propres croissantes, vecteurs propres orthonormes
    valeurs, vecteurs = np.linalg.eigh(C)
    # on remet par variance decroissante
    ordre = np.argsort(valeurs)[::-1]
    valeurs = valeurs[ordre]
    vecteurs = vecteurs[:, ordre]
    composantes = vecteurs[:, :k]          # les k axes principaux
    X_proj = Xc @ composantes
    # les valeurs propres sont les variances le long des axes -> ratio explique
    ratio = valeurs / valeurs.sum()
    infos = {
        "moyenne": moyenne,
        "composantes": composantes,
        "variance_expliquee": ratio[:k],
        "variance_cumulee": np.cumsum(ratio)[:k],
    }
    return X_proj, infos


def applique_acp(X, infos):
    """ projette de nouvelles donnees avec la moyenne et les composantes apprises sur le train """
    X = np.asarray(X, dtype=float)
    return (X - infos["moyenne"]) @ infos["composantes"]
