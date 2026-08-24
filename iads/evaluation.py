# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# ------------------------

def validation_croisee(C, DS, nb_iter, verbose=False):
    """ Validation croisée en nb_iter paquets
        C : classifieur
        DS : tuple (desc, label)
        nb_iter : nombre de paquets
    """
    desc, label = DS
    n = len(desc)

    # on mélange les indices
    indices = np.arange(n)
    np.random.shuffle(indices)

    # on decoupe en nb_iter paquets qui couvrent tout le jeu (meme si n % nb_iter != 0)
    paquets = np.array_split(indices, nb_iter)

    res = []
    if verbose:
        print("------ affichage validation croisée (verbose)")

    for i in range(nb_iter):
        test_idx = paquets[i]
        train_idx = np.concatenate([paquets[j] for j in range(nb_iter) if j != i])

        desc_train = desc[train_idx]
        label_train = label[train_idx]
        desc_test = desc[test_idx]
        label_test = label[test_idx]

        C.train(desc_train, label_train)

        acc_test = C.accuracy(desc_test, label_test)

        if verbose:
            acc_train = C.accuracy(desc_train, label_train)
            print(f"Itération {i+1}: taille base app.= {len(train_idx)}\ttaille base test= {len(test_idx)}")
            print(f"\tTaux de bonne classif train:\t{acc_train:.4f}")
            print(f"\tTaux de bonne classif test:\t{acc_test:.4f}")

        res.append(acc_test)

    if verbose:
        print("------ fin affichage validation croisée")

    return (res, np.mean(res), np.std(res))


# nouveau (la cellule du TME-04 etait un stub) : validation croisee stratifiee
def crossval_strat(X, Y, n_iterations, iteration):
    """ rend (Xapp, Yapp, Xtest, Ytest) pour le pli numero 'iteration'
        en gardant la meme proportion de classes dans chaque paquet
    """
    index_test = []
    for c in np.unique(Y):
        # on coupe les indices de la classe c en n_iterations paquets
        idx_c = np.where(Y == c)[0]
        paquets = np.array_split(idx_c, n_iterations)
        index_test += list(paquets[iteration])
    index_test = np.array(index_test)
    # le reste sert a l'apprentissage
    masque = np.ones(len(Y), dtype=bool)
    masque[index_test] = False
    return X[masque], Y[masque], X[index_test], Y[index_test]


def validation_croisee_strat(C, DS, nb_iter):
    """ lance la validation croisee stratifiee sur tous les plis
        rend (liste des accuracies, moyenne, ecart-type)
    """
    X, Y = DS
    perfs = []
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        clf = copy.deepcopy(C)          # un classifieur neuf a chaque pli
        clf.train(Xapp, Yapp)
        perfs.append(clf.accuracy(Xtest, Ytest))
    return perfs, np.mean(perfs), np.std(perfs)


# nouveau : CV stratifiee SANS FUITE (pretraitement appris dans chaque pli)
def validation_croisee_strat_pretrait(C, DS, nb_iter, pretraitement):
    """ CV stratifiee ou le pretraitement est appris a l'interieur de chaque pli.
        pretraitement(Xapp, Xtest) -> (Xapp_t, Xtest_t) : apprend la transformation
        sur le train du pli SEUL puis l'applique au train et au test du pli.
        Evite la fuite de donnees (le pli de test ne participe pas a l'apprentissage
        de la transformation).
    """
    X, Y = DS
    perfs = []
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        Xapp_t, Xtest_t = pretraitement(Xapp, Xtest)   # appris sur le train du pli seul
        clf = copy.deepcopy(C)
        clf.train(Xapp_t, Yapp)
        perfs.append(clf.accuracy(Xtest_t, Ytest))
    return perfs, np.mean(perfs), np.std(perfs)


def predict_all(clf, X):
    """ applique predict sur chaque exemple, rend un array de predictions """
    return np.array([clf.predict(X[i]) for i in range(len(X))])


def matrice_confusion(y_true, y_pred, nb_classes):
    """ rend la matrice de confusion (nb_classes x nb_classes)
        Hyp: les classes sont les entiers 0..nb_classes-1
        M[i,j] = nb d'exemples de vraie classe i predits en classe j
    """
    M = np.zeros((nb_classes, nb_classes), dtype=int)
    for vrai, pred in zip(y_true, y_pred):
        M[int(vrai), int(pred)] += 1
    return M


def affiche_matrice_confusion(M, noms_classes=None):
    """ affiche la matrice de confusion sous forme de heatmap lisible """
    n = len(M)
    if noms_classes is None:
        noms_classes = [str(i) for i in range(n)]
    plt.figure(figsize=(8, 7))
    plt.imshow(M, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(n), noms_classes, rotation=45, ha='right')
    plt.yticks(range(n), noms_classes)
    # on ecrit le nombre dans chaque case
    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(M[i, j]), ha='center', va='center')
    plt.xlabel("classe predite")
    plt.ylabel("vraie classe")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()


def metriques_par_classe(M):
    """ a partir de la matrice de confusion, rend (accuracy_globale, precisions, rappels)
        precision[i] = bien classes i / total predits i
        rappel[i]    = bien classes i / total vrais i
    """
    M = np.array(M)
    nb_classes = len(M)
    accuracy = np.trace(M) / M.sum()
    precisions = np.zeros(nb_classes)
    rappels = np.zeros(nb_classes)
    for i in range(nb_classes):
        col = M[:, i].sum()   # total predits classe i
        lig = M[i, :].sum()   # total vrais classe i
        precisions[i] = M[i, i] / col if col > 0 else 0.0
        rappels[i] = M[i, i] / lig if lig > 0 else 0.0
    return accuracy, precisions, rappels
