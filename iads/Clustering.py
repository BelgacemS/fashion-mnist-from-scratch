# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from abc import ABC, abstractmethod
import scipy.cluster.hierarchy


def normalisation(df):
    # min-max colonne par colonne ; si une colonne est constante on evite la division par zero
    ecart = df.max() - df.min()
    ecart[ecart == 0] = 1
    return (df - df.min()) / ecart


class Distance(ABC):
    def __init__(self, nom):
        self.__nom: str = nom

    @abstractmethod
    def calcule(self, v, M):
        pass

    def __str__(self) -> str:
        return "Distance " + self.__nom


class DistanceEuclidienne(Distance):
    def __init__(self):
        super().__init__("euclidienne")

    def calcule(self, v, M):
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")
        if M.ndim == 1:
            return np.linalg.norm(v - M)
        return np.sqrt(((M - v) ** 2).sum(axis=1))

    def __str__(self) -> str:
        return super().__str__()


class DistanceMinkowski(Distance):
    def __init__(self, p=2):
        super().__init__(f"minkowski(p={p})")
        self.__p = p

    def calcule(self, v, M):
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")
        if M.ndim == 1:
            return (np.abs(v - M) ** self.__p).sum() ** (1 / self.__p)
        return ((np.abs(M - v) ** self.__p).sum(axis=1)) ** (1 / self.__p)

    def __str__(self) -> str:
        return super().__str__()


class Linkage(ABC):
    def __init__(self, nom):
        self.__nom: str = nom

    @abstractmethod
    def calcule(self, G1, G2, verbose=False):
        pass

    def __str__(self) -> str:
        return "Linkage " + self.__nom


class LinkageComplete(Linkage):
    def __init__(self, distance=DistanceEuclidienne()):
        super().__init__("complete")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        G1a = np.array(G1)
        G2a = np.array(G2)
        if G1a.ndim == 1:
            G1a = G1a.reshape(1, -1)
        if G2a.ndim == 1:
            G2a = G2a.reshape(1, -1)
        d_max = 0.0
        for i in range(len(G1a)):
            d = self.__distance.calcule(G1a[i], G2a)
            if np.isscalar(d):
                dmax_i = d
            else:
                dmax_i = np.max(d)
            if dmax_i > d_max:
                d_max = dmax_i
        return d_max

    def __str__(self) -> str:
        return super().__str__() + " (" + self.__distance.__str__() + ")"


class LinkageSimple(Linkage):
    def __init__(self, distance=DistanceEuclidienne()):
        super().__init__("simple")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        G1a = np.array(G1)
        G2a = np.array(G2)
        if G1a.ndim == 1:
            G1a = G1a.reshape(1, -1)
        if G2a.ndim == 1:
            G2a = G2a.reshape(1, -1)
        d_min = float('inf')
        for i in range(len(G1a)):
            d = self.__distance.calcule(G1a[i], G2a)
            if np.isscalar(d):
                dmin_i = d
            else:
                dmin_i = np.min(d)
            if dmin_i < d_min:
                d_min = dmin_i
        return d_min

    def __str__(self) -> str:
        return super().__str__() + " (" + self.__distance.__str__() + ")"


class LinkageAverage(Linkage):
    def __init__(self, distance=DistanceEuclidienne()):
        super().__init__("average")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        G1a = np.array(G1)
        G2a = np.array(G2)
        if G1a.ndim == 1:
            G1a = G1a.reshape(1, -1)
        if G2a.ndim == 1:
            G2a = G2a.reshape(1, -1)
        total = 0.0
        n = 0
        for i in range(len(G1a)):
            d = self.__distance.calcule(G1a[i], G2a)
            if np.isscalar(d):
                total += d
                n += 1
            else:
                total += np.sum(d)
                n += len(d)
        return total / n

    def __str__(self) -> str:
        return super().__str__() + " (" + self.__distance.__str__() + ")"


class LinkageCentroide(Linkage):
    def __init__(self, distance=DistanceEuclidienne()):
        super().__init__("centroide")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        G1a = np.array(G1)
        G2a = np.array(G2)
        if G1a.ndim == 1:
            G1a = G1a.reshape(1, -1)
        if G2a.ndim == 1:
            G2a = G2a.reshape(1, -1)
        c1 = G1a.mean(axis=0)
        c2 = G2a.mean(axis=0)
        return self.__distance.calcule(c1, c2)

    def __str__(self) -> str:
        return super().__str__() + " (" + self.__distance.__str__() + ")"


def CHA_initialise(DF):
    return {i: [i] for i in range(len(DF))}


def CHA_fusionne(DF, P0, linkage, verbose=False):
    cles = list(P0.keys())
    dist_min = float('inf')
    cle1, cle2 = None, None
    for i in range(len(cles)):
        for j in range(i + 1, len(cles)):
            k1, k2 = cles[i], cles[j]
            G1 = DF.iloc[P0[k1]]
            G2 = DF.iloc[P0[k2]]
            d = linkage.calcule(G1, G2)
            if d < dist_min:
                dist_min = d
                cle1, cle2 = k1, k2

    nouvelle_cle = max(P0.keys()) + 1
    P1 = {k: v for k, v in P0.items() if k not in (cle1, cle2)}
    P1[nouvelle_cle] = P0[cle1] + P0[cle2]

    if verbose:
        print(f"CHA_fusionne: distance mininimale trouvée entre [{cle1}, {cle2}] = {dist_min:.4f}")
        print(f"CHA_fusionne: les 2 clusters dont les clés sont  [{cle1}, {cle2}]  sont fusionnés")
        print(f"CHA_fusionne: on crée la  nouvelle clé {nouvelle_cle}  dans le dictionnaire.")
        print(f"CHA_fusionne: les clés de  [{cle1}, {cle2}]  sont supprimées car leurs clusters ont été fusionnés.")

    return P1, cle1, cle2, dist_min


def CHA_algorithme(DF, linkage, verbose=False):
    P = CHA_initialise(DF)
    resultat = []
    while len(P) > 1:
        P, k1, k2, d = CHA_fusionne(DF, P, linkage, verbose=verbose)
        nouvelle_cle = max(P.keys())
        taille = len(P[nouvelle_cle])
        resultat.append([k1, k2, d, taille])
    return resultat


def CHA_dendrogramme(liste_resultat, info_linkage):
    plt.figure(figsize=(30, 15))
    plt.title(f'Dendrogramme - {info_linkage}', fontsize=25)
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)
    scipy.cluster.hierarchy.dendrogram(liste_resultat, leaf_font_size=24.)
    plt.show()


# repris du TME-10 : algorithme des K-moyennes
class KMoyennes():
    """ Algorithme des K-moyennes (k-means). Base attendue : un pandas DataFrame """
    def __init__(self, K, distance=DistanceEuclidienne()):
        if K < 1:
            raise TypeError("KMoyennes: K doit être strictement plus grand que 0 !")
        self.__K = K
        self.__distance = distance

    def get_K(self):
        return self.__K

    def __str__(self) -> str:
        return f"KMoyennes (K={self.__K}, {self.__distance})"

    def inertie_cluster(self, Ens):
        # somme des distances au carre au centre du cluster
        Ens_arr = np.array(Ens)
        centre = Ens_arr.mean(axis=0)
        return float(np.sum(np.sum((Ens_arr - centre) ** 2, axis=1)))

    def init(self, Ens):
        # K centres tires au hasard parmi les exemples
        n = len(Ens)
        indices = np.random.choice(n, self.__K, replace=False)
        return np.array(Ens)[indices]

    def plus_proche(self, exemple, Centres):
        ex = np.array(exemple)
        distances = self.__distance.calcule(ex, np.array(Centres))
        return int(np.argmin(distances))

    def affecte_cluster(self, Base, Centres):
        # vectorise : matrice des distances (n points x K centres) puis argmin sur les centres
        Base_arr = np.array(Base)
        Centres = np.array(Centres)
        # une colonne de distances par centre (calcule gere un vecteur contre toute la matrice)
        D = np.column_stack([self.__distance.calcule(Centres[j], Base_arr) for j in range(self.__K)])
        affect = np.argmin(D, axis=1)
        U = {j: [] for j in range(self.__K)}
        for i in range(len(Base_arr)):
            U[int(affect[i])].append(i)
        return U

    def centroides(self, Base, U):
        Base_arr = np.array(Base)
        centres = []
        for j in sorted(U.keys()):
            centres.append(Base_arr[U[j]].mean(axis=0))
        return np.array(centres)

    def inertie_globale(self, Base, U):
        Base_arr = np.array(Base)
        total = 0.0
        for j in U:
            if len(U[j]) >= 1:
                total += self.inertie_cluster(Base_arr[U[j]])
        return total

    def train(self, Base, epsilon, iter_max, verbose=False):
        Centres = self.init(Base)
        U = self.affecte_cluster(Base, Centres)
        inertie = self.inertie_globale(Base, U)
        if verbose:
            print(f"\nEtapes de l'apprentissage (epsilon = {epsilon:.3f} et iter_max = {iter_max}) :")
            print(f"\titeration n°1 : Inertie = {inertie:.4f}\tDifference = --")
        for it in range(2, iter_max + 1):
            Centres = self.centroides(Base, U)
            U = self.affecte_cluster(Base, Centres)
            nouvelle_inertie = self.inertie_globale(Base, U)
            diff = abs(inertie - nouvelle_inertie)
            if verbose:
                print(f"\titeration n°{it} : Inertie = {nouvelle_inertie:.4f}\tDifference = {diff:.4f}")
            inertie = nouvelle_inertie
            if diff < epsilon:
                break
        return Centres, U


def affiche_resultat(Base, Centres, Affect):
    # affiche les clusters (2D) et les centres en rouge
    couleurs = cm.tab20(np.linspace(0, 1, 20))
    Base_arr = np.array(Base)
    for j in Affect:
        points = Base_arr[Affect[j]]
        plt.scatter(points[:, 0], points[:, 1], color=couleurs[j % 20])
    plt.scatter(Centres[:, 0], Centres[:, 1], color='red', marker='x')
    plt.show()


def index_Dunn(Base, Centres, U, distance=DistanceEuclidienne()):
    # rapport (plus petite distance entre centres) / (plus grand diametre de cluster)
    Base_arr = np.array(Base)
    K = len(Centres)
    diametres = []
    for j in U:
        points = Base_arr[U[j]]
        d_max = 0.0
        for a in range(len(points)):
            for b in range(a+1, len(points)):
                d = distance.calcule(points[a], points[b])
                if d > d_max:
                    d_max = d
        diametres.append(d_max)
    diam_max = max(diametres) if diametres else 0.0
    d_min = float('inf')
    for j in range(K):
        for k in range(j+1, K):
            d = distance.calcule(Centres[j], Centres[k])
            if d < d_min:
                d_min = d
    return d_min / diam_max if diam_max > 0 else 0.0


def index_XieBeni(Base, Centres, U, distance=DistanceEuclidienne()):
    # inertie totale / (n * plus petite distance au carre entre centres)
    Base_arr = np.array(Base)
    n = len(Base_arr)
    K = len(Centres)
    inertie = 0.0
    for j in U:
        for i in U[j]:
            inertie += distance.calcule(Base_arr[i], Centres[j]) ** 2
    d_min = float('inf')
    for j in range(K):
        for k in range(j+1, K):
            d = distance.calcule(Centres[j], Centres[k]) ** 2
            if d < d_min:
                d_min = d
    return inertie / (n * d_min) if d_min > 0 else float('inf')
