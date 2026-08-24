# Fashion-MNIST from scratch

Classification and clustering on Fashion-MNIST, using a small machine learning library
written from scratch (no scikit-learn), to check whether classification errors come from
the algorithms or from the data itself.

## Motivation

Fashion-MNIST has 10 clothing classes as 28x28 grayscale images. Some classes are easy to
tell apart (trousers vs. bags), others look almost identical at that resolution, mainly
the four "tops" (T-shirt, pullover, coat, shirt). When a classifier struggles on those, is
that a limitation of the algorithm, or do the classes genuinely overlap in the data? A
single classifier can't answer that on its own, so this project compares several
classifiers and representations, then checks the conclusion against clustering, which
never sees the labels at all.

## Approach

Everything (kNN, perceptron, decision tree, k-means, hierarchical clustering, PCA) is
implemented by hand in `iads/`, a small library built during the course this project comes
from. No scikit-learn.

- Balanced subsamples with a fixed random seed, for reproducibility.
- Stratified k-fold cross-validation, with preprocessing (normalization, PCA) refit inside
  each fold to avoid data leakage.
- Three representations compared: raw pixels, min-max normalized, and PCA (50 components,
  ~89% of variance).
- The decision tree, implemented "by hand" with no shortcuts, doesn't scale well past a few
  hundred dimensions, so it's only tested on raw pixels (784 dimensions) on a smaller sample,
  as a reference point rather than a fair comparison.
- For the unsupervised part, k-means and hierarchical clustering are run without the
  labels, and only compared against the true classes afterwards, as an independent check on
  what the supervised models found.

## Results

**Binary classification** (best classifier: kNN)

| Pair | Accuracy |
|---|---|
| Trousers / Bag (easy) | 0.996 |
| Pullover / Coat (hard) | 0.847 |

Two very different silhouettes separate almost perfectly (8 errors out of 2000). Two
similar-looking tops plateau around 85%: the confusion is already visible in the raw
images, not just in the classifier's mistakes.

**Multi-class classification** (10 classes, one-vs-all)

| Classifier | Cross-val accuracy |
|---|---|
| kNN (k=9) | 0.814 |
| Perceptron | 0.793 |
| Decision tree | 0.698 |

kNN wins, reaching 0.821 on the full test set. Errors concentrate on the "tops": recall for
the "shirt" class is only 0.53, mostly confused with T-shirt, pullover and coat.

**Unsupervised: k-means**

Without ever seeing the labels, the clustering quality indices (Dunn, Xie-Beni) prefer
k=5 over k=10 (Dunn 0.344 vs. 0.296, Xie-Beni 0.79 vs. 1.05). At k=5, the four "tops"
collapse into a single cluster, the same confusion found in supervised classification,
recovered without any labels.

**Unsupervised: hierarchical clustering**

The dendrogram cleanly separates shoes from clothes, independently confirming the same
split found by k-means.

**PCA projection (2D, 47% of variance)**

Shoes, bags and trousers form clearly separated groups; the four tops overlap into a
single cloud of points.

Three independent methods (supervised classification, k-means, hierarchical clustering)
agree: the errors come from real visual similarity between certain clothing categories, not
from weaknesses in the algorithms.

## Project structure

```
.
├── iads/                          # from-scratch ML library
│   ├── Classifiers.py             # kNN, perceptron, decision tree, one-vs-all wrapper
│   ├── Clustering.py              # k-means, hierarchical clustering, quality indices
│   ├── evaluation.py              # cross-validation, confusion matrix, metrics
│   └── utils.py                   # dataset generation, PCA, plotting
├── Projet/
│   └── notebook.ipynb          # main notebook, all experiments
├── poster.pdf                     # one-page summary of the results
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt
```

The notebook expects Fashion-MNIST as two CSV files (785 columns: one label + 784 pixel
values), placed in a `data/` folder at the root of the repo:

```
data/
├── fashion-mnist_train.csv
└── fashion-mnist_test.csv
```

They can be downloaded from the [`zalando-research/fashionmnist`](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
dataset on Kaggle.

Then open `Projet/notebook.ipynb` with Jupyter and run all cells. The notebook
needs to be launched from inside `Projet/`, since it reaches the library with
`sys.path.append('../')`.
