import logging
import warnings
from collections import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh as lapack_eigh
from scipy.sparse.linalg import eigsh as arpack_eigh
import sklearn.manifold as skl_manifold

import openTSNE

import openTSNE.initialization

import Orange
from Orange.data import Table, Domain, ContinuousVariable
from Orange.distance import Distance, DistanceModel, Euclidean
from Orange.projection import SklProjector, Projector, Projection
from Orange.projection.base import TransformDomain, ComputeValueProjector

# multi-ann files
#import openTSNE.affinity
from Orange.widgets.bakk.additional_files import affinity

from Orange.projection.manifold import torgerson, MDS, Isomap, LocallyLinearEmbedding, \
                                        SpectralEmbedding, TSNEModel, TSNE


__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding", "SpectralEmbedding",
           "TSNE"]

# Disable t-SNE user warnings
openTSNE.tsne.log.setLevel(logging.ERROR)
openTSNE.affinity.log.setLevel(logging.ERROR)

class MultiANNTSNE(TSNE):

    # def fit(self, X: np.ndarray, Y: np.ndarray = None) -> openTSNE.TSNEEmbedding:
    #     # Sparse data are not supported
    #     if sp.issparse(X):
    #         raise TypeError(
    #             "A sparse matrix was passed, but dense data is required. Use "
    #             "X.toarray() to convert to a dense numpy array."
    #         )
    #
    #     # Build up the affinity matrix, using multiscale if needed
    #     if self.multiscale:
    #         # The local perplexity should be on the order ~50 while the higher
    #         # perplexity should be on the order ~N/50
    #         if not isinstance(self.perplexity, Iterable):
    #             raise ValueError(
    #                 "Perplexity should be an instance of `Iterable`, `%s` "
    #                 "given." % type(self.perplexity).__name__)
    #         affinities = affinity.MultiANNMultiscale(
    #             X, perplexities=self.perplexity, metric=self.metric,
    #             method=self.neighbors, random_state=self.random_state, n_jobs=self.n_jobs)
    #     else:
    #         if isinstance(self.perplexity, Iterable):
    #             raise ValueError(
    #                 "Perplexity should be an instance of `float`, `%s` "
    #                 "given." % type(self.perplexity).__name__)
    #         affinities = affinity.MutliANNPerplexityBasedNN(
    #             X, perplexity=self.perplexity, metric=self.metric,
    #             method=self.neighbors, random_state=self.random_state, n_jobs=self.n_jobs)
    #
    #     # Create an initial embedding
    #     if isinstance(self.initialization, np.ndarray):
    #         initialization = self.initialization
    #     elif self.initialization == "pca":
    #         initialization = openTSNE.initialization.pca(
    #             X, self.n_components, random_state=self.random_state)
    #     elif self.initialization == "random":
    #         initialization = openTSNE.initialization.random(
    #             X, self.n_components, random_state=self.random_state)
    #     else:
    #         raise ValueError(
    #             "Invalid initialization `%s`. Please use either `pca` or "
    #             "`random` or provide a numpy array." % self.initialization)
    #
    #     embedding = openTSNE.TSNEEmbedding(
    #         initialization, affinities, learning_rate=self.learning_rate,
    #         theta=self.theta, min_num_intervals=self.min_num_intervals,
    #         ints_in_interval=self.ints_in_interval, n_jobs=self.n_jobs,
    #         negative_gradient_method=self.negative_gradient_method,
    #         callbacks=self.callbacks, callbacks_every_iters=self.callbacks_every_iters,
    #     )
    #
    #     # Run standard t-SNE optimization
    #     embedding.optimize(
    #         n_iter=self.early_exaggeration_iter, exaggeration=self.early_exaggeration,
    #         inplace=True, momentum=0.5, propagate_exception=True,
    #     )
    #     embedding.optimize(
    #         n_iter=self.n_iter, exaggeration=self.exaggeration,
    #         inplace=True, momentum=0.8, propagate_exception=True,
    #     )
    #
    #     return embedding

    def compute_affinities(self, X):
        # Sparse data are not supported
        if sp.issparse(X):
            raise TypeError(
                "A sparse matrix was passed, but dense data is required. Use "
                "X.toarray() to convert to a dense numpy array."
            )

        # Build up the affinity matrix, using multiscale if needed
        if self.multiscale:
            # The local perplexity should be on the order ~50 while the higher
            # perplexity should be on the order ~N/50
            if not isinstance(self.perplexity, Iterable):
                raise ValueError(
                    "Perplexity should be an instance of `Iterable`, `%s` "
                    "given." % type(self.perplexity).__name__
                )
            affinities = affinity.MultiANNMultiscale(
                X,
                perplexities=self.perplexity,
                metric=self.metric,
                method=self.neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            if isinstance(self.perplexity, Iterable):
                raise ValueError(
                    "Perplexity should be an instance of `float`, `%s` "
                    "given." % type(self.perplexity).__name__
                )
            affinities = affinity.MultiANNPerplexityBasedNN(
                X,
                perplexity=self.perplexity,
                metric=self.metric,
                method=self.neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        return affinities
