import logging
import operator
from functools import reduce

import numpy as np
import scipy.sparse as sp

#from . import nearest_neighbors

log = logging.getLogger(__name__)

import openTSNE
from openTSNE import nearest_neighbors
from openTSNE.affinity import PerplexityBasedNN, joint_probabilities_nn, MultiscaleMixture, FixedSigmaNN

from Orange.widgets.bakk.additional_files.algorithms import annoy, nearpy, nmslib



def build_knn_index(
    data, method, k, metric, metric_params=None, n_jobs=1, random_state=None
):
    methods = {
        "NNDescent": nearest_neighbors.NNDescent,
        "BallTree": nearest_neighbors.BallTree,
        "Annoy": annoy.Annoy,
        "Hnsw": nmslib.Hnsw,
        "SW-Graph": nmslib.SWGraph,
        #"vp-tree": nmslib.VPTree,
        "NAPP": nmslib.NAPP,
        #"simple_invindx": nmslib.SimpleInvindx,
        "Brute Force": nmslib.BruteForce,
        "NearPy": nearpy.NearPy,
    }
    if isinstance(method, nearest_neighbors.KNNIndex):
        knn_index = method

    elif method not in methods:
        raise ValueError(
            "Unrecognized nearest neighbor algorithm `%s`. Please choose one "
            "of the supported methods or provide a valid `KNNIndex` instance."
            % method
        )
    else:
        knn_index = methods[method](
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    neighbors, distances = knn_index.build(data, k=k)

    return knn_index, neighbors, distances


class MultiANNPerplexityBasedNN(PerplexityBasedNN):
    """Compute affinities using nearest neighbors.
    """

    #super init !
    def __init__(
        self,
        data,
        perplexity=30,
        method="approx",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = data.shape[0]
        self.perplexity = self.check_perplexity(perplexity)

        # self.knn_index = build_knn_index(
        #     data, method, metric, metric_params, n_jobs, random_state
        # )
        #
        # # Find and store the nearest neighbors so we can reuse them if the
        # # perplexity is ever lowered
        # k_neighbors = min(self.n_samples - 1, int(3 * self.perplexity))
        # self.__neighbors, self.__distances = self.knn_index.query_train(
        #     data, k=k_neighbors
        # )

        k_neighbors = min(self.n_samples - 1, int(3 * self.perplexity))
        self.knn_index, self.__neighbors, self.__distances = build_knn_index(
            data, method, k_neighbors, metric, metric_params, n_jobs, random_state
        )

        self.P = joint_probabilities_nn(
            self.__neighbors,
            self.__distances,
            [self.perplexity],
            symmetrize=symmetrize,
            n_jobs=n_jobs,
        )

        self.n_jobs = n_jobs

class MultiANNFixedSigmaNN(FixedSigmaNN):
    """Compute affinities using using nearest neighbors and a fixed bandwidth
    for the Gaussians in the ambient space.

    Using a fixed Gaussian bandwidth can enable us to find smaller clusters of
    data points than we might be able to using the automatically determined
    bandwidths using perplexity. Note however that this requires mostly trial
    and error.
    """

    def __init__(
        self,
        data,
        sigma,
        k=30,
        method="exact",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = n_samples = data.shape[0]

        if k >= self.n_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than N-1 (%d)." % (k, self.n_samples)
            )

        knn_index, neighbors, distances = build_knn_index(
            data, method, k, metric, metric_params, n_jobs, random_state
        )

        self.knn_index = knn_index

        # Compute asymmetric pairwise input similarities
        conditional_P = np.exp(-distances ** 2 / (2 * sigma ** 2))
        conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

        P = sp.csr_matrix(
            (conditional_P.ravel(), neighbors.ravel(), range(0, n_samples * k + 1, k)),
            shape=(n_samples, n_samples),
        )

        # Symmetrize the probability matrix
        if symmetrize:
            P = (P + P.T) / 2

        # Convert weights to probabilities
        P /= np.sum(P)

        self.sigma = sigma
        self.k = k
        self.P = P
        self.n_jobs = n_jobs


class MultiANNMultiscaleMixture(MultiscaleMixture):
    """Calculate affinities using a Gaussian mixture kernel.

    Instead of using a single perplexity to compute the affinities between data
    points, we can use a multiscale Gaussian kernel instead. This allows us to
    incorporate long range interactions.
    """

    def __init__(
        self,
        data,
        perplexities,
        method="exact",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = data.shape[0]

        # We will compute the nearest neighbors to the max value of perplexity,
        # smaller values can just use indexing to truncate unneeded neighbors
        perplexities = self.check_perplexities(perplexities)
        max_perplexity = np.max(perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        self.knn_index, self.__neighbors, self.__distances = build_knn_index(
            data, method, k_neighbors, metric, metric_params, n_jobs, random_state
        )

        self.P = self._calculate_P(
            self.__neighbors,
            self.__distances,
            perplexities,
            symmetrize=symmetrize,
            n_jobs=n_jobs,
        )

        self.perplexities = perplexities
        self.n_jobs = n_jobs

# CS: Has to be redefined, because the base-class wouldn't include Multi-ANN
# if one would use it simply from openTSNE.affinity.Multiscale
class MultiANNMultiscale(MultiANNMultiscaleMixture):
    """Calculate affinities using averaged Gaussian perplexities.

    In contrast to :class:`MultiscaleMixture`, which uses a Gaussian mixture
    kernel, here, we first compute single scale Gaussian kernels, convert them
    to probability distributions, then average them out between scales.

    Please see the :ref:`parameter-guide` for more information.
    """

    @staticmethod
    def _calculate_P(
        neighbors,
        distances,
        perplexities,
        symmetrize=True,
        normalization="pair-wise",
        n_reference_samples=None,
        n_jobs=1,
    ):
        # Compute normalized probabilities for each perplexity
        partial_Ps = [
            joint_probabilities_nn(
                neighbors,
                distances,
                [perplexity],
                symmetrize=symmetrize,
                normalization=normalization,
                n_reference_samples=n_reference_samples,
                n_jobs=n_jobs,
            )
            for perplexity in perplexities
        ]
        # Sum them together, then normalize
        P = reduce(operator.add, partial_Ps, 0)

        # Take care to properly normalize the affinity matrix
        if normalization == "pair-wise":
            P /= np.sum(P)
        elif normalization == "point-wise":
            P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

        return P
