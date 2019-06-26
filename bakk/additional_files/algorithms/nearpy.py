from openTSNE.nearest_neighbors import KNNIndex
import nearpy
import numpy as np
from nearpy.filters import NearestFilter
import sklearn.preprocessing

from Orange.widgets.bakk.additional_files.parameters import init_method_param

class NearPy(KNNIndex):
    #VALID_METRICS = neighbors.NearPy.valid_metrics
    #METHOD_PARAMS = neighbors.NearPy.params (text/config file)

    def build(self, data, k):
        n_items, vector_length = data.shape
        print(data.shape)
        #parameters init
        method_param = init_method_param("nearpy")
        hash_counts = method_param["hash_counts"]
        n_bits = method_param["n_bits"]

        self.filter = NearestFilter(10)

        hashes = []
        for k in range(hash_counts):
            nearpy_rbp = nearpy.hashes.RandomBinaryProjections(
                'rbp_%d' % k, n_bits)
            hashes.append(nearpy_rbp)

        if self.metric == 'euclidean':
            dist = nearpy.distances.EuclideanDistance()
            self.index = nearpy.Engine(
                vector_length,
                lshashes=hashes,
                distance=dist,
                vector_filters=[self.filter])
        else:  # Default (angular) = Cosine distance
            self.index = nearpy.Engine(
                vector_length,
                lshashes=hashes,
                vector_filters=[self.filter])

        #if self.metric == 'angular':
            #data = sklearn.preprocessing.normalize(data, axis=1, norm='l2')
        for i, x in enumerate(data):
            self.index.store_vector(x, i)

        # def query_train(self, data, k):
        self.filter.N = k
        #if self.metric == 'angular':
            #data = sklearn.preprocessing.normalize([data], axis=1, norm='l2')[0]

        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))

        for i in range(len(data)):
            item_single = self.index.neighbours(data[i])
            dp_n = []
            dp_d = []
            for j in range(len(item_single)):
                dp_n.append(item_single[j][1])
                dp_d.append(item_single[j][2])
            neighbors[i] = np.asarray(dp_n)
            distances[i] = np.asarray(dp_d)

        return neighbors, distances

    def query(self, query, k):
        self.filter.N = k
        #if self.metric == 'angular':
        #    query = sklearn.preprocessing.normalize([query], axis=1, norm='l2')[0]
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))

        for i in range(len(query)):
            item_single = self.index.neighbours(data[i])
            dp_n = []
            dp_d = []
            for j in range(len(item_single)):
                dp_n.append(item_single[j][1])
                dp_d.append(item_single[j][2])
            neighbors[i] = np.asarray(dp_n)
            distances[i] = np.asarray(dp_d)

        return neighbors, distances
