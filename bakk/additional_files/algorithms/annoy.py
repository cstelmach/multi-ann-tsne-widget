from openTSNE.nearest_neighbors import KNNIndex
import annoy
import numpy as np

class Annoy(KNNIndex):
    #VALID_METRICS = neighbors.Annoy.valid_metrics

    def build(self, data):
        n_items, vector_length = data.shape
        #initalize parameters
        method_param = init_method_param("annoy", data)
        ntrees = method_param["ntrees"]
        #build index
        self.index = annoy.AnnoyIndex(vector_length, metric=self.metric)
        for i in range(n_items):
            self.index.add_item(i, data[i])
        self.index.build(ntrees)

    def query_train(self, data, k):
        #add search_k parameter: tradeoff between speed and accuracy?
        #neighbors_single, distances_single = np.asarray(self.index.get_nns_by_vector(data[i], n=k, search_k=-1, include_distances=True))
        #output array with points x neighbors:
        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))
        for i in range(len(data)):
            neighbors_single, distances_single = np.asarray(self.index.get_nns_by_item(i, n=k, search_k=-1 ,include_distances=True))
            neighbors[i] = neighbors_single
            distances[i] = distances_single
        print("neighbors.shape: {}".format(neighbors.shape))
        print("neighbors[0]: {}".format(neighbors[0]))
        print(neighbors.shape)
        print("distances.shape: {}".format(distances.shape))
        print("distances[0]: {}".format(distances[0]))
        return neighbors, distances

    def query(self, query, k):
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))
        for i in range(len(query)):
            neighbors_single, distances_single = np.asarray(self.index.get_nns_by_vector(query[i], n=k, search_k=-1, include_distances=True))
            neighbors[i] = neighbors_single
            distances[i] = distances_single
        return neighbors, distances
