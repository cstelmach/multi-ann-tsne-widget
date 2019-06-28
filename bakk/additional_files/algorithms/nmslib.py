from openTSNE.nearest_neighbors import KNNIndex

import nmslib
import os
import numpy as np

from Orange.widgets.bakk.additional_files.parameters import init_method_param

class Hnsw(KNNIndex):
    #VALID_METRICS = neighbors.Nmslib.valid_metrics

    def build(self, data, k):
        n_items, vector_length = data.shape
        self._method_name = "hnsw"
        method_param = init_method_param(self._method_name, data)
        self._index_param = method_param["index_param"]
        self._index_param["indexThreadQty"] = self.n_jobs
        self._query_param = method_param["query_param"]
        self._metric = {
        'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]

        self.index = nmslib.init(
            space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
        self.index.addDataPointBatch(data)
        self.index.createIndex(self._index_param)
        self.index.setQueryTimeParams(self._query_param)

        # def query_train(self, data, k):
        result = np.asarray(self.index.knnQueryBatch(data, k))
        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))
        for i in range(len(data)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

    def query(self, query, k):
        result = np.asarray(self.index.knnQueryBatch(query, k))
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))
        for i in range(len(query)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

class SWGraph(KNNIndex):
    #VALID_METRICS = neighbors.Nmslib.valid_metrics

    def build(self, data, k):
        n_items, vector_length = data.shape
        self._method_name = "sw-graph"
        method_param = init_method_param(self._method_name, data)
        self._index_param = method_param["index_param"]
        self._index_param["indexThreadQty"] = self.n_jobs
        self._query_param = method_param["query_param"]
        self._metric = {
        'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]

        self.index = nmslib.init(
            space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
        self.index.addDataPointBatch(data)
        self.index.createIndex(self._index_param)
        self.index.setQueryTimeParams(self._query_param)

        # def query_train(self, data, k):
        result = np.asarray(self.index.knnQueryBatch(data, k))
        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))
        for i in range(len(data)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

    def query(self, query, k):
        result = np.asarray(self.index.knnQueryBatch(query, k))
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))
        for i in range(len(query)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances


class NAPP(KNNIndex):
    #VALID_METRICS = neighbors.Nmslib.valid_metrics

    def build(self, data, k):
        n_items, vector_length = data.shape
        self._method_name = "napp"
        method_param = init_method_param(self._method_name, data)
        self._index_param = method_param["index_param"]
        self._index_param["indexThreadQty"] = self.n_jobs
        self._query_param = method_param["query_param"]
        self._metric = {
        'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]

        self.index = nmslib.init(
            space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
        self.index.addDataPointBatch(data)
        self.index.createIndex(self._index_param)
        self.index.setQueryTimeParams(self._query_param)

        # def query_train(self, data, k):
        result = np.asarray(self.index.knnQueryBatch(data, k))
        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))
        for i in range(len(data)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

    def query(self, query, k):
        result = np.asarray(self.index.knnQueryBatch(query, k))
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))
        for i in range(len(query)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

# class VPTree(KNNIndex):
#     #VALID_METRICS = neighbors.Nmslib.valid_metrics
#
#     def build(self, data, k):
#         n_items, vector_length = data.shape
#         self._method_name = "vp-tree"
#         method_param = init_method_param(self._method_name, data=data, k=k)
#         self._index_param = method_param["index_param"]
#         self._index_param["indexThreadQty"] = self.n_jobs
#         self._query_param = method_param["query_param"]
#         self._metric = {
#         'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]
#
#         self.index = nmslib.init(
#             space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
#         self.index.addDataPointBatch(data)
#         self.index.createIndex(self._index_param)
#         self.index.setQueryTimeParams(self._query_param)
#
#         # def query_train(self, data, k):
#         result = np.asarray(self.index.knnQueryBatch(data, k))
#         neighbors = np.empty((data.shape[0],k), dtype=int)
#         distances = np.empty((data.shape[0],k))
#         for i in range(len(data)):
#             neighbors[i] = result[i][0]
#             distances[i] = result[i][1]
#         return neighbors, distances
#
#     def query(self, query, k):
#         result = np.asarray(self.index.knnQueryBatch(query, k))
#         neighbors = np.empty((query.shape[0],k), dtype=int)
#         distances = np.empty((query.shape[0],k))
#         for i in range(len(query)):
#             neighbors[i] = result[i][0]
#             distances[i] = result[i][1]
#         return neighbors, distances

# class SimpleInvindx(KNNIndex):
#     #VALID_METRICS = neighbors.Nmslib.valid_metrics
#
#     def build(self, data, k):
#         n_items, vector_length = data.shape
#         self._method_name = "simple_invindx"
#         self._metric = {
#         'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]
#
#
#         # The method simple_invindx works only with the space negdotprod_sparse_fast
#         self.index = nmslib.init(
#             space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
#
#         self.index.addDataPointBatch(data)
#         self.index.createIndex()
#
#         #def query_train(self, data, k):
#         result = np.asarray(self.index.knnQueryBatch(data, k))
#         neighbors = np.empty((data.shape[0],k), dtype=int)
#         distances = np.empty((data.shape[0],k))
#         for i in range(len(data)):
#             neighbors[i] = result[i][0]
#             distances[i] = result[i][1]
#         return neighbors, distances
#
#     def query(self, query, k):
#         result = np.asarray(self.index.knnQueryBatch(query, k))
#         neighbors = np.empty((query.shape[0],k), dtype=int)
#         distances = np.empty((query.shape[0],k))
#         for i in range(len(query)):
#             neighbors[i] = result[i][0]
#             distances[i] = result[i][1]
#         return neighbors, distances

class BruteForce(KNNIndex):
    #VALID_METRICS = neighbors.Nmslib.valid_metrics

    def build(self, data, k):
        n_items, vector_length = data.shape
        self._method_name = "brute_force"
        self._metric = {
        'angular': 'cosinesimil', 'euclidean': 'l2'}[self.metric]

        self.index = nmslib.init(
            space=self._metric, method=self._method_name, data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)

        self.index.addDataPointBatch(data)
        self.index.createIndex()

        #def query_train(self, data, k):
        result = np.asarray(self.index.knnQueryBatch(data, k))
        neighbors = np.empty((data.shape[0],k), dtype=int)
        distances = np.empty((data.shape[0],k))
        for i in range(len(data)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances

    def query(self, query, k):
        result = np.asarray(self.index.knnQueryBatch(query, k))
        neighbors = np.empty((query.shape[0],k), dtype=int)
        distances = np.empty((query.shape[0],k))
        for i in range(len(query)):
            neighbors[i] = result[i][0]
            distances[i] = result[i][1]
        return neighbors, distances
