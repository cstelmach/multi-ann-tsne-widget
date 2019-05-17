import numpy as np
from annoy import AnnoyIndex

class Annoy(object):

    # initialozation of index
    def __init__(
        self,
        data,
        metric="euclidean",
        #metric_kwds=None,
        n_neighbors=15, #standard value?
        n_trees=10, #standard value?
        #random_state=np.random,
        n_iters=10,
    ):

        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.metric = metric

        # if you do not provide Additional metric arguments, provide empty dictionary
        if metric_kwds is None:
            metric_kwds = dict()
        self.metric_kwds = metric_kwds

        self.n_iters = n_iters

        # ???
        self.dim = data.shape[1]

        # data saved as type float32
        data = check_array(data).astype(np.float32)


        # save metric args as tuple
        self._dist_args = tuple(metric_kwds.values())

        # ???
        self.random_state = check_random_state(random_state)

        # copy data as raw data
        self._raw_data = data.copy()

        # define what to do depending on the metric
        if callable(metric):
            self._distance_func = metric
        elif metric in dist.named_distances:
            self._distance_func = dist.named_distances[metric]

        if metric in ("cosine"):
            self._angular_trees = True
        else:
            self._angular_trees = False

        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )

        indices = np.arange(data.shape[0])

        n_vertices = data.shape[0]
        annoy = AnnoyIndex(n_vertices,
                       metric = self.metric
                       )  # Length of item vector that will be indexed
        for i in range(n_iters):
            v = [np.random.normal(0, 1) for z in range(data)]
            annoy.add_item(i, v)

        annoy.build(n_trees) # 10 trees
        annoy.save('test.ann')

        #u = AnnoyIndex(n_vertices)
        #u.load('test.ann') # super fast, will just mmap the file
        indices, distances = annoy.get_nns_by_item(0, n_neighbors, include_distances=True) # will find the 1000 nearest neighbors

        self._neighbor_graph = indices, distances


        #self._search_graph = lil_matrix(
        #    (data.shape[0], data.shape[0]), dtype=np.float32
        #)
        #self._search_graph.rows = self._neighbor_graph[0]
        #self._search_graph.data = self._neighbor_graph[1]
        #self._search_graph = self._search_graph.maximum(
        #    self._search_graph.transpose()
        #).tocsr()
        #self._search_graph = prune(
        #    self._search_graph,
        #    prune_level=self.prune_level,
        #    n_neighbors=self.n_neighbors,
        #)
        #self._search_graph = (self._search_graph != 0).astype(np.int8)

        #self._random_init, self._tree_init = make_initialisations(
        #    self._distance_func, self._dist_args
        #)

        #self._search = make_initialized_nnd_search(self._distance_func, self._dist_args)

        return

    def query(self, query_data, k=10, search_k=-1):
        """
        Query the training data for the k nearest neighbors
        """

        n_vertices = query_data.shape[0]
        annoy = AnnoyIndex(n_vertices)
        annoy.load("test.ann")
        indices, distances = annoy.get_nns_by_item(i, n, search_k, include_distances=True)
        """
        returns the n closest items.
        During the query it will inspect up to search_k nodes which defaults to
        n_trees * n if not provided.
        search_k gives you a run-time tradeoff between better accuracy and speed.
        If you set include_distances to True, it will return a 2 element tuple
        with two lists in it: the second one containing all corresponding
        distances.
        """

        return indices[:, :k], distances[:, :k]
