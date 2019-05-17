class Annoy(object):

    # initialozation of index
    def __init__(
        self,
        data,
        metric="euclidean",
        #metric_kwds=None,
        n_neighbors=15, #standard value?
        n_trees=8, #standard value?
        #leaf_size=15,
        #pruning_level=0,
        #tree_init=True,
        random_state=np.random,
        #algorithm="standard",
        #max_candidates=20,
        n_iters=10,
        #delta=0.001,
        #rho=0.5,
    ):

        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.metric = metric

        # if you do not provide Additional metric arguments, provide empty dictionary
        if metric_kwds is None:
            metric_kwds = dict()
        self.metric_kwds = metric_kwds

        #self.leaf_size = leaf_size
        #self.prune_level = pruning_level
        #self.max_candidates = max_candidates
        self.n_iters = n_iters
        #self.delta = delta
        #self.rho = rho

        # ???
        self.dim = data.shape[1]

        # data saved as type float32
        data = check_array(data).astype(np.float32)

        # see if there has been a tree initialization
        #if not tree_init or n_trees == 0:
        #    self.tree_init = False
        #else:
        #    self.tree_init = True

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

        #if self.tree_init:
        #    if self._angular_trees:
        #        self._rp_forest = [
        #            flatten_tree(
        #                make_angular_tree(
        #                    data, indices, self.rng_state, self.leaf_size
        #                ),
        #                self.leaf_size,
        #            )
        #            for i in range(n_trees)
        #        ]
        #    else:
        #        self._rp_forest = [
        #            flatten_tree(
        #                make_euclidean_tree(
        #                    data, indices, self.rng_state, self.leaf_size
        #                ),
        #                self.leaf_size,
        #            )
        #            for i in range(n_trees)
        #        ]
        #
        #    leaf_array = np.vstack([tree.indices for tree in self._rp_forest])
        #else:
        #    self._rp_forest = None
        #    leaf_array = np.array([[-1]])

        self._neighbor_graph = AnnoyIndex()
        #if algorithm == "standard" or leaf_array.shape[0] == 1:
        #    nn_descent = make_nn_descent(self._distance_func, self._dist_args)
        #    self._neighbor_graph = nn_descent(
        #        self._raw_data,
        #        self.n_neighbors,
        #        self.rng_state,
        #        self.max_candidates,
        #        self.n_iters,
        #        self.delta,
        #        self.rho,
        #        True,
        #        leaf_array,
        #    )
        #elif algorithm == "alternative":
        #    self._search = make_initialized_nnd_search(
        #        self._distance_func, self._dist_args
        #    )

        #    init_heaps = make_heap_initializer(self._distance_func, self._dist_args)
        #    graph_heap, search_heap = init_heaps(
        #        self._raw_data, self.n_neighbors, leaf_array
        #    )
        #    graph = lil_matrix((data.shape[0], data.shape[0]))
        #    graph.rows, graph.data = deheap_sort(graph_heap)
        #    graph = graph.maximum(graph.transpose())
        #    self._neighbor_graph = deheap_sort(
        #        self._search(
        #            self._raw_data,
        #            graph.indptr,
        #            graph.indices,
        #            search_heap,
        #            self._raw_data,
        #        )
        #    )
        #else:
        #    raise ValueError("Unknown algorithm selected")

        self._search_graph = lil_matrix(
            (data.shape[0], data.shape[0]), dtype=np.float32
        )
        self._search_graph.rows = self._neighbor_graph[0]
        self._search_graph.data = self._neighbor_graph[1]
        self._search_graph = self._search_graph.maximum(
            self._search_graph.transpose()
        ).tocsr()
        self._search_graph = prune(
            self._search_graph,
            prune_level=self.prune_level,
            n_neighbors=self.n_neighbors,
        )
        self._search_graph = (self._search_graph != 0).astype(np.int8)

        self._random_init, self._tree_init = make_initialisations(
            self._distance_func, self._dist_args
        )

        #self._search = make_initialized_nnd_search(self._distance_func, self._dist_args)

        return

    def query(self, query_data, k=10, queue_size=5.0):
        """Query the training data for the k nearest neighbors

        Parameters
        ----------
        query_data: array-like, last dimension self.dim
            An array of points to query

        k: integer (default = 10)
            The number of nearest neighbors to return

        queue_size: float (default 5.0)
            The multiplier of the internal search queue. This controls the
            speed/accuracy tradeoff. Low values will search faster but with
            more approximate results. High values will search more
            accurately, but will require more computation to do so. Values
            should generally be in the range 1.0 to 10.0.

        Returns
        -------
        indices, distances: array (n_query_points, k), array (n_query_points, k)
            The first array, ``indices``, provides the indices of the data
            points in the training set that are the nearest neighbors of
            each query point. Thus ``indices[i, j]`` is the index into the
            training data of the jth nearest neighbor of the ith query points.

            Similarly ``distances`` provides the distances to the neighbors
            of the query points such that ``distances[i, j]`` is the distance
            from the ith query point to its jth nearest neighbor in the
            training data.
        """
        # query_data = check_array(query_data, dtype=np.float64, order='C')
        query_data = np.asarray(query_data).astype(np.float32)
        init = initialise_search(
            self._rp_forest,
            self._raw_data,
            query_data,
            int(k * queue_size),
            self._random_init,
            self._tree_init,
            self.rng_state,
        )
        result = self._search(
            self._raw_data,
            self._search_graph.indptr,
            self._search_graph.indices,
            init,
            query_data,
        )

        indices, dists = deheap_sort(result)
        return indices[:, :k], dists[:, :k]




         n_neighbors = 400
         t = AnnoyIndex(n_neighbors,
                        metric = self.metric
                        )  # Length of item vector that will be indexed
         for i in range(n_iters):
         v = [random.gauss(0, 1) for z in range(data)]
         t.add_item(i, v)

         t.build(n_trees) # 10 trees
         t.save('test.ann')

         u = AnnoyIndex(f)
         u.load('test.ann') # super fast, will just mmap the file
         indices, distances = u.get_nns_by_item(0, 1000, include_distances=True)) # will find the 1000 nearest neighbors

         indices, distances = self.index._neighbor_graph
         return
