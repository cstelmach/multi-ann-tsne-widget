import math



def init_method_param(method, data=None, k=None):
    """
    Get method parameters based on approximate nearest neighbor algorithm and data (shape, vector)

    Input: name of method as string, data as np.array ([num_items][vector_length])
    Output: method_param as directory. Keys corresponding to the method-specific parameters
    """
    config_file = ""
    config_file = None

    if method == "annoy":

        if not config_file:
            n_items, vector_length = data.shape
            mp = {}

            # number of trees, as int
            mp["ntrees"] = 5 + int(round((n_items) ** 0.5 / 20))

            mp["search_k"] = -1

    elif method == "nearpy":
        if not config_file:
            mp = {}

            # number of bits, as int
            mp['n_bits'] = 20 # ????? How many as standard?

            # hash counts, as int
            mp['hash_counts'] = 20 # ????? How many as standard?

    elif method == "hnsw":

        if not config_file:
            mp = {}

            M = 15
            efC = 100
            post = 0 #postprocessing: 0 = None, 1 = little, 2 = much
            mp["index_param"] = {'M': M, 'efConstruction': efC, 'post' : post}
            efS = 100
            mp['query_param'] = {'efSearch': efS}

    elif method == "sw-graph":

        if not config_file:
            mp = {}

            NN = 15
            efC = 100
            mp["index_param"] = {'NN': NN, 'efConstruction': efC}
            efS = 100
            mp['query_param'] = {'efSearch': efS}

    elif method == "vp-tree":

        if not config_file:
            n_items, dim = data.shape
            mp = {}

            #M = 15
            #efC = 100
            desiredRecall = 15
            bucketSize = min(int(dim * 0.0005), 1000)

            tuneK = k # or tuneR = __
            # Optional Parameters
            # tuneQty =
            # minExp =
            # maxExp =
            mp["index_param"] = {'bucketSize': bucketSize,
            'desiredRecall': desiredRecall, 'tuneK': tuneK}
            mp['query_param'] = None

    elif method == "napp":

        if not config_file:
            n_items, vector_length = data.shape

            mp = {}

            numPivot = math.sqrt(n_items)
            numPivotIndex = math.sqrt(n_items)
            # Optional Parameters
            # hashTrickDim = 50000 # long tail high dimensional data
            # chunkIndexSize
            mp["index_param"] = {'numPivot':numPivot,
            'numPivotIndex':numPivotIndex}
            mp['query_param'] = None

    else:
        print("Error: ANN not found in method parameter file")
        return

    return mp

def init_method_param(method, data=None, k=None):
    """
    Get method parameters based on approximate nearest neighbor algorithm and data (shape, vector)

    Input: name of method as string, data as np.array ([num_items][vector_length])
    Output: method_param as directory. Keys corresponding to the method-specific parameters
    """
    config_file = ""
    config_file = None

    if method == "annoy":

        if not config_file:
            n_items, vector_length = data.shape
            mp = {}

            # number of trees, as int
            mp["ntrees"] = 5 + int(round((n_items) ** 0.5 / 20))

            mp["search_k"] = -1

    elif method == "nearpy":
        if not config_file:
            mp = {}

            # number of bits, as int
            mp['n_bits'] = 20 # ????? How many as standard?

            # hash counts, as int
            mp['hash_counts'] = 20 # ????? How many as standard?

    elif method == "hnsw":

        if not config_file:
            mp = {}

            M = 15
            efC = 100
            post = 0 #postprocessing: 0 = None, 1 = little, 2 = much
            mp["index_param"] = {'M': M, 'efConstruction': efC, 'post' : post}
            efS = 100
            mp['query_param'] = {'efSearch': efS}

    elif method == "sw-graph":

        if not config_file:
            mp = {}

            NN = 15
            efC = 100
            mp["index_param"] = {'NN': NN, 'efConstruction': efC}
            efS = 100
            mp['query_param'] = {'efSearch': efS}

    elif method == "vp-tree":

        if not config_file:
            n_items, dim = data.shape
            mp = {}

            #M = 15
            #efC = 100
            desiredRecall = 15
            bucketSize = min(int(dim * 0.0005), 1000)

            tuneK = k # or tuneR = __
            # Optional Parameters
            # tuneQty =
            # minExp =
            # maxExp =
            mp["index_param"] = {'bucketSize': bucketSize,
            'desiredRecall': desiredRecall, 'tuneK': tuneK}
            mp['query_param'] = None

    elif method == "napp":

        if not config_file:
            n_items, vector_length = data.shape

            mp = {}

            numPivot = math.sqrt(n_items)
            numPivotIndex = math.sqrt(n_items)
            # Optional Parameters
            # hashTrickDim = 50000 # long tail high dimensional data
            # chunkIndexSize
            mp["index_param"] = {'numPivot':numPivot,
            'numPivotIndex':numPivotIndex}
            mp['query_param'] = None

    else:
        print("Error: ANN not found in method parameter file")
        return

    return mp

    # elif method == "onng":
    #     """
    #     https://github.com/yahoojapan/NGT/blob/master/python/README-ngtpy.md
    #
    #     object_type: Specifies the data object type.
    #         c: 1 byte unsigned integer
    #         f: 4 byte floating point number (default)
    #
    #
    #     """
    #
    #     if not config_file:
    #         mp = {}
    #         mp["object_type"] = "f"
    #
    #         mp["edge_size_for_search"] = -2
    #
    #         mp["build_time_limit"] = 4
    #
    #         mp["epsilon"] = 10
    #
    #         mp["edge"] = 100
    #
    #         mp["outdegree"] = 10
    #
    #         mp["indegree"] = 120
