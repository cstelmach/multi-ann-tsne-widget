import math

def init_method_param(method, data=None, k=None, cp=None):
    """
    Get method parameters based on approximate nearest neighbor algorithm and data (shape, vector)

    Input: name of method as string, data as np.array ([num_items][vector_length])
    Output: method_param as directory. Keys corresponding to the method-specific parameters
    """

    #if customp

    if method == "annoy":
        n_items, vector_length = data.shape
        mp = {}

        if cp['search_k']:
            mp["search_k"] = cp['search_k']
        else:
            mp["search_k"] = -1
        #print("n_trees:" + str(cp['n_trees']))
        #print(type(cp['n_trees']))
        if cp['n_trees']:
            mp["n_trees"] = int(cp['n_trees'])
        else:
            mp["n_trees"] = 5 + int(round((n_items) ** 0.5 / 20))

    elif method == "nearpy":
        n_items, vector_length = data.shape
        mp = {}
        if cp['n_bits']:
            mp['n_bits'] = cp['n_bits']
        else:
            mp['n_bits'] = min(vector_length, 20)
            #mp['n_bits'] = 20

        if cp['hash_counts']:
            mp['hash_counts'] = cp['hash_counts']
        else:
            mp['hash_counts'] = min(vector_length, 20)
            #mp['hash_counts'] = 20

    elif method == "hnsw":
        mp = {}
        if cp['M']:
            # Reasonable value between 5-100
            M = cp['M']
        else:
            M = int(k / 1.8)
            # find auto-values depending on n_items & dim

        if cp['hnsw_efConstruction']:
            # Reasonable value between 100-2000
            efC = cp['hnsw_efConstruction']
        else:
            efC = max(k * 0.8, 100)

        if cp['post']:
            #postprocessing:
            #0 = no, 1 = some, 2 = more
            post = cp['post']
        else:
            post = 0

        mp["index_param"] = {'M': M, 'efConstruction': efC, 'post' : post}

        if cp['hnsw_efSearch']:
            # Reasonable value between 100-2000
            efS = cp['hnsw_efSearch']
        else:
            efS = max(k * 0.8, 100)
        mp['query_param'] = {'efSearch': efS}

    elif method == "sw-graph":
        mp = {}
        if cp['NN']:
            # Reasonable value between 5-100
            NN = cp['NN']
        else:
            NN = int(k / 1.8)

        if cp['swg_efConstruction']:
            # Reasonable value between 100-2000
            efC = cp['swg_efConstruction']
        else:
            efC = max(k * 0.8, 100)

        mp["index_param"] = {'NN': NN, 'efConstruction': efC}

        if cp['swg_efSearch']:
            # Reasonable value between 100-2000
            efS = cp['swg_efSearch']
        else:
            efS = max(k * 0.8, 100)
        mp['query_param'] = {'efSearch': efS}

    elif method == "napp":
        n_items, vector_length = data.shape
        mp = {}

        if cp['numPivot']:
            numPivot = cp['numPivot']
        else:
            numPivot = int(math.sqrt(n_items))

        if cp['numPivotIndex']:
            numPivotIndex = cp['numPivotIndex']
        else:
            numPivotIndex = int(math.sqrt(n_items))

        if cp['hashTrickDim']:
            hashTrickDim = cp['hashTrickDim']
            mp["index_param"] = {'numPivot':numPivot,
            'numPivotIndex':numPivotIndex, 'hashTrickDim': hashTrickDim}
        else:
            hashTrickDim = None
            mp["index_param"] = {'numPivot':numPivot,
            'numPivotIndex':numPivotIndex}

            #mp['query_param'] = None

    else:
        print("Error: b  not found in method parameter file")
        return

    return mp
