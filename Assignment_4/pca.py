import numpy as np

# dataset: dict {classkey: array[feature vectors - np.array(float)]}
def pca (dataset, newdim):
    # assumption: data normalized, n x d
    full_data = []
    keysize = {}
    keyorder = []
    for c in dataset:
        keyorder.append (c)
        keysize[c] = len (dataset[c])
        full_data.extend (dataset[c])

    full_data = np.array (full_data)
    evals, evecs = np.linalg.eig (full_data.T @ full_data)

    PairComplexCompareFunc = lambda p : - abs (p[0])
    evals, evecs = zip (*sorted (zip (evals, evecs.T), key=PairComplexCompareFunc))

    W = (np.asarray (evecs)).T[:, :newdim]
    full_data = full_data @ W

    newdata = {}
    last = 0
    for c in keyorder:
        newdata[c] = full_data[last:last+keysize[c]].astype (float)
        last += keysize[c]

    return newdata
    