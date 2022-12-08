import numpy as np

# dataset: dict {classkey: array[feature vectors - np.array(float)]}
def lda (dataset, newdim):
    # assumption: data normalized, n x d
    means = {}
    covs = {}

    s_w_i = 0
    s_b_i = np.zeros ((len (dataset[list(dataset.keys())[0]][0]), len (dataset[list(dataset.keys())[0]][0])))
    mean = 0.
    for c in dataset:
        means[c] = np.mean (dataset[c], axis=0)
        covs[c] = np.zeros ((len (dataset[c][0]), len (dataset[c][0])))

        for x in dataset[c]:
            covs[c] += (x - means[c]).T @ ((x - means[c]))

        s_w_i += covs[c]
        mean += means[c]
    mean /= len (dataset)
    
    for c in means:
        s_b_i += (1 / len (dataset)) * (means[c] - mean).T @ ((means[c] - mean))
        
    evals, evecs = np.linalg.eig (np.linalg.pinv (s_w_i) @ s_b_i)

    PairComplexCompareFunc = lambda p : - abs (p[0])
    evals, evecs = zip (*sorted (zip (evals, evecs.T), key=PairComplexCompareFunc))

    W = (np.asarray (evecs)).T[:, :newdim].astype (float)

    newdata = {}
    for c in dataset:
        newdata[c] = dataset[c] @ W

    return newdata