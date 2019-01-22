
from external.Robust_PCA.r_pca import R_pca


def RPCA(data):
    # use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
    rpca = R_pca(data)
    L, S = rpca.fit(max_iter=10000, iter_print=100)

    return L, S
















