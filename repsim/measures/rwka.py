from typing import Union

import numpy.typing as npt
import torch
from repsim.measures.utils import flatten
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.measures.utils import SHAPE_TYPE
from repsim.measures.utils import to_torch_if_needed

# --------------------------------- --------------------------------- #
# Random walk kernel alignment (RWKA)
# here we implment our method, which is a kernel alignment method based on random walk kernels

def random_walk_kernel_alignment(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Random walk kernel alignment (RWKA)"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_torch_if_needed(R, Rp)

    # compute distance matrices
    dist_matR = torch.cdist(R, R)
    dist_matRp = torch.cdist(Rp, Rp)

    # compute median distance for each matrix
    # use median heuristic for bandwidth using lower triangular part (excluding diagonal)
    R_median_dist = get_median_distance(dist_matR)
    Rp_median_dist = get_median_distance(dist_matRp)

    sigma_vec = torch.linspace(0.1, 10.0, steps=30)
    similarity_vec = []
    for sigma in sigma_vec:
        # compute the random walk kernels
        K = torch.exp(-dist_matR / ((R_median_dist * sigma) ** 2))
        Kp = torch.exp(-dist_matRp / ((Rp_median_dist * sigma) ** 2))

        # add to the similarity vector
        similarity_vec.append(rw_similarity(K, Kp, unbiased=False))

    # aggregate the similarity scores across different sigma values by the sum of all values 
    # which corresponds to the area under the curve (AUC) of the similarity scores across different sigma values
    return sum(similarity_vec) / len(similarity_vec)


def get_median_distance(dist_mat):
    """
    Compute the median distance from the distance matrix, 
    using the lower triangular part (excluding diagonal)
    """
    tril_indices = torch.tril_indices(dist_mat.shape[0], dist_mat.shape[1], offset=-1)
    return torch.median(dist_mat[tril_indices[0], tril_indices[1]]).item()


def rw_similarity(K, Kp, unbiased=False) -> float:
    """
    Computes the similarity between two kernels K and L for random-walk based methods.
    Used for random walk kernel alignment (RWKA) similarity.
    Args:
        K: Kernel matrix of shape (N, N)
        L: Kernel matrix of shape (N, N)
        unbiased: Whether to use the unbiased estimator
    """
    if unbiased:            
        K_hat = K.clone().fill_diagonal_(0.0)
        Kp_hat = Kp.clone().fill_diagonal_(0.0)
    else:
        K_hat, Kp_hat = K, Kp
    
    K_norm = normalize_random_walk_kernel(K_hat)
    Kp_norm = normalize_random_walk_kernel(Kp_hat)

    sim_kk = torch.trace(K_norm @ K_norm)
    sim_kpkp = torch.trace(Kp_norm @ Kp_norm)
    sim_kkp = torch.trace(K_norm @ Kp_norm)

    return (sim_kkp / torch.sqrt(sim_kk * sim_kpkp)).item()


def normalize_random_walk_kernel(K):
    """ Compute the normalized kernel for random walk based methods """
    device = K.device
    col_sum = torch.sum(K, dim=0)
    col_sum[col_sum == 0] = 1.0  # avoid division by zero
    D_K_mhalf  = torch.diag(col_sum ** (-0.5)).to(device)
    return D_K_mhalf @ K @ D_K_mhalf



class RWKA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=random_walk_kernel_alignment,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )
