from typing import Union

import numpy.typing as npt
import torch
from repsim.measures.utils import flatten
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.measures.utils import SHAPE_TYPE
from repsim.measures.utils import to_torch_if_needed
from repsim.measures.cka import centered_kernel_alignment, hsic




def hsic_biased(K, Kp):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ Kp @ H)

def hsic_unbiased(K, Kp):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    Kp_tilde = Kp.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * Kp_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(Kp_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, Kp_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def cka_kernel(K, Kp, unbiased=False) -> float:
    """
    Computes the similarity between two kernels K and L for random-walk based methods.
    Used for random walk kernel alignment (RWKA) similarity.
    Args:
        K: Kernel matrix of shape (N, N)
        L: Kernel matrix of shape (N, N)
        unbiased: Whether to use the unbiased estimator
    """
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_kpkp = hsic_fn(Kp, Kp)
    hsic_kkp = hsic_fn(K, Kp)

    return (hsic_kkp / (torch.sqrt(hsic_kk * hsic_kpkp) + 1e-6)).item()


def get_median_distance(dist_mat):
    """
    Compute the median distance from the distance matrix, 
    using the lower triangular part (excluding diagonal)
    """
    tril_indices = torch.tril_indices(dist_mat.shape[0], dist_mat.shape[1], offset=-1)
    return torch.median(dist_mat[tril_indices[0], tril_indices[1]]).item()


def rbf_centered_kernel_alignment(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """RBF centered kernel alignment"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_torch_if_needed(R, Rp)

    # higher precision to avoid numerical issues with small sigma values
    R = R.double()
    Rp = Rp.double()

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
        similarity_vec.append(cka(K, Kp, unbiased=False))

    # aggregate the similarity scores across different sigma values by the sum of all values 
    # which corresponds to the area under the curve (AUC) of the similarity scores across different sigma values
    return sum(similarity_vec) / len(similarity_vec)



class CKA_RBF(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=rbf_centered_kernel_alignment,
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


