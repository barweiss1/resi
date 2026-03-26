from typing import Union

import numpy.typing as npt
import torch
from repsim.measures.utils import flatten
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.measures.utils import SHAPE_TYPE
from repsim.measures.utils import to_torch_if_needed
from repsim.measures.cka_rbf import get_median_distance
from repsim.measures.rwka import rw_similarity
from repsim.measures.cka_rbf import get_median_distance

# --------------------------------- --------------------------------- #
# Random walk kernel alignment (RWKA)
# here we implment our method, which is a kernel alignment method based on random walk kernels

def random_walk_kernel_alignment_auc(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Random walk kernel alignment (RWKA)"""
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
        similarity_vec.append(rw_similarity(K, Kp, unbiased=False))

    # aggregate the similarity scores across different sigma values by the sum of all values 
    # which corresponds to the area under the curve (AUC) of the similarity scores across different sigma values
    return sum(similarity_vec) / len(similarity_vec)


class RWKA_AUC(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=random_walk_kernel_alignment_auc,
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
