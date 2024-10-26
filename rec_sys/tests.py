import numpy as np
from scipy.sparse import coo_array

import rec_sys.cf_algorithms_to_complete as cfa


def test_centered_cosine_distance_neg_corr(k: int = 100, tol: float = 1e-6):
    x = np.array([i + 1 for i in range(k)])
    y = x[::-1]

    res = cfa.centered_cosine_sim(
        coo_array(x),
        coo_array(y),
    )

    expected_res = -1
    assert np.abs(res - expected_res) <= tol


def test_centered_cosine_distance_nan(k: int = 100, tol: float = 1e-6):
    x = np.array([i + 1 for i in range(k)], dtype=float)
    for c in [2, 3, 4, 5, 6]:
        for shift in range(0, 100, 10):
            # In scipy.sparse, the missing values have to be 0.
            # This does not affect the functionality of the code,
            # since the algorithm presented in the lecture
            # replaces the nan's with 0's after centering and
            # the sparse vectors only use the non-zero entries
            # for opperations.
            x[c + shift] = 0

    y = x[::-1]

    res = cfa.centered_cosine_sim(
        coo_array(x),
        coo_array(y),
    )

    expected_res = 0.0311755
    assert np.abs(res - expected_res) <= tol


if __name__ == "__main__":
    test_centered_cosine_distance_neg_corr()
    test_centered_cosine_distance_nan()