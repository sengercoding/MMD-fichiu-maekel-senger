# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
from scipy.sparse import (
    coo_array,
    csr_array,
)

def centered_cosine_sim(
    x: coo_array,
    y: coo_array,
    centered: bool = False,
) -> float:
    """Compute centered cosine similarity for sparse arrays."""
    if (
        x.shape[0] != y.shape[0]
        or x.ndim > 1
        or y.ndim > 1
    ):
        raise ValueError

    x_centered = x.tocsr()
    y_centered = y.tocsr()
        
    if not centered:
        # .mean() compute the mean over all elements.
        mean_x = csr_array(
            (np.array([x.sum() / x.size] * x.size), x.coords),
            shape=x.shape,
        )
        # .mean() compute the mean over all elements.
        mean_y = csr_array(
            (np.array([y.sum() / y.size] * y.size), y.coords),
            shape=y.shape,
        )
    
        x_centered -= mean_x
        y_centered -= mean_y

    # Get indices of common elements.
    common_idxs = np.intersect1d(x_centered.indices, y_centered.indices)
    # Compute norms based on common elements.
    # Note: The lecture presents two implementations for the
    # centered cosine similarity for sparse arrays.
    # The first one (Lec. 1, slide 35) computes the norm
    # using all elements. This corresponds to the commented
    # code below. On the next slide, however, only the 
    # common elements are used. This corresponds to the code below.
    # Depending on the implementation, different
    # results are produced (i.e., lower similaritt for full norm).
    #
    # In the context of 1d. arrays, the common elements implementation
    # doesn't produce overhead to implement. However, for matrices it leads
    # to massive overhead. This is why, for the exercise,
    # we will compute the norm over all the elements.
    # norm_x = np.sqrt((x_centered * x_centered).sum())
    # norm_y = np.sqrt((y_centered * y_centered).sum())

    norm_x = np.linalg.norm([x_centered[idx] for idx in common_idxs])
    norm_y = np.linalg.norm([y_centered[idx] for idx in common_idxs])

    return x_centered.dot(y_centered) / (norm_x * norm_y)


def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = um_normalized.T @ vector
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(similarities[users_who_rated])
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            rating_of_item = np.round(
                (
                    np.nanmean(orig_utility_matrix[:, user_index])
                    + (
                        np.sum(similarities[best_among_who_rated] * clean_utility_matrix[item_index, best_among_who_rated])
                        / np.sum(np.abs(similarities[best_among_who_rated]))
                    )
                ),
                1,
            )
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings
