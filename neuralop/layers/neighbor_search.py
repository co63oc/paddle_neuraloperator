import paddle

import neuralop.paddle_aux  # noqa


# Requires either open3d instalation or paddle_cluster
# Uses open3d by default which, as of 07/23/2023
class NeighborSearch(paddle.nn.Layer):
    """
    Neighborhood search between two arbitrary coordinate meshes.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data`
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or native Paddle implementation
        NOTE: open3d implementation requires 3d data
    """

    def __init__(self, use_open3d=False):  # Paddle not support open3d
        super().__init__()
        if use_open3d:  # slightly faster, works on GPU in 3d only
            pass
            import open3d

            self.search_fn = open3d.ml.torch.layers.FixedRadiusSearch()
            self.use_open3d = use_open3d
        else:  # slower fallback, works on GPU and CPU
            self.search_fn = native_neighbor_search
            self.use_open3d = False

    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : paddle.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : paddle.Tensor of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: paddle.Tensor with dtype=paddle.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and paddle_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: paddle.Tensor of shape [m+1] with dtype=paddle.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}
        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict["neighbors_index"] = search_return.neighbors_index.long()
            return_dict["neighbors_row_splits"] = search_return.neighbors_row_splits.long()
        else:
            return_dict = self.search_fn(data, queries, radius)
        return return_dict


def native_neighbor_search(data: paddle.Tensor, queries: paddle.Tensor, radius: float):
    """
    Native Paddle implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    -----------

    data : paddle.Tensor
        vector of data points from which to find neighbors
    queries : paddle.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    # compute pairwise distances
    dists = paddle.cdist(x=queries, y=data).to(
        queries.place
    )  # shaped num query points x num data points
    in_nbr = paddle.where(
        condition=dists <= radius, x=1.0, y=0.0
    )  # i,j is one if j is i's neighbor
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(-1)  # only keep the column indices
    nbrhd_sizes = paddle.cumsum(x=paddle.sum(x=in_nbr, axis=1), axis=0).astype(
        paddle.float32
    )  # num points in each neighborhood, summed cumulatively
    splits = paddle.concat(x=(paddle.to_tensor(data=[0.0]).to(queries.place), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict["neighbors_index"] = nbr_indices.astype(dtype="int64").to(queries.place)
    nbr_dict["neighbors_row_splits"] = splits.astype(dtype="int64")
    return nbr_dict
