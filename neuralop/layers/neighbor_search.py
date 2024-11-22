import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle


class NeighborSearch(paddle.nn.Layer):
    """
    Neighborhood search between two arbitrary coordinate meshes.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data` 
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or native PyTorch implementation
        NOTE: open3d implementation requires 3d data
    """

    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d:
            pass
            self.search_fn = open3d.ml.torch.layers.FixedRadiusSearch()
            self.use_open3d = use_open3d
        else:
            self.search_fn = native_neighbor_search
            self.use_open3d = False

    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the 
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}
        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict['neighbors_index'
                ] = search_return.neighbors_index.long()
            return_dict['neighbors_row_splits'
                ] = search_return.neighbors_row_splits.long()
        else:
            return_dict = self.search_fn(data, queries, radius)
        return return_dict


def native_neighbor_search(data: paddle.Tensor, queries: paddle.Tensor,
    radius: float):
    """
    Native PyTorch implementation of a neighborhood search
    between two arbitrary coordinate meshes.
     
    Parameters
    -----------

    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """
    dists = paddle.cdist(x=queries, y=data).to(queries.place)
    in_nbr = paddle.where(condition=dists <= radius, x=1.0, y=0.0)
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(-1)
    nbrhd_sizes = paddle.cumsum(x=paddle.sum(x=in_nbr, axis=1), axis=0)
    splits = paddle.concat(x=(paddle.to_tensor(data=[0.0]).to(queries.place
        ), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.astype(dtype='int64').to(queries
        .place)
    nbr_dict['neighbors_row_splits'] = splits.astype(dtype='int64')
    return nbr_dict
