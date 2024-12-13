import paddle

from neuralop import paddle_aux  # noqa


class AdaIN(paddle.nn.Layer):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-05):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps
        if mlp is None:
            mlp = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=embed_dim, out_features=512),
                paddle.nn.GELU(),
                paddle.nn.Linear(in_features=512, out_features=2 * in_channels),
            )
        self.mlp = mlp
        self.embedding = None

    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim)

    def forward(self, x):
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"
        weight, bias = paddle_aux.split(
            x=self.mlp(self.embedding), num_or_sections=self.in_channels, axis=0
        )
        return paddle.nn.functional.group_norm(
            x=x, num_groups=self.in_channels, weight=weight, bias=bias, epsilon=self.eps
        )


class InstanceNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Layer.

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = tuple(x.shape)
        x = paddle.nn.functional.instance_norm(x, **self.kwargs)
        assert tuple(x.shape) == size
        return x
