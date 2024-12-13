from copy import deepcopy

import paddle

from .base_model import BaseModel


class UQNO(BaseModel, name="UQNO"):
    """General N-dim (alpha, delta) Risk-Controlling Neural Operator
    Source: https://arxiv.org/abs/2402.01960

    The UQNO must be trained on

    Parameters
    ----------
    base_model : nn.Layer
        pre-trained solution operator
    residual_model : nn.Layer, optional
        architecture to train as the UQNO's
        quantile model
    """

    def __init__(
        self, base_model: paddle.nn.Layer, residual_model: paddle.nn.Layer = None, **kwargs
    ):
        super().__init__()
        self.base_model = base_model
        if residual_model is None:
            residual_model = deepcopy(base_model)
        self.residual_model = residual_model

    def forward(self, *args, **kwargs):
        """
        Forward pass returns the solution u(a,x)
        and the uncertainty ball E(a,x) as a pair
        for pointwise quantile loss
        """
        self.base_model.eval()  # base-model weights are frozen
        # another way to handle this would be to use LoRA, or similar
        # ie freeze the  weights, and train a low-rank matrix of weight perturbations
        with paddle.no_grad():
            solution = self.base_model(*args, **kwargs)
        quantile = self.residual_model(*args, **kwargs)
        return solution, quantile
