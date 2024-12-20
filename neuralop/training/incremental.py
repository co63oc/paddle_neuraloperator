"""
Trainer for Incremental-FNO
"""
import paddle

from neuralop import paddle_aux

from ..models import FNO
from ..models import TFNO
from ..utils import compute_explained_variance
from .trainer import Trainer


class IncrementalFNOTrainer(Trainer):
    """IncrementalFNOTrainer subclasses the Trainer
    to implement specific logic for the Incremental-FNO
    as described in [1]_.

    References
    -----------

    .. _[1]:

    George, R., Zhao, J., Kossaifi, J., Li, Z., and Anandkumar, A. (2024)
        "Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs".
        ArXiv preprint, https://arxiv.org/pdf/2211.15188
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: paddle.nn.Layer = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        incremental_grad: bool = False,
        incremental_loss_gap: bool = False,
        incremental_grad_eps: float = 0.001,
        incremental_buffer: int = 5,
        incremental_max_iter: int = 1,
        incremental_grad_max_iter: int = 10,
        incremental_loss_eps: float = 0.001,
    ):
        assert isinstance(model, FNO) or isinstance(
            self.model, TFNO
        ), f"Error:             IncrementalFNOTrainer is designed to work with FNO or TFNO, instead got            a model of type {model.__class__.__name__}"
        super().__init__(
            model=model,
            n_epochs=n_epochs,
            wandb_log=wandb_log,
            device=device,
            mixed_precision=mixed_precision,
            data_processor=data_processor,
            eval_interval=eval_interval,
            log_output=log_output,
            use_distributed=use_distributed,
            verbose=verbose,
        )
        self.incremental_loss_gap = incremental_loss_gap
        self.incremental_grad = incremental_grad
        self.incremental = self.incremental_loss_gap or self.incremental_grad
        assert (
            self.incremental
        ), "Error: IncrementalTrainer expects at least one incremental algorithm to be True."
        assert not (
            self.incremental_loss_gap and self.incremental_grad
        ), "Error: IncrementalTrainer expects only one incremental algorithm to be True."
        self.incremental_grad_eps = incremental_grad_eps
        self.incremental_buffer = incremental_buffer
        self.incremental_max_iter = incremental_max_iter
        self.incremental_grad_max_iter = incremental_grad_max_iter
        self.incremental_loss_eps = incremental_loss_eps
        self.loss_list = []

    # Main step function: which algorithm to run
    def incremental_update(self, loss=None):
        if self.incremental_loss_gap and loss is not None:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch inherits from the base Trainer's method
            and adds the computation of the incremental-FNO algorithm
            before returning the training epoch's metrics.

        Parameters
        ----------
        epoch : int
            epoch of training
        train_loader : DataLoader
        training_loss : callable
            loss function to train with

        Returns
        -------
        train_err, avg_loss, avg_lasso_loss, epoch_train_time
        """
        self.training = True
        if self.data_processor:
            self.data_processor.epoch = epoch
        train_err, avg_loss, avg_lasso_loss, epoch_train_time = super().train_one_epoch(
            epoch, train_loader, training_loss
        )
        self.incremental_update(avg_loss)

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    # Algorithm 1: Incremental
    def loss_gap(self, loss):
        """
        loss_gap increases the model's incremental modes if
        the epoch's training loss does not decrease sufficiently

        Parameters
        -----------
        loss : float | scalar paddle.Tensor
            scalar value of epoch's training loss
        """
        self.loss_list.append(loss)
        self.ndim = len(self.model.fno_blocks.convs.n_modes)

        # method 1: loss_gap
        incremental_modes = self.model.fno_blocks.convs.n_modes[0]
        max_modes = self.model.fno_blocks.convs.max_n_modes[0]
        if len(self.loss_list) > 1:
            if abs(self.loss_list[-1] - self.loss_list[-2]) <= self.incremental_loss_eps:
                if incremental_modes < max_modes:
                    incremental_modes += 1
        modes_list = tuple([incremental_modes] * self.ndim)
        self.model.fno_blocks.convs.n_modes = modes_list

    # Algorithm 2: Gradient based explained ratio
    def grad_explained(self):
        # for mode 1
        if not hasattr(self, "accumulated_grad"):
            w = self.model.fno_blocks.convs.weight[0]
            self.accumulated_grad = paddle.zeros_like(w.to_tensor())
        if not hasattr(self, "grad_iter"):
            self.grad_iter = 1

        self.ndim = len(self.model.fno_blocks.convs.n_modes)
        if self.grad_iter <= self.incremental_grad_max_iter:
            self.grad_iter += 1
            w = self.model.fno_blocks.convs.weight[0].to_tensor()
            self.accumulated_grad += w
        else:
            incremental_final = []
            for i in range(self.ndim):
                max_modes = self.model.fno_blocks.convs.max_n_modes[i]
                incremental_modes = self.model.fno_blocks.convs.n_modes[i]
                weight = self.accumulated_grad
                strength_vector = []
                for mode_index in range(min(tuple(weight.shape)[1], incremental_modes)):
                    strength = paddle_aux.norm_complex(x=weight[:, mode_index, :], p="fro")
                    strength_vector.append(strength)
                expained_ratio = compute_explained_variance(
                    incremental_modes - self.incremental_buffer,
                    paddle.to_tensor(data=strength_vector),
                )
                if expained_ratio < self.incremental_grad_eps:
                    if incremental_modes < max_modes:
                        incremental_modes += 1
                incremental_final.append(incremental_modes)

            # update the modes and frequency dimensions
            self.grad_iter = 1
            self.accumulated_grad = paddle.zeros_like(
                x=self.model.fno_blocks.convs.weight[0].to_tensor()
            )
            main_modes = incremental_final[0]
            modes_list = tuple([main_modes] * self.ndim)
            self.model.fno_blocks.convs.n_modes = tuple(modes_list)
