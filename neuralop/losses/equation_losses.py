import paddle
from .data_losses import central_diff_2d


class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method='fdm', loss=paddle.nn.functional.
        mse_loss, domain_length=1.0):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u):
        u = u.squeeze(axis=1)
        _, nt, nx = tuple(u.shape)
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd
            =True)
        dudxx = (paddle.roll(x=u, shifts=-1, axis=-1) - 2 * u + paddle.roll
            (x=u, shifts=1, axis=-1)) / dx ** 2
        dudxx[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / dx ** 2
        dudxx[..., -1] = (u[..., -1] - 2 * u[..., -2] + u[..., -3]) / dx ** 2
        right_hand_side = -dudx * u + self.visc * dudxx
        return self.loss(dudt, right_hand_side)

    def __call__(self, y_pred, **kwargs):
        if self.method == 'fdm':
            return self.fdm(u=y_pred)
        raise NotImplementedError()


class ICLoss(object):
    """
    Computes loss for initial value problems.
    """

    def __init__(self, loss=paddle.nn.functional.mse_loss):
        super().__init__()
        self.loss = loss

    def initial_condition_loss(self, y_pred, x):
        boundary_true = x[:, 0, 0, :]
        boundary_pred = y_pred[:, 0, 0, :]
        return self.loss(boundary_pred, boundary_true)

    def __call__(self, y_pred, x, **kwargs):
        return self.initial_condition_loss(y_pred, x)
