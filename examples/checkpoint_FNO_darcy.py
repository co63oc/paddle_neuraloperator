"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""
import sys

import paddle

from neuralop import H1Loss
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models import TFNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params

device = "cpu"
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)
# We create a tensorized FNO model
model = TFNO(
    n_modes=(16, 16),
    in_channels=1,
    hidden_channels=32,
    projection_channels=64,
    factorization="tucker",
    rank=0.42,
)
model = model.to(device)
n_params = count_model_params(model)
print(
    f"""
Our model has {n_params} parameters."""
)
sys.stdout.flush()
# Create the optimizer
optimizer = AdamW(model.parameters(), lr=0.008, weight_decay=0.0001)
tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
print("\n### MODEL ###\n", model)
print(
    """
### OPTIMIZER ###
""",
    optimizer,
)
print(
    """
### SCHEDULER ###
""",
    scheduler,
)
print("\n### LOSSES ###")
print(
    f"""
 * Train: {train_loss}"""
)
print(
    f"""
 * Test: {eval_losses}"""
)
sys.stdout.flush()
# Create the trainer
trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=3,
    use_distributed=False,
    verbose=True,
)
trainer.train(
    # Actually train the model on our small Darcy-Flow dataset
    train_loader=train_loader,
    test_loaders={},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    save_every=1,
    save_dir="./checkpoints",
)
# resume training from saved checkpoint at epoch 10
trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=3,
    use_distributed=False,
    verbose=True,
)
trainer.train(
    train_loader=train_loader,
    test_loaders={},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    resume_from_dir="./checkpoints",
)
