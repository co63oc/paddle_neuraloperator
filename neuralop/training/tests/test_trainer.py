import shutil
from pathlib import Path

import paddle

from neuralop import H1Loss
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models import FNO
from neuralop.tests.test_utils import DummyDataset
from neuralop.tests.test_utils import DummyModel
from neuralop.training import IncrementalFNOTrainer


def test_model_checkpoint_saves():
    save_pth = Path("./test_checkpoints")
    model = DummyModel(50)
    train_loader = paddle.io.DataLoader(dataset=DummyDataset(100))
    trainer = Trainer(model=model, n_epochs=5)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.0003, weight_decay=0.0001
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    trainer.train(
        train_loader=train_loader,
        test_loaders={},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=None,
        save_dir=save_pth,
        save_every=1,
    )
    for file_ext in [
        "model_state_dict.pt",
        "model_metadata.pkl",
        "optimizer.pt",
        "scheduler.pt",
    ]:
        file_pth = save_pth / file_ext
        assert file_pth.exists()

    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./test_checkpoints")


def test_model_checkpoint_and_resume():
    save_pth = Path("./full_states")
    model = DummyModel(50)
    train_loader = paddle.io.DataLoader(dataset=DummyDataset(100))
    test_loader = paddle.io.DataLoader(dataset=DummyDataset(20))
    trainer = Trainer(model=model, n_epochs=5)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.0003, weight_decay=0.0001
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    eval_losses = {"h1": h1loss, "l2": l2loss}
    trainer.train(
        train_loader=train_loader,
        test_loaders={"test": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        save_best="test_h1",
        save_dir=save_pth,
        save_every=1,
    )
    for file_ext in [
        "best_model_state_dict.pt",
        "best_model_metadata.pkl",
        "optimizer.pt",
        "scheduler.pt",
    ]:
        file_pth = save_pth / file_ext
        assert file_pth.exists()
    # Resume from checkpoint
    trainer = Trainer(model=model, n_epochs=5)
    errors = trainer.train(  # noqa
        train_loader=train_loader,
        test_loaders={"": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        resume_from_dir=save_pth,
    )
    # clean up dummy checkpoint directory after testing
    shutil.rmtree(save_pth)


# ensure that model accuracy after loading from checkpoint
# is comparable to accuracy at time of save
def test_load_from_checkpoint():
    model = DummyModel(50)
    train_loader = paddle.io.DataLoader(dataset=DummyDataset(100))
    test_loader = paddle.io.DataLoader(dataset=DummyDataset(100))
    trainer = Trainer(model=model, n_epochs=10)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.0003, weight_decay=0.0001
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    eval_losses = {"h1": h1loss, "l2": l2loss}
    orig_model_eval_errors = trainer.train(
        train_loader=train_loader,
        test_loaders={"test": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        save_dir="./full_states",
        save_every=1,
    )
    # create a new model from saved checkpoint and evaluate
    loaded_model = DummyModel.from_checkpoint(save_folder="./full_states", save_name="model")
    trainer = Trainer(model=loaded_model, n_epochs=1)
    loaded_model_eval_errors = trainer.evaluate(
        loss_dict=eval_losses, data_loader=test_loader, log_prefix="test"
    )
    # test l2 difference should be small
    assert (
        orig_model_eval_errors["test_l2"] - loaded_model_eval_errors["test_l2"]
    ) / orig_model_eval_errors["test_l2"] < 0.1
    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./full_states")


# Adam not support complex64
# enure that the model incrementally increases in frequency modes
def _test_incremental():
    # Loading the Darcy flow dataset
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=10,
        batch_size=16,
        test_resolutions=[16, 32],
        n_tests=[10, 5],
        test_batch_sizes=[32, 32],
    )
    initial_n_modes = 2, 2
    initial_max_modes = 16, 16
    model = FNO(
        n_modes=initial_n_modes,
        max_n_modes=initial_max_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    )
    trainer = IncrementalFNOTrainer(
        model=model,
        n_epochs=20,
        incremental_loss_gap=False,
        incremental_grad=True,
        incremental_grad_eps=0.9999,
        incremental_buffer=5,
        incremental_max_iter=1,
        incremental_grad_max_iter=2,
    )
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.0003, weight_decay=0.0001
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    eval_losses = {"h1": h1loss, "l2": l2loss}
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
    )
    # assert that the model has increased in frequency modes
    for i in range(len(initial_n_modes)):
        assert model.fno_blocks.convs.n_modes[i] > initial_n_modes[i]

    # assert that the model has not changed the max modes
    for i in range(len(initial_max_modes)):
        assert model.fno_blocks.convs.max_n_modes[i] == initial_max_modes[i]
