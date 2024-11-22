import paddle
"""
Training a SFNO on the spherical Shallow Water equations
==========================================================

In this example, we demonstrate how to use the small Spherical Shallow Water Equations example we ship with the package
to train a Spherical Fourier-Neural Operator
"""
import matplotlib.pyplot as plt
import sys
from neuralop.models import SFNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_spherical_swe
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
device = str('cuda:0' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
train_loader, test_loaders = load_spherical_swe(n_train=200, batch_size=4,
    train_resolution=(32, 64), test_resolutions=[(32, 64), (64, 128)],
    n_tests=[50, 50], test_batch_sizes=[10, 10])
model = SFNO(n_modes=(32, 32), in_channels=3, out_channels=3,
    hidden_channels=32, projection_channels=64, factorization='dense')
model = model.to(device)
n_params = count_model_params(model)
print(f"""
Our model has {n_params} parameters.""")
sys.stdout.flush()
optimizer = AdamW(model.parameters(), lr=0.0008, weight_decay=0.0)
tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=
    optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr
l2loss = LpLoss(d=2, p=2, reduce_dims=(0, 1))
train_loss = l2loss
eval_losses = {'l2': l2loss}
print('\n### MODEL ###\n', model)
print("""
### OPTIMIZER ###
""", optimizer)
print("""
### SCHEDULER ###
""", scheduler)
print('\n### LOSSES ###')
print(f"""
 * Train: {train_loss}""")
print(f"""
 * Test: {eval_losses}""")
sys.stdout.flush()
trainer = Trainer(model=model, n_epochs=20, device=device, wandb_log=False,
    eval_interval=3, use_distributed=False, verbose=True)
trainer.train(train_loader=train_loader, test_loaders=test_loaders,
    optimizer=optimizer, scheduler=scheduler, regularizer=False,
    training_loss=train_loss, eval_losses=eval_losses)
fig = plt.figure(figsize=(7, 7))
for index, resolution in enumerate([(32, 64), (64, 128)]):
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    x = data['x']
    y = data['y'][0, ...].numpy()
    x_in = x.unsqueeze(axis=0).to(device)
    out = model(x_in).squeeze()[0, ...].detach().cpu().numpy()
    x = x[0, ...].detach().numpy()
    ax = fig.add_subplot(2, 3, index * 3 + 1)
    ax.imshow(x)
    ax.set_title(f'Input x {resolution}')
    plt.xticks([], [])
    plt.yticks([], [])
    ax = fig.add_subplot(2, 3, index * 3 + 2)
    ax.imshow(y)
    ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])
    ax = fig.add_subplot(2, 3, index * 3 + 3)
    ax.imshow(out)
    ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
