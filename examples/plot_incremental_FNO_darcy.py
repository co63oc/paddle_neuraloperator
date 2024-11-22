import paddle
"""
Training a neural operator on Darcy-Flow - Author Robert Joseph George
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO and Incremental Resolution
"""
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop.training.incremental import IncrementalFNOTrainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from neuralop import LpLoss, H1Loss
train_loader, test_loaders, output_encoder = load_darcy_flow_small(n_train=
    100, batch_size=16, test_resolutions=[16, 32], n_tests=[100, 50],
    test_batch_sizes=[32, 32])
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
incremental = True
if incremental:
    starting_modes = 2, 2
else:
    starting_modes = 16, 16
model = FNO(max_n_modes=(16, 16), n_modes=starting_modes, hidden_channels=
    32, in_channels=1, out_channels=1)
model = model.to(device)
n_params = count_model_params(model)
optimizer = AdamW(model.parameters(), lr=0.008, weight_decay=0.0001)
tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=30, learning_rate=
    optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr
data_transform = IncrementalDataProcessor(in_normalizer=None,
    out_normalizer=None, device=device, subsampling_rates=[2, 1],
    dataset_resolution=16, dataset_indices=[2, 3], epoch_gap=10, verbose=True)
data_transform = data_transform.to(device)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}
print("""
### N PARAMS ###
""", n_params)
print("""
### OPTIMIZER ###
""", optimizer)
print("""
### SCHEDULER ###
""", scheduler)
print('\n### LOSSES ###')
print("""
### INCREMENTAL RESOLUTION + GRADIENT EXPLAINED ###""")
print(f"""
 * Train: {train_loss}""")
print(f"""
 * Test: {eval_losses}""")
sys.stdout.flush()
trainer = IncrementalFNOTrainer(model=model, n_epochs=20, data_processor=
    data_transform, device=device, verbose=True, incremental_loss_gap=False,
    incremental_grad=True, incremental_grad_eps=0.9999,
    incremental_loss_eps=0.001, incremental_buffer=5, incremental_max_iter=
    1, incremental_grad_max_iter=2)
trainer.train(train_loader, test_loaders, optimizer, scheduler, regularizer
    =False, training_loss=train_loss, eval_losses=eval_losses)
test_samples = test_loaders[32].dataset
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    x = data['x'].to(device)
    y = data['y'].to(device)
    out = model(x.unsqueeze(axis=0))
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    x = x.cpu().squeeze().detach().numpy()
    y = y.cpu().squeeze().detach().numpy()
    ax.imshow(x, cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.cpu().squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
