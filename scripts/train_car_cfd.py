import paddle
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import MeshDataModule
from neuralop.data.transforms.data_processors import DataProcessor
from copy import deepcopy
config_name = 'cfd'
pipe = ConfigPipeline([YamlConfig('./car_cfd_config.yaml', config_name=
    config_name, config_folder='../config'), ArgparseConfig(infer_types=
    True, config_name=None, config_file=None), YamlConfig(config_folder=
    '../config')])
config = pipe.read_conf()
device, is_logger = setup(config)
if config.data.sdf_query_resolution < config.fnogno.fno_n_modes[0]:
    config.fnogno.fno_n_modes = [config.data.sdf_query_resolution] * 3
wandb_init_args = {}
config_name = 'car-pressure'
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(f'{var}' for var in [config_name, config.data
            .sdf_query_resolution])
    wandb_init_args = dict(config=config, name=wandb_name, group=config.
        wandb.group, project=config.wandb.project, entity=config.wandb.entity)
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
data_module = MeshDataModule(config.data.path, config.data.entity_name,
    query_res=[config.data.sdf_query_resolution] * 3, n_train=config.data.
    n_train, n_test=config.data.n_test, attributes=config.data.load_attributes)
train_loader = data_module.train_dataloader(batch_size=1, shuffle=True)
test_loader = data_module.test_dataloader(batch_size=1, shuffle=False)
model = get_model(config)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay)
if config.opt.scheduler == 'ReduceLROnPlateau':
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(factor=config.opt.gamma,
        patience=config.opt.scheduler_patience, mode='min', learning_rate=
        optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
elif config.opt.scheduler == 'CosineAnnealingLR':
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=config.opt.
        scheduler_T_max, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
elif config.opt.scheduler == 'StepLR':
    tmp_lr = paddle.optimizer.lr.StepDecay(step_size=config.opt.step_size,
        gamma=config.opt.gamma, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
else:
    raise ValueError(f'Got config.opt.scheduler={config.opt.scheduler!r}')
l2loss = LpLoss(d=2, p=2)
if config.opt.training_loss == 'l2':
    train_loss_fn = l2loss
else:
    raise ValueError(
        f'Got config.opt.training_loss={config.opt.training_loss!r}')
if config.opt.testing_loss == 'l2':
    test_loss_fn = l2loss
else:
    raise ValueError(f'Got config.opt.testing_loss={config.opt.testing_loss!r}'
        )


class CFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an FNOGNO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()
        self.normalizer = normalizer
        self.device = device

    def preprocess(self, sample):
        in_p = sample['query_points'].squeeze(axis=0).to(self.device)
        out_p = sample['centroids'].squeeze(axis=0).to(self.device)
        f = sample['distance'].squeeze(axis=0).to(self.device)
        weights = sample['triangle_areas'].squeeze(axis=0).to(self.device)
        truth = sample['press'].squeeze(axis=0).unsqueeze(axis=-1)
        output_vertices = tuple(truth.shape)[1]
        if tuple(out_p.shape)[0] > output_vertices:
            out_p = out_p[:output_vertices, :]
        truth = truth.to(device)
        inward_normals = -sample['triangle_normals'].squeeze(axis=0).to(self
            .device)
        flow_normals = paddle.zeros(shape=(tuple(sample[triangle_areas].
            shape)[1], 3)).to(self.device)
        flow_normals[:, 0] = -1.0
        batch_dict = dict(in_p=in_p, out_p=out_p, f=f, y=truth,
            inward_normals=inward_normals, flow_normals=flow_normals,
            flow_speed=None, vol_elm=weights, reference_area=None)
        sample.update(batch_dict)
        return sample

    def postprocess(self, out, sample):
        out = self.normalizer.inverse_transform(out)
        y = self.normalizer.inverse_transform(sample['y'].squeeze(axis=0))
        sample['y'] = y
        return out, sample

    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


output_encoder = deepcopy(data_module.normalizers['press']).to(device)
data_processor = CFDDataProcessor(normalizer=output_encoder, device=device)
trainer = Trainer(model=model, n_epochs=config.opt.n_epochs, data_processor
    =data_processor, device=device, wandb_log=config.wandb.log)
if config.wandb.log:
    wandb.log({'time_to_distance': data_module.time_to_distance}, commit=False)
trainer.train(train_loader=train_loader, test_loaders={'': test_loader},
    optimizer=optimizer, scheduler=scheduler, training_loss=train_loss_fn,
    eval_losses={config.opt.testing_loss: test_loss_fn}, regularizer=None)
