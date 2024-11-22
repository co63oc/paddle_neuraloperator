import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle
import sys
import copy
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import numpy as np
import wandb
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import DataProcessor, DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses.data_losses import PointwiseQuantileLoss
from neuralop.models import UQNO
from neuralop.training import setup
from neuralop.utils import get_wandb_api_key, count_model_params
config_name = 'default'
pipe = ConfigPipeline([YamlConfig('./uqno_config.yaml', config_name=
    'default', config_folder='../config'), ArgparseConfig(infer_types=True,
    config_name=None, config_file=None), YamlConfig(config_folder='../config')]
    )
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
device, is_logger = setup(config)
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(f'{var}' for var in [config_name, config.
            tfno2d.n_layers, config.tfno2d.hidden_channels, config.tfno2d.
            n_modes_width, config.tfno2d.n_modes_height, config.tfno2d.
            factorization, config.tfno2d.rank, config.patching.levels,
            config.patching.padding])
    wandb_args = dict(config=config, name=wandb_name, group=config.wandb.
        group, project=config.wandb.project, entity=config.wandb.entity)
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)
config.verbose = config.verbose and is_logger
if config.verbose and is_logger and config.opt.solution.n_epochs > 0:
    pipe.log()
    sys.stdout.flush()
solution_dataset = neuralop.data.datasets.darcy.DarcyDataset(root_dir=
    config.data.root, n_train=config.data.n_train_total, n_tests=[config.
    data.n_test], batch_size=config.data.batch_size, test_batch_sizes=[
    config.data.test_batch_size], train_resolution=421, test_resolutions=[
    421], encode_input=config.data.encode_input, encode_output=config.data.
    encode_output)
train_db = solution_dataset.train_db
test_db = solution_dataset.test_dbs[421]
test_loaders = {(421): paddle.io.DataLoader(dataset=test_db, shuffle=False,
    batch_size=config.data.test_batch_size)}
data_processor = solution_dataset.data_processor
solution_train_db = neuralop.data.datasets.tensor_dataset.TensorDataset(**
    train_db[:config.data.n_train_solution])
residual_train_db = neuralop.data.datasets.tensor_dataset.TensorDataset(**
    train_db[config.data.n_train_solution:config.data.n_train_solution +
    config.data.n_train_residual])
residual_calib_db = neuralop.data.datasets.tensor_dataset.TensorDataset(**
    train_db[config.data.n_train_solution + config.data.n_train_residual:
    config.data.n_train_solution + config.data.n_train_residual + config.
    data.n_calib_residual])
data_processor = data_processor.to(device)
solution_model = get_model(config)
solution_model = solution_model.to(device)
optimizer = paddle.optimizer.Adam(parameters=solution_model.parameters(),
    learning_rate=config.opt.solution.learning_rate, weight_decay=config.
    opt.solution.weight_decay)
if config.opt.solution.scheduler == 'ReduceLROnPlateau':
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(factor=config.opt.solution
        .gamma, patience=config.opt.solution.scheduler_patience, mode='min',
        learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
elif config.opt.solution.scheduler == 'CosineAnnealingLR':
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=config.opt.
        solution.scheduler_T_max, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
elif config.opt.solution.scheduler == 'StepLR':
    tmp_lr = paddle.optimizer.lr.StepDecay(step_size=config.opt.solution.
        step_size, gamma=config.opt.solution.gamma, learning_rate=optimizer
        .get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
else:
    raise ValueError(f'Got scheduler={config.opt.solution.scheduler}')
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.solution.training_loss == 'l2':
    train_loss = l2loss
elif config.opt.solution.training_loss == 'h1':
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.solution.training_loss} but expected one of ["l2", "h1"]'
        )
eval_losses = {'h1': h1loss, 'l2': l2loss}
if config.verbose and is_logger and config.opt.solution.n_epochs > 0:
    print('\n### MODEL ###\n', solution_model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    print(f'\n### Beginning Training...\n')
    sys.stdout.flush()
if is_logger:
    n_params = count_model_params(solution_model)
    if config.verbose:
        print(f'\nn_params: {n_params}')
        sys.stdout.flush()
    if config.wandb.log:
        to_log = {'n_params': n_params}
        if config.n_params_baseline is not None:
            to_log['n_params_baseline'] = config.n_params_baseline,
            to_log['compression_ratio'] = config.n_params_baseline / n_params,
            to_log['space_savings'] = 1 - n_params / config.n_params_baseline
        wandb.log(to_log)
solution_train_loader = paddle.io.DataLoader(dataset=solution_train_db,
    batch_size=config.data.batch_size, shuffle=True, num_workers=1)
trainer = Trainer(model=solution_model, n_epochs=config.opt.solution.
    n_epochs, device=device, data_processor=data_processor, amp_autocast=
    config.opt.solution.amp_autocast, wandb_log=config.wandb.log,
    eval_interval=config.wandb.eval_interval, log_output=config.wandb.
    log_output, use_distributed=config.distributed.use_distributed, verbose
    =config.verbose and is_logger)
if config.opt.solution.n_epochs > 0:
    if config.opt.solution.resume == True:
        resume_dir = './solution_ckpts'
    else:
        resume_dir = None
    trainer.train(train_loader=solution_train_loader, test_loaders=
        test_loaders, optimizer=optimizer, scheduler=scheduler, regularizer
        =False, training_loss=train_loss, eval_losses=eval_losses,
        save_best='421_l2', save_dir='./solution_ckpts', resume_from_dir=
        resume_dir)


def loader_to_residual_db(model, data_processor, loader, device,
    train_val_split=True):
    """
    loader_to_residual_db converts a dataset of x: a(x), y: u(x) to 
    x: a(x), y: G(a,x) - u(x) for use training the residual model.

    model : nn.Module
        trained solution model (frozen)
    data_processor: DataProcessor
        data processor used to train solution model
    loader: DataLoader
        data loader to convert to a dataloader of residuals
        must be drawn from the same distribution as the solution
        model's training distribution
    device: str or torch.device
    train_val_split: whether to split into a training and validation dataset, default True
    """
    error_list = []
    x_list = []
    model = model.to(device)
    model.eval()
    data_processor.eval()
    data_processor = data_processor.to(device)
    for idx, sample in enumerate(loader):
        sample = data_processor.preprocess(sample)
        out = model(**sample)
        out, sample = data_processor.postprocess(out, sample)
        x_list.append(sample['x'].to('cpu'))
        error = (out - sample['y']).detach().to('cpu')
        error_list.append(error)
        del sample, out
>>>>>>    errors = torch.cat(error_list, axis=0)
>>>>>>    xs = torch.cat(x_list, axis=0)
    residual_encoder = UnitGaussianNormalizer()
    residual_encoder.fit(errors)
    residual_data_processor = DefaultDataProcessor(in_normalizer=None,
        out_normalizer=residual_encoder)
    residual_data_processor.train()
    if train_val_split:
        val_start = int(0.8 * tuple(xs.shape)[0])
        residual_train_db = (neuralop.data.datasets.tensor_dataset.
            TensorDataset(x=xs[:val_start], y=errors[:val_start]))
        residual_val_db = neuralop.data.datasets.tensor_dataset.TensorDataset(x
            =xs[val_start:], y=errors[val_start:])
    else:
        residual_val_db = None
    return residual_train_db, residual_val_db, residual_data_processor


class UQNODataProcessor(DataProcessor):

    def __init__(self, base_data_processor: DataProcessor,
        resid_data_processor: DataProcessor, device: str='cpu'):
        """UQNODataProcessor converts tuple (G_hat(a,x), E(a,x)) and 
        sample['y'] = G_true(a,x) into the form expected by PointwiseQuantileLoss

        y_pred = E(a,x)
        y_true = abs(G_hat(a,x) - G_true(a,x))

        It also preserves any transformations that need to be performed
        on inputs/outputs from the solution model. 

        Parameters
        ----------
        base_data_processor : DataProcessor
            transforms required for base solution_model input/output
        resid_data_processor : DataProcessor
            transforms required for residual input/output
        device: str
            "cpu" or "cuda" 
        """
        super().__init__()
        self.base_data_processor = base_data_processor
        self.residual_normalizer = resid_data_processor.out_normalizer
        self.device = device
        self.scale_factor = None

    def set_scale_factor(self, factor):
        self.scale_factor = factor.to(device)

    def wrap(self, model):
        self.model = model
        return self

    def to(self, device):
        self.device = device
        self.base_data_processor = self.base_data_processor.to(device)
        self.residual_normalizer = self.residual_normalizer.to(device)
        return self

    def train(self):
        self.base_data_processor.train()

    def eval(self):
        self.base_data_processor.eval()

    def preprocess(self, *args, **kwargs):
        """
        nothing required at preprocessing - just wrap the base DataProcessor
        """
        return self.base_data_processor.preprocess(*args, **kwargs)

    def postprocess(self, out, sample):
        """
        unnormalize the residual prediction as well as the output
        """
        self.base_data_processor.eval()
        g_hat, pred_uncertainty = out
        pred_uncertainty = self.residual_normalizer.inverse_transform(
            pred_uncertainty)
        g_hat, sample = self.base_data_processor.postprocess(g_hat, sample)
        g_true = sample['y']
        sample['y'] = g_true - g_hat
        sample.pop('x')
        if self.scale_factor is not None:
            pred_uncertainty = pred_uncertainty * self.scale_factor
        return pred_uncertainty, sample

    def forward(self, **sample):
        sample = self.preprocess(sample)
        out = self.model(**sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


solution_model = solution_model.from_checkpoint(save_folder=
    './solution_ckpts', save_name='best_model_815')
solution_model = solution_model.to(device)
eval_metrics = trainer.evaluate(eval_losses, data_loader=test_loaders[421],
    epoch=1)
print(f'Eval metrics = {eval_metrics}')
residual_model = copy.deepcopy(solution_model)
residual_model = residual_model.to(device)
quantile_loss = PointwiseQuantileLoss(alpha=1 - config.opt.alpha)
residual_optimizer = paddle.optimizer.Adam(parameters=residual_model.
    parameters(), learning_rate=config.opt.residual.learning_rate,
    weight_decay=config.opt.residual.weight_decay)
if wandb_args is not None:
    uq_wandb_name = 'uq_' + wandb_args['name']
    wandb_args['name'] = uq_wandb_name
residual_train_loader_unprocessed = paddle.io.DataLoader(dataset=
    residual_train_db, batch_size=1, shuffle=True, num_workers=0)
(processed_residual_train_db, processed_residual_val_db,
    residual_data_processor) = (loader_to_residual_db(solution_model,
    data_processor, residual_train_loader_unprocessed, device))
residual_data_processor = residual_data_processor.to(device)
residual_train_loader = paddle.io.DataLoader(dataset=
    processed_residual_train_db, batch_size=config.data.batch_size, shuffle
    =True, num_workers=0)
residual_val_loader = paddle.io.DataLoader(dataset=
    processed_residual_val_db, batch_size=config.data.batch_size, shuffle=
    True, num_workers=0)
if config.opt.residual.scheduler == 'ReduceLROnPlateau':
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(factor=config.opt.residual
        .gamma, patience=config.opt.residual.scheduler_patience, mode='min',
        learning_rate=residual_optimizer.get_lr())
    residual_optimizer.set_lr_scheduler(tmp_lr)
    resid_scheduler = tmp_lr
elif config.opt.residual.scheduler == 'CosineAnnealingLR':
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=config.opt.
        residual.scheduler_T_max, learning_rate=residual_optimizer.get_lr())
    residual_optimizer.set_lr_scheduler(tmp_lr)
    resid_scheduler = tmp_lr
elif config.opt.residual.scheduler == 'StepLR':
    tmp_lr = paddle.optimizer.lr.StepDecay(step_size=config.opt.solution.
        step_size, gamma=config.opt.solution.gamma, learning_rate=
        residual_optimizer.get_lr())
    residual_optimizer.set_lr_scheduler(tmp_lr)
    resid_scheduler = tmp_lr
else:
    raise ValueError(f'Got residual scheduler={config.opt.residual.scheduler}')
if config.opt.residual.n_epochs > 0:
    residual_trainer = Trainer(model=residual_model, n_epochs=config.opt.
        residual.n_epochs, data_processor=residual_data_processor,
        wandb_log=config.wandb.log, device=device, amp_autocast=config.opt.
        residual.amp_autocast, eval_interval=config.wandb.eval_interval,
        log_output=config.wandb.log_output, use_distributed=config.
        distributed.use_distributed, verbose=config.verbose and is_logger)
    residual_trainer.train(train_loader=residual_train_loader, test_loaders
        ={'test': residual_val_loader}, optimizer=residual_optimizer,
        scheduler=resid_scheduler, regularizer=False, training_loss=
        quantile_loss, eval_losses={'quantile': quantile_loss, 'l2': l2loss
        }, save_best='test_quantile', save_dir='./residual_ckpts')
residual_model = residual_model.from_checkpoint(save_name='best_model',
    save_folder='./residual_ckpts')
residual_model = residual_model.to(device)


def get_coeff_quantile_idx(alpha, delta, n_samples, n_gridpts):
    """
    get the index of (ranked) sigma's for given delta and t
    we take the min alpha for given delta
    delta is percentage of functions that satisfy alpha threshold in domain
    alpha is percentage of points in ball on domain
    return 2 idxs
    domain_idx is the k for which kth (ranked descending by ptwise |err|/quantile_model_pred_err)
    value we take per function
    func_idx is the j for which jth (ranked descending) value we take among n_sample functions
    Note: there is a min alpha we can take based on number of gridpoints, n and delta, we specify lower bounds lb1 and lb2
    t needs to be between the lower bound and alpha
    """
    lb = np.sqrt(-np.log(delta) / 2 / n_gridpts)
    t = (alpha - lb) / 3 + lb
    print(f'we set alpha (on domain): {alpha}, t={t}')
    percentile = alpha - t
    domain_idx = int(np.ceil(percentile * n_gridpts))
    print(f"domain index: {domain_idx}'th largest of {n_gridpts}")
    function_percentile = np.ceil((n_samples + 1) * (delta - np.exp(-2 *
        n_gridpts * t * t))) / n_samples
    function_idx = int(np.ceil(function_percentile * n_samples))
    print(f"function index: {function_idx}'th largest of {n_samples}")
    return domain_idx, function_idx


uqno = UQNO(base_model=solution_model, residual_model=residual_model)
uqno_data_proc = UQNODataProcessor(base_data_processor=data_processor,
    resid_data_processor=residual_data_processor, device=device)
uqno_data_proc.eval()
val_ratio_list = []
calib_loader = paddle.io.DataLoader(dataset=residual_calib_db, shuffle=True,
    batch_size=1)
with paddle.no_grad():
    for idx, sample in enumerate(calib_loader):
        sample = uqno_data_proc.preprocess(sample)
        out = uqno(sample['x'])
        out, sample = uqno_data_proc.postprocess(out, sample)
        ratio = paddle.abs(x=sample['y']) / out
        val_ratio_list.append(ratio.squeeze().to('cpu'))
        del sample, out
val_ratios = paddle.stack(x=val_ratio_list)
vr_view = val_ratios.view(tuple(val_ratios.shape)[0], -1)


def eval_coverage_bandwidth(test_loader, alpha, device='cuda'):
    """
    Get percentage of instances hitting target-percentage pointwise coverage
    (e.g. pctg of instances with >1-alpha points being covered by quantile model)
    as well as avg band length
    """
    in_pred_list = []
    avg_interval_list = []
    with paddle.no_grad():
        for _, sample in enumerate(test_loader):
            sample = {k: v.to(device) for k, v in sample.items() if paddle.
                is_tensor(x=v)}
            sample = uqno_data_proc.preprocess(sample)
            out = uqno(**sample)
            uncertainty_pred, sample = uqno_data_proc.postprocess(out, sample)
            pointwise_true_err = sample['y']
            in_pred = (paddle.abs(x=pointwise_true_err) < paddle.abs(x=
                uncertainty_pred)).astype(dtype='float32').squeeze()
            avg_interval = paddle.abs(x=uncertainty_pred.squeeze()).view(tuple
                (uncertainty_pred.shape)[0], -1).mean(axis=1)
            avg_interval_list.append(avg_interval.to('cpu'))
            in_pred_flattened = in_pred.view(tuple(in_pred.shape)[0], -1)
            in_pred_instancewise = paddle.mean(x=in_pred_flattened, axis=1
                ) >= 1 - alpha
            in_pred_list.append(in_pred_instancewise.astype(dtype='float32'
                ).to('cpu'))
>>>>>>    in_pred = torch.cat(in_pred_list, axis=0)
>>>>>>    intervals = torch.cat(avg_interval_list, axis=0)
    mean_interval = paddle.mean(x=intervals, axis=0)
    in_pred_percentage = paddle.mean(x=in_pred, axis=0)
    print(
        f'{in_pred_percentage} of instances satisfy that >= {1 - alpha} pts drawn are inside the predicted quantile'
        )
    print(f'Mean interval width is {mean_interval}')
    return mean_interval, in_pred_percentage


for alpha in [0.02, 0.05, 0.1]:
    for delta in [0.02, 0.05, 0.1]:
        darcy_discretization = tuple(train_db[0]['x'].shape)[-1] ** 2
        domain_idx, function_idx = get_coeff_quantile_idx(alpha, delta,
            n_samples=len(calib_loader), n_gridpts=darcy_discretization)
        val_ratios_pointwise_quantile = paddle.topk(k=domain_idx + 1, x=
            val_ratios.view(tuple(val_ratios.shape)[0], -1), axis=1).values[
            :, -1]
        uncertainty_scaling_factor = paddle.abs(x=paddle.topk(k=
            function_idx + 1, x=val_ratios_pointwise_quantile, axis=0).
            values[-1])
        print(f'scale factor: {uncertainty_scaling_factor}')
        uqno_data_proc.set_scale_factor(uncertainty_scaling_factor)
        uqno_data_proc.eval()
        print(f'------- for values alpha={alpha!r} delta={delta!r} ----------')
        interval, percentage = eval_coverage_bandwidth(test_loader=
            test_loaders[tuple(train_db[0]['x'].shape)[-1]], alpha=alpha,
            device=device)
        if config.wandb.log and is_logger:
            wandb.log(interval, percentage)
if config.wandb.log and is_logger:
    wandb.finish()
