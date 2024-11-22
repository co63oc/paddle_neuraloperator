import paddle
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import wandb
import optuna
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_darcy_flow_small
from neuralop.training import setup
from neuralop.training.callbacks import MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_params
config_name = 'default'
pipe = ConfigPipeline([YamlConfig('./darcy_config.yaml', config_name=
    'default', config_folder='../config'), ArgparseConfig(infer_types=True,
    config_name=None, config_file=None), YamlConfig(config_folder='../config')]
    )
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
device, is_logger = setup(config)
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
    wandb.init(config=config, name=wandb_name, group=config.wandb.group,
        project=config.wandb.project, entity=config.wandb.entity)
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
config.verbose = config.verbose and is_logger
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()
train_loader, test_loaders, output_encoder = load_darcy_flow_small(n_train=
    config.data.n_train, batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding, test_resolutions=
    config.data.test_resolutions, n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes, encode_input=config.data
    .encode_input, encode_output=config.data.encode_output)


def objective(trial):
    config = pipe.read_conf()
    learning_rate = trial.suggest_float('learning_rate', 5e-05, 0.5)
    batch_size = trial.suggest_float('batch_size', 8, 64)
    config.opt.learning_rate = learning_rate
    config.opt.batch_size = batch_size
    config.opt.n_epochs = 10
    model = get_model(config)
    model = model.to(device)
    if config.distributed.use_distributed:
>>>>>>        model = torch.nn.parallel.DistributedDataParallel(model, device_ids
            =[device.index], output_device=device.index, static_graph=True)
    if is_logger:
        n_params = count_params(model)
        if config.verbose:
            print(f'\nn_params: {n_params}')
            sys.stdout.flush()
        if config.wandb.log:
            to_log = {'n_params': n_params}
            if config.n_params_baseline is not None:
                to_log['n_params_baseline'] = config.n_params_baseline,
                to_log['compression_ratio'
                    ] = config.n_params_baseline / n_params,
                to_log['space_savings'
                    ] = 1 - n_params / config.n_params_baseline
            wandb.log(to_log)
            wandb.watch(model)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=config.opt.learning_rate, weight_decay=config.opt.
        weight_decay)
    if config.opt.scheduler == 'ReduceLROnPlateau':
        tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(factor=config.opt.
            gamma, patience=config.opt.scheduler_patience, mode='min',
            learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    elif config.opt.scheduler == 'CosineAnnealingLR':
        tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=config.opt.
            scheduler_T_max, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    elif config.opt.scheduler == 'StepLR':
        tmp_lr = paddle.optimizer.lr.StepDecay(step_size=config.opt.
            step_size, gamma=config.opt.gamma, learning_rate=optimizer.get_lr()
            )
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    else:
        raise ValueError(f'Got scheduler={config.opt.scheduler}')
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    if config.opt.training_loss == 'l2':
        train_loss = l2loss
    elif config.opt.training_loss == 'h1':
        train_loss = h1loss
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} but expected one of ["l2", "h1"]'
            )
    eval_losses = {'h1': h1loss, 'l2': l2loss}
    if config.verbose and is_logger:
        print('\n### MODEL ###\n', model)
        print('\n### OPTIMIZER ###\n', optimizer)
        print('\n### SCHEDULER ###\n', scheduler)
        print('\n### LOSSES ###')
        print(f'\n * Train: {train_loss}')
        print(f'\n * Test: {eval_losses}')
        print(f'\n### Beginning Training...\n')
        sys.stdout.flush()
    trainer = Trainer(model=model, n_epochs=config.opt.n_epochs, device=
        device, amp_autocast=config.opt.amp_autocast, wandb_log=config.
        wandb.log, log_test_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output, use_distributed=config.
        distributed.use_distributed, verbose=config.verbose and is_logger,
        callbacks=[MGPatchingCallback(levels=config.patching.levels,
        padding_fraction=config.patching.padding, stitching=config.patching
        .stitching, encoder=output_encoder), SimpleWandBLoggerCallback()])
    errors = trainer.train(train_loader=train_loader, test_loaders=
        test_loaders, optimizer=optimizer, scheduler=scheduler, regularizer
        =False, training_loss=train_loss, eval_losses=eval_losses)
    if config.wandb.log and is_logger:
        wandb.finish()
    return errors['32_h1']


study = optuna.create_study()
study.optimize(objective, n_trials=100)
