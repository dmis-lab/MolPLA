import random
import argparse
import tarfile
import wandb
import pickle
import setproctitle
import os 
import os.path
import json
import numpy as np

from os import kill
from os import getpid
from signal import SIGKILL
import time
from trainer import *
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

num_cpus = os.cpu_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
ngpus_per_node = torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument('--session_name', '-sn', default='defaultsession', type=str)
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_wandb(conf):
    wandb_init = dict()
    wandb_init['project'] = conf.wandb.project_name
    wandb_init['group'] = conf.wandb.session_name
    if not conf.experiment.testing_mode:
        wandb_init['name'] = f'training_{conf.dataprep.dataset}_{conf.dataprep.version}' 
    else:
        wandb_init['name'] = f'testing_{conf.dataprep.dataset}_{conf.dataprep.version}'
    wandb_init['notes'] = conf.wandb.session_name
    os.environ['WANDB_START_METHOD'] = 'thread'

    return wandb_init

def reset_wandb_env():
    exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
    for k, v in os.environ.items():
        if k.startswith('WANDB_') and k not in exclude:
            del os.environ[k]

def load_dataset_collate_model(conf):
    from dataloaders.MolPLA import DatasetBenchmark
    from dataloaders.MolPLA import collate_fn
    from models.MolPLA import NetBench

    dataset = DatasetBenchmark(conf)
    if conf.train_params.finetuning.from_pretrained:
        session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}'
        pretrained_checkpoint = os.path.join(conf.path.checkpoint, session_name)

        pretrained_checkpoint = os.path.join(pretrained_checkpoint, conf.train_params.finetuning.from_pretrained)
        pretrained_checkpoint = os.path.join(pretrained_checkpoint, 'model_config.pkl')
        conf_pretrained = pickle.load(open(pretrained_checkpoint, 'rb'))  
        conf_pretrained.model_params.output_dim   = dataset.output_dim
        conf_pretrained.model_params.dropout_rate = conf.model_params.dropout_rate
        conf_pretrained.train_params.finetuning   = conf.train_params.finetuning

        conf_pretrained.dataprep.dataset          = conf.dataprep.dataset
    net              = NetBench(conf_pretrained)
    num_params       = sum(p.numel() for p in net.parameters())
    num_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of All Model Parameters in [{conf.model_params.model_type}]: {num_params}")
    print(f"Number of Trainable Parameters in [{conf.model_params.model_type}]: {num_params_train}")
    
    dataset.make_scaffold_splits_mk3(random_seed=conf.experiment.random_seed)

    return dataset, collate_fn, net

def run_debug_mode(conf, dataset, collate_fn, net):
    setup_seed(conf.experiment.random_seed)
    setproctitle.setproctitle(f'{conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_debug')

    session_name = 'testproject_testgroup_testsession'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # Distributed DataLoaders
    ddp_batch_size = int(conf.train_params.batch_size/ngpus_per_node)
    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[0]]

    train      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[0], collate_fn=collate_fn)
    valid      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[1], collate_fn=collate_fn)
    test       = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[2], collate_fn=collate_fn)

    rank = 0
    trainer = TrainerBench(conf, rank, None, False)

    # train, valid, test and save the model
    trained_model, train_loss, valid_loss = trainer.train_valid(net, train, valid)
    if rank == 0: print('Finish Debugging Mode')

def run_single_fold(rank, ngpus_per_node, conf, dataset, collate_fn, net):
    setup_seed(conf.experiment.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    process_name = f'{conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_fold_{conf.experiment.fold_num}_gpu_{rank}'
    setproctitle.setproctitle(process_name)
    session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # WANDB setup /// conf
    if rank == 0:
        reset_wandb_env()
        wandb_init = setup_wandb(conf)
        _name = f'_{conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_fold_{conf.experiment.fold_num}'
        wandb_init['name'] += _name
        run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
        run.define_metric('train/step'); run.define_metric('train/*', step_metric='train/step')
        run.define_metric('valid/step'); run.define_metric('valid/*', step_metric='valid/step')
        run.define_metric('test/step'); run.define_metric('test/*', step_metric='test/step')
        run.watch(net, log="gradients", log_freq=10)
    else: run = None

    # initailize pytorch distributed
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', 
            init_method=f'tcp://localhost:{conf.ddp.port}',
            rank=rank, world_size=ngpus_per_node)

    trainer = TrainerBench(conf, rank, run, True)
    ddp_batch_size = conf.train_params.batch_size // ngpus_per_node
    if rank == 0:
        print('Batch size', conf.train_params.batch_size)
        print('Distributed batch size', ddp_batch_size)
    dataloaders, train_sampler = [], None

    for idx, indices in enumerate(dataset.kfold_splits[0]):
        sampler = DistributedSampler(Subset(dataset,indices), shuffle=True, drop_last=True if conf.dataprep.dataset == 'freesolv' else False)
        loader  = DataLoader(Subset(dataset,indices), batch_size=ddp_batch_size, 
                                                      sampler=sampler, 
                                                      collate_fn=collate_fn)
        dataloaders.append(loader)
        if idx == 0: train_sampler = sampler
    del dataset
    train, valid, test = dataloaders

    if conf.dev_mode.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 1
    if rank == 0 and not conf.experiment.testing_mode:
        pickle.dump(conf, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))
        OmegaConf.save(config=conf, f=open(f'{trainer.checkpoint_path}/{args.session_name}.yaml', 'w'))

    if not conf.experiment.testing_mode:
        net = trainer.train_valid(net, train, valid)
    else:
        net = trainer.test(net, test)
    print(net)

    print(f'FINISHED: {conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_gpu_{rank}')
    time.sleep(10)
    kill(pid, SIGKILL)


def run_single_fold_single_gpu(rank, ngpus_per_node, conf, dataset, collate_fn, net):
    setup_seed(conf.experiment.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    setproctitle.setproctitle(f'{conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_gpu_{rank}')
    session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # WANDB setup /// conf
    reset_wandb_env()
    wandb_init = setup_wandb(conf)
    wandb_init['name'] += f'_{conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}'
    run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
    run.define_metric('train/step'); run.define_metric('train/*', step_metric='train/step')
    run.define_metric('valid/step'); run.define_metric('valid/*', step_metric='valid/step')
    run.define_metric('test/step'); run.define_metric('test/*', step_metric='test/step')
    run.watch(net, log="gradients", log_freq=100)

    torch.cuda.set_device(rank)
    trainer = TrainerBench(conf, rank, run, False)
    print('Batch size', conf.train_params.batch_size)
    B = conf.train_params.batch_size
    dataloaders, train_sampler = [], None

    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[0]]
    train    = DataLoader(dataset, batch_size=B, sampler=samplers[0], collate_fn=collate_fn)
    valid    = DataLoader(dataset, batch_size=B, sampler=samplers[1], collate_fn=collate_fn)
    test     = DataLoader(dataset, batch_size=B, sampler=samplers[2], collate_fn=collate_fn)
    del dataset

    if conf.dev_mode.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 1
    if rank == 0 and not conf.experiment.testing_mode:
        pickle.dump(conf, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))
        OmegaConf.save(config=conf, f=open(f'{trainer.checkpoint_path}/{args.session_name}.yaml', 'w'))

    if not conf.experiment.testing_mode:
        net = trainer.train_valid(net, train, valid)
    else:
        net = trainer.test(net, test)

    print(f'FINISHED: {conf.model_params.model_type}_benchmark_{conf.dataprep.dataset}_gpu_{rank}')

def run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, collate_fn, net):
    torch.multiprocessing.spawn(run_single_fold, 
                                args=(ngpus_per_node, conf, dataset, collate_fn, net), 
                                nprocs=ngpus_per_node, 
                                join=True)
    print("Finished Multiprocessing")


def setup_gpu(conf):
    if torch.cuda.is_available():
        gpu_available = os.environ['CUDA_VISIBLE_DEVICES']
        device = f'cuda: {gpu_available}'
    else:
        device = 'cpu'

    print(f'The current world has {ngpus_per_node} GPUs')
    print(f'Current device is {device}\n')
    
    return conf

if __name__ == "__main__":
    conf = OmegaConf.load(f'sessions_benchmark/{args.session_name}.yaml') 
    print(conf)
    setup_seed(conf.experiment.random_seed)

    print("Setting WANDB Environment.....")
    wandb_init = setup_wandb(conf)

    print("Loading Dataset and Model.....")
    dataset, collate_fn, net = load_dataset_collate_model(conf)
    conf.model_params.output_dim = dataset.output_dim

    if conf.dev_mode.debugging:
        run_debug_mode(conf, dataset, collate_fn, net)
    else:
        conf = setup_gpu(conf)
        if ngpus_per_node > 1:
            run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, collate_fn, net)
        else:
            print("Automatically Not Using Data Distributed Parallel (1 GPU)")
            run_single_fold_single_gpu(0, ngpus_per_node, conf, dataset, collate_fn, net)
