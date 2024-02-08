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

import torch, torch_geometric

from os import kill
from os import getpid
from signal import SIGKILL
import time
from trainer import *
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
torch.multiprocessing.set_sharing_strategy('file_system')

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
    if 'molpla' in conf.model_params.model_type:
        from dataloaders.MolPLA import DatasetPretraining
        from models.MolPLA import Net
    else:
        raise ValueError("Invalid Model Type")

    dataset          = DatasetPretraining(conf)
    net              = Net(conf)
    num_params       = sum(p.numel() for p in net.parameters())
    num_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of All Model Parameters in [{conf.model_params.model_type}]: {num_params}")
    print(f"Number of Trainable Parameters in [{conf.model_params.model_type}]: {num_params_train}")

    if conf.dataprep.dataset == 'geom':
        dataset.make_random_splits_single()
    else:
        dataset.make_zeroshot_testing()

    return dataset, net

def run_debug_mode(conf, dataset, net):
    setup_seed(conf.experiment.random_seed)
    setproctitle.setproctitle(f'{conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}_debug')

    session_name = 'testproject_testgroup_testsession'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # Distributed DataLoaders
    ddp_batch_size = int(conf.train_params.batch_size/ngpus_per_node)
    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[conf.experiment.fold_num-1]]

    train    = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[0])
    valid    = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[1])
    test     = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[2])

    rank = 0
    trainer = TrainerPretraining(conf, rank, None, False, dataset.unique_arms)

    # train, valid, test and save the model
    trained_model, train_loss, valid_loss = trainer.train_valid_test(net, train, valid, test)
    if rank == 0: print('Finish Debugging Mode')

def run_single_fold(rank, ngpus_per_node, conf, dataset, net):
    setup_seed(conf.experiment.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    setproctitle.setproctitle(f'{conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}_gpu_{rank}')
    session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # WANDB setup /// conf
    if rank == 0:
        reset_wandb_env()
        wandb_init = setup_wandb(conf)
        wandb_init['name'] += f'_{conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}'
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

    trainer = TrainerPretraining(conf, rank, run, True, dataset.unique_arms)
    ddp_batch_size = conf.train_params.batch_size // ngpus_per_node
    if rank == 0:
        print('Batch size', conf.train_params.batch_size)
        print('Distributed batch size', ddp_batch_size)
    dataloaders, train_sampler = [], None

    for idx, indices in enumerate(dataset.kfold_splits[0]):
        sampler = DistributedSampler(Subset(dataset,indices), shuffle=idx == 0)
        loader  = DataLoader(Subset(dataset,indices), batch_size=ddp_batch_size, 
                                                      sampler=sampler, 
                                                      pin_memory=True,
                                                      num_workers=16//ngpus_per_node)
        dataloaders.append(loader)
        if idx == 0: train_sampler = sampler
    del dataset
    train, valid, test = dataloaders

    if conf.dev_mode.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 1
    if rank == 0 and not conf.experiment.testing_mode:
        pickle.dump(conf, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))
        OmegaConf.save(config=conf, f=open(f'{trainer.checkpoint_path}/{args.session_name}.yaml', 'w'))

    net = trainer.train_valid_test(net, train, valid, test)
    print(net)

    print(f'FINISHED: {conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}_gpu_{rank}')
    time.sleep(10)
    # kill(pid, SIGKILL)

def run_single_fold_single_gpu(rank, ngpus_per_node, conf, dataset, net):
    setup_seed(conf.experiment.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    setproctitle.setproctitle(f'{conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}_gpu_{rank}')
    session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # WANDB setup /// conf
    reset_wandb_env()
    wandb_init = setup_wandb(conf)
    wandb_init['name'] += f'_{conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}'
    run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
    run.define_metric('train/step'); run.define_metric('train/*', step_metric='train/step')
    run.define_metric('valid/step'); run.define_metric('valid/*', step_metric='valid/step')
    run.define_metric('test/step'); run.define_metric('test/*', step_metric='test/step')
    run.watch(net, log="gradients", log_freq=10)

    torch.cuda.set_device(rank)
    trainer = TrainerPretraining(conf, rank, run, False, dataset.unique_arms)
    print('Batch size', conf.train_params.batch_size)
    B = conf.train_params.batch_size
    dataloaders, train_sampler = [], None

    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[conf.experiment.fold_num-1]]
    train    = DataLoader(dataset, batch_size=B, sampler=samplers[0], pin_memory=True, num_workers=16)
    valid    = DataLoader(dataset, batch_size=B, sampler=samplers[1])
    test     = DataLoader(dataset, batch_size=B, sampler=samplers[2])
    del dataset

    if conf.dev_mode.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 1
    if rank == 0 and not conf.experiment.testing_mode:
        pickle.dump(conf, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))
        OmegaConf.save(config=conf, f=open(f'{trainer.checkpoint_path}/{args.session_name}.yaml', 'w'))

    net = trainer.train_valid_test(net, train, valid, test)
    print(net)

    print(f'FINISHED: {conf.model_params.model_type}_pretrain_{conf.dataprep.dataset}_gpu_{rank}')

def run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, net):
    torch.multiprocessing.spawn(run_single_fold, 
                                args=(ngpus_per_node, conf, dataset, net), 
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
    conf = OmegaConf.load(f'sessions_pretraining/{args.session_name}.yaml') 
    print(conf)
    setup_seed(conf.experiment.random_seed)

    print("Setting WANDB Environment.....")
    wandb_init = setup_wandb(conf)

    print("Loading Dataset and Model.....")
    dataset, net = load_dataset_collate_model(conf)

    if conf.dev_mode.debugging:
        run_debug_mode(conf, dataset, net)
    else:
        conf = setup_gpu(conf)
        if ngpus_per_node > 1:
            run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, net)
        else:
            print("Automatically Not Using Data Distributed Parallel (1 GPU)")
            run_single_fold_single_gpu(0, ngpus_per_node, conf, dataset, net)