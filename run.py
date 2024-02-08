import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
from omegaconf import OmegaConf
from datetime import datetime
from multiprocessing import Pool
now = datetime.now()

fold2seed   = {1: 961104, 2: 990220, 3: 940107, 4: 940110, 5: 921222}
task2metric = {
        'tox21':         'clf/auroc',
        'bace':          'clf/auroc',
        'bbbp':          'clf/auroc',
        'sider':         'clf/auroc',
        'freesolv':      'reg/rmse',
        'lipophilicity': 'reg/rmse',
        'esol':          'reg/rmse',
        'toxcast':       'clf/auroc',
        'clintox':       'clf/auroc',
        # 'hiv':           'clf/auroc',
        # 'muv':           'clf/auroc'
    }

parser = argparse.ArgumentParser()
parser.add_argument('--session_name',        '-sn', default='development', type=str)
parser.add_argument('--debug_mode',          '-dm', default=False,   action='store_true')
parser.add_argument('--toy_test',            '-tt', default=False,   action='store_true')
parser.add_argument('--multi_gpu',           '-mg', default='0,1',   type=str)
parser.add_argument('--bench_gpu',           '-bg', default='2',     type=str)
parser.add_argument('--multi_fold',          '-mf', default=1,       type=int)
parser.add_argument('--start_fold',          '-sf', default=1,       type=int)
parser.add_argument('--end_fold',            '-ef', default=5,       type=int)
# parser.add_argument('--testing_mode',     '-tm', default=False,   action='store_true')
parser.add_argument('--port_offset',         '-po', default=23,      type=int)
parser.add_argument('--skip_pretrain',       '-sp', default=False,   action='store_true')
parser.add_argument('--skip_benchmark',      '-sb', default=False,   action='store_true')
parser.add_argument('--skip_benchtrain',     '-sh', default=False,   action='store_true')
parser.add_argument('--force_pretrain_test', '-fp', default=False,   action='store_true')

# parser.add_argument('--skip_ablation',    '-sa', default=False,   action='store_true')])

args = parser.parse_args()
original_session_name = args.session_name
SCRIPT_LINE_PRETRAIN = f'CUDA_VISIBLE_DEVICES={args.multi_gpu} python -W ignore src/pretrain_ddp.py'
conf = OmegaConf.load('./settings.yaml')[args.session_name]
OmegaConf.save(config=conf, f=open(f'sessions_pretraining/{args.session_name}.yaml', 'w'))
if args.toy_test:
    conf.dev_mode.toy_test = True
    args.session_name += '_toytest'
    OmegaConf.save(config=conf, f=open(f'sessions_pretraining/{args.session_name}.yaml', 'w'))
    args.end_fold = 1

SCRIPT_LINE_BENCHMARK = f'CUDA_VISIBLE_DEVICES={args.bench_gpu} python -W ignore src/benchmark_ddp.py'
conf_bench = OmegaConf.load('./settings.yaml')[original_session_name+'_bench']
#
conf.train_params              = conf_bench.dataprep
conf.experiment                = conf_bench.experiment
conf.train_params              = conf_bench.train_params
conf.model_params.dropout_rate = conf_bench.model_params.dropout_rate
#
OmegaConf.save(config=conf, f=open(f'sessions_benchmark/{args.session_name}.yaml', 'w'))
if args.toy_test:
    conf.dev_mode.toy_test = True
    OmegaConf.save(config=conf, f=open(f'sessions_benchmark/{args.session_name}.yaml', 'w'))
if args.debug_mode:
    conf.dev_mode.debugging = True
    OmegaConf.save(config=conf, f=open(f'sessions_benchmark/{args.session_name}.yaml', 'w'))


def run_process_pretrain(fold_num, port_offset, setting='main'):
    conf                          = OmegaConf.load(f'sessions_pretraining/{args.session_name}.yaml')
    conf.experiment.fold_num      = fold_num
    conf.ddp.port                += fold_num
    conf.wandb.session_name      += f'_{setting}'
    conf.model_params.model_type += f'_{setting}'
    if not args.force_pretrain_test:
        omega_path                    = f'sessions_pretraining/{args.session_name}_{setting}_{fold_num}.yaml'
        OmegaConf.save(config=conf, f=open(omega_path, 'w'))
        os.system(f'{SCRIPT_LINE_PRETRAIN} -sn {args.session_name}_{setting}_{fold_num}')

    conf.experiment.testing_mode  = True
    omega_path                    = f'sessions_pretraining/{args.session_name}_{setting}_{fold_num}_test.yaml'
    OmegaConf.save(config=conf, f=open(omega_path, 'w'))
    os.system(f'{SCRIPT_LINE_PRETRAIN} -sn {args.session_name}_{setting}_{fold_num}_test')

    conf.dataprep.dataset         = 'drugbank'
    omega_path                    = f'sessions_pretraining/{args.session_name}_{setting}_{fold_num}_test_external.yaml'
    OmegaConf.save(config=conf, f=open(omega_path, 'w'))
    os.system(f'{SCRIPT_LINE_PRETRAIN} -sn {args.session_name}_{setting}_{fold_num}_test_external')

    return fold_num


def run_process_benchmark(fold_num, port_offset, benchmark_dataset, setting='main'):
    conf = OmegaConf.load(f'sessions_benchmark/{args.session_name}.yaml')
    conf.experiment.fold_num         = fold_num
    conf.experiment.random_seed      = fold2seed[fold_num]
    conf.train_params.early_stopping = task2metric[benchmark_dataset]
    conf.experiment.which_best       = task2metric[benchmark_dataset]
    conf.ddp.port                   += fold_num
    conf.dataprep.dataset            = benchmark_dataset
    conf.wandb.session_name         += f'_{setting}'
    conf.model_params.model_type    += f'_{setting}'
    omega_path                       = f'sessions_benchmark/{args.session_name}_{setting}_{fold_num}.yaml'
    OmegaConf.save(config=conf, f=open(omega_path, 'w'))
    os.system(f'{SCRIPT_LINE_BENCHMARK} -sn {args.session_name}_{setting}_{fold_num}')

    return fold_num


def run_process_benchmark_test(fold_num, port_offset, benchmark_dataset, setting='main'):
    conf = OmegaConf.load(f'sessions_benchmark/{args.session_name}.yaml')
    conf.experiment.fold_num      = fold_num
    conf.experiment.random_seed   = fold2seed[fold_num]
    conf.experiment.which_best    = task2metric[benchmark_dataset]
    conf.ddp.port                += fold_num
    conf.dataprep.dataset         = benchmark_dataset
    conf.experiment.testing_mode  = True
    conf.wandb.session_name      += f'_{setting}'
    conf.model_params.model_type += f'_{setting}'
    omega_path                    = f'sessions_benchmark/{args.session_name}_{setting}_testmode_{fold_num}.yaml'
    OmegaConf.save(config=conf, f=open(omega_path, 'w'))
    os.system(f'{SCRIPT_LINE_BENCHMARK} -sn {args.session_name}_{setting}_testmode_{fold_num}')

    return fold_num


def multiprocess(benchmark_dataset, setting='main'):
    if not args.skip_benchtrain:
        print("")
        print(f"######################## TRAINING ON BENCHMARK DATASET [{benchmark_dataset}]")
        print("")
        pool = Pool(args.multi_fold)
        all_folds = [*range(args.start_fold, args.end_fold+1)]
        run_folds_list = [all_folds[start_fold:(start_fold+args.end_fold)]
                          for start_fold in range(0, args.end_fold, args.end_fold)]
        fold_results_list = []
        for fold in run_folds_list:
            print('Dataset Fold Index: ', fold)
            args_list = [(fold_idx, args.port_offset+100, benchmark_dataset, setting) for fold_idx in fold]
            fold_results_list.extend(pool.starmap(run_process_benchmark, args_list))
        pool.close()
        pool.join()

    print("")
    print(f"######################## TESTING ON BENCHMARK DATASET [{benchmark_dataset}]")
    print("")
    pool = Pool(args.multi_fold)
    all_folds = [*range(args.start_fold, args.end_fold+1)]
    run_folds_list = [all_folds[start_fold:(start_fold+args.end_fold)]
                      for start_fold in range(0, args.end_fold, args.end_fold)]
    fold_results_list = []
    for fold in run_folds_list:
        print('Dataset Fold Index: ', fold)
        args_list = [(fold_idx, args.port_offset, benchmark_dataset, setting) for fold_idx in fold]
        fold_results_list.extend(pool.starmap(run_process_benchmark_test, args_list))
    pool.close()
    pool.join()


if __name__ == "__main__":
    if not args.skip_pretrain:
        print("")
        print(f"######################## PRETRAINING ON LARGE-SCALE DATASET")
        print("")
        run_process_pretrain(1, 0)
    if not args.skip_benchmark:
        for dataset in task2metric.keys():
            multiprocess(dataset)