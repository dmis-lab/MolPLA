from typing import Any, Callable, List, Tuple, Union
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW as adamw
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from utils import *
from dataloaders.MolPLA import DatasetMolPLAArms
from dataloaders.MolPLA import collate_arms, collate_arms_nomask
from torch.utils.data import DataLoader
import json
import torch, torch_geometric

from time import sleep
import timeit
import pickle
import faiss

import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score as aup
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index as c_index
import pandas as pd

from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

dataset2learntype = {'geom':          'pre',
                     'zinc':          'pre',
                     'drugbank':      'pre',
                     'pubchem':       'pre',
                     'malaria':       'reg',
                     'lipophilicity': 'reg',
                     'esol':          'reg',
                     'freesolv':      'reg',
                     'toxcast':       'clf',
                     'tox21':         'clf',
                     'sider':         'clf',
                     'muv':           'clf',
                     'hiv':           'clf',
                     'bbbp':          'clf',
                     'bace':          'clf',
                     'clintox':       'clf'}

class Trainer:
    def __init__(self, conf, rank, wandb_run=None, ddp_mode=True):
        self.debug_mode = conf.dev_mode.debugging
        self.ddp_mode   = ddp_mode
        self.rank       = rank
        self.wandb      = wandb_run 
        self.model_type = conf.model_params.model_type
        self.task_name  = conf.dataprep.dataset    
        self.learn_type = dataset2learntype[self.task_name]

        self.optimizer_name  = conf.train_params.optimizer
        self.scheduler_name  = conf.train_params.scheduler

        self.batch_size      = conf.train_params.batch_size
        self.num_epochs      = conf.train_params.num_epochs
        self.num_warmups     = conf.train_params.num_warmups
        self.learning_rate   = conf.train_params.learning_rate
        self.weight_decay    = conf.train_params.weight_decay

        self.early_stopping = conf.train_params.early_stopping
        self.early_patience = conf.train_params.early_patience
        self.best_metric    = conf.experiment.which_best.replace("/","_")

        if self.rank == 0:            
            print("Prediction Model:                        ", self.model_type)
            print("# of Epochs:                             ", self.num_epochs)
            print("Learning Rate:                           ", self.learning_rate)
            print("Weight Decay:                            ", self.weight_decay)

            print("Model Optimizer:                         ", self.optimizer_name)
            print("Learning Rate Scheduler:                 ", self.scheduler_name)

            print("Early Stopping Criteria:                 ", self.early_stopping)
            print("Early Stopping Patience:                 ", self.early_patience)
            print("")

        self.save_name            = conf.wandb.session_name
        self.lookup_values        = dict()
        self.best_valid_metric    = {'loss': np.Inf}
        self.current_valid_metric = {'loss': np.Inf}

    def print0(self, text: str, end=None):
        if self.rank == 0:
            print(text, end=end)

    def reset_lookup_values(self, batch):

        return

    def store_lookup_values(self, batch):

        return

    def wandb_lookup_values(self, batch):

        return

    def get_optimizer_scheduler(self, parameters, num_batches):
        if self.optimizer_name == 'Adam':
            optimizer = optim.Adam(parameters,     lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adadelta':    
            optimizer = optim.Adadelta(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adagrad':    
            optimizer = optim.Adagrad(parameters,  lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':    
            optimizer = optim.AdamW(parameters,    lr=self.learning_rate, weight_decay=self.weight_decay)  
        elif self.optimizer_name == 'Adamax':    
            optimizer = optim.Adamax(parameters,   lr=self.learning_rate, weight_decay=self.weight_decay)  
        elif self.optimizer_name == 'SGD': 
            optimizer = optim.SGD(parameters,      lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError("Invalid Optimizer")

        if self.scheduler_name == 'dummy':
            scheduler = DummyScheduler()
            self.sche_step = 'batch'
        elif self.scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                            optimizer=optimizer, 
                                            eta_min=0, 
                                            last_epoch=-1, 
                                            T_max=self.num_epochs-self.num_warmups)
            self.sche_step = 'epoch'
        elif self.scheduler_name == 'CyclicLR':
            scheduler = optim.lr_scheduler.CyclicLR(
                                            optimizer=optimizer,
                                            base_lr=self.learning_rate,
                                            max_lr=min(self.learning_rate*100,0.001),
                                            cycle_momentum=False if self.optimizer_name != 'SGD' else True,
                                            step_size_up=num_batches//2)
            self.sche_step = 'batch'
        else:
            raise ValueError("Invalid Scheduler")

        return optimizer, scheduler

    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            return optim.Adam(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid Optimizer")

    def get_scheduler(self):
        if self.scheduler_name == 'dummy':
            return DummyScheduler()
        elif self.scheduler_name == 'CyclicLR':
            return optim.lr_scheduler.CyclicLR(
                                            optimizer=self.optimizer,
                                            base_lr=self.learning_rate,
                                            max_lr=min(self.learning_rate*1000,0.001),
                                            cycle_momentum=False)
        else:
            raise ValueError("Invalid Scheduler")

    def check_improvement(self, model):
        CHECK_FLAG = False
        for k, v in self.best_valid_metric.items():
            if k in ['loss', 'reg/mae', 'reg/rmse']:
                if self.best_valid_metric[k] > self.current_valid_metric[k]:
                    self.best_valid_metric[k] = self.current_valid_metric[k]
                    if self.rank == 0:
                        print(f"Improved {k} ====> {self.best_valid_metric[k]:.3f}")
                        torch.save(model.state_dict(), self.checkpoint_path + f'/best_{k.replace("/","_")}.mdl')
                        with open(self.checkpoint_path + f'/best_{k.replace("/","_")}.scores', 'wb') as f:
                            pickle.dump(self.best_valid_metric, f)
                    if k == self.early_stopping: CHECK_FLAG = True
            else:
                if self.best_valid_metric[k] < self.current_valid_metric[k]:
                    self.best_valid_metric[k] = self.current_valid_metric[k]
                    if self.rank == 0:
                        print(f"Improved {k} ====> {self.best_valid_metric[k]:.3f}")
                        torch.save(model.state_dict(), self.checkpoint_path + f'/best_{k.replace("/","_")}.mdl')
                        with open(self.checkpoint_path + f'/best_{k.replace("/","_")}.scores', 'wb') as f:
                            pickle.dump(self.best_valid_metric, f)
                    if k == self.early_stopping: CHECK_FLAG = True
        self.print0("")
        return CHECK_FLAG

class TrainerPretraining(Trainer):
    def __init__(self, conf, rank, wandb_run, ddp_mode, unique_arms=None):
        super(TrainerPretraining, self).__init__(conf, rank, wandb_run, ddp_mode)
        self.only_test_mode      = conf.experiment.testing_mode

        self.args_graph_contrastive = dict(
                                score_function=conf.train_params.pretraining.graph_contrastive.score_func,
                                temperature_scalar=conf.train_params.pretraining.graph_contrastive.tau,
                                rank=self.rank)
        self.args_linker_contrastive = dict(
                                score_function=conf.train_params.pretraining.linker_contrastive.score_func,
                                temperature_scalar=conf.train_params.pretraining.linker_contrastive.tau,
                                rank=self.rank)
        self.args_rgroup_contrastive = dict(
                                score_function=conf.train_params.pretraining.rgroup_contrastive.score_func,
                                temperature_scalar=conf.train_params.pretraining.rgroup_contrastive.tau,
                                rank=self.rank)
        self.negation_linker_constrastive = conf.train_params.pretraining.linker_contrastive.negation
        self.negation_rgroup_constrastive = conf.train_params.pretraining.rgroup_contrastive.negation

        self.checkpoint_path = os.path.join(
                                        conf.path.checkpoint, 
                                        f'pretrained_{conf.dataprep.dataset}_{conf.dataprep.version}')
        if not conf.experiment.testing_mode:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        # Faiss Related Stuff
        self.lookup_vectors   = {'instance_ids': [], 'pred_arms_vector': [], 'true_arms_smiles': []}
        arms_vocab            = DatasetMolPLAArms(conf, unique_arms)
        self.print0(f"Loaded {len(arms_vocab)} R-Groups")
        self.print0("")
        self.arms_loader      = DataLoader(arms_vocab, batch_size=1024, collate_fn=collate_arms)
        self.list_arms_smiles = arms_vocab.data_instances
        self.vector_dimension = conf.model_params.hidden_dim
        self.faiss_gpu        = False

        if conf.model_params.faiss_metric == 'l1':
            self.faiss_metric = faiss.METRIC_L1
        elif conf.model_params.faiss_metric == 'l2':
            self.faiss_metric = faiss.METRIC_L2
        elif conf.model_params.faiss_metric == 'inner_product':
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError("Invalid Faiss Metric")

        # Early Stopping 
        self.best_valid_metric    = {'loss': np.Inf, 'MRR': 0.0, 'Hit@5': 0.0, 'Hit@10': 0.0, 'Hit@20': 0.0,  
                                     'Hit@50': 0.0,  'Hit@100': 0.0, 'Hit@500': 0.0, 'Hit@1000': 0.0}
        self.current_valid_metric = {'loss': np.Inf, 'MRR': 0.0, 'Hit@5': 0.0, 'Hit@10': 0.0, 'Hit@20': 0.0,  
                                     'Hit@50': 0.0,  'Hit@100': 0.0, 'Hit@500': 0.0, 'Hit@1000': 0.0}

    @torch.no_grad()
    def make_vector_library_arms(self, model):
        results = None
        self.print0("Building a Vector Library of R-Groups...")
        if self.ddp_mode: 
            model = model.module.module
        v_list = []
        for batch in self.arms_loader:
            v_list.append(numpify(model.get_rgroup_library(batch, self.rank)))
            # torch.cuda.empty_cache()
        results = np.vstack(v_list)

        return results 

    def reset_lookup_values(self):

        return

    def reset_lookup_vectors(self):
        self.lookup_vectors = {'instance_ids': [], 'pred_arms_vector': [], 'true_arms_smiles': []}

        return

    def store_lookup_values(self, gscores, ascores):

        return

    def store_lookup_vectors(self, batch):
        self.lookup_vectors['instance_ids'].extend(batch['data_instance_id'])
        self.lookup_vectors['pred_arms_vector'].append(numpify(batch['rgroup_contrastive'][0]))
        self.lookup_vectors['true_arms_smiles'].extend(batch['true_arms_smiles'])

        return

    def wandb_lookup_values(self, label, epoch, losses, df_faiss_arms):
        num_ranks = torch.cuda.device_count()
        wandb_dict = {f'{label}/step': epoch}

        rankwise_losses = [None for _ in range(num_ranks)]
        if self.ddp_mode:
            dist.all_gather_object(rankwise_losses, losses)
            rankwise_losses = np.vstack(rankwise_losses).mean(0).tolist()
        else:
            rankwise_losses = losses.reshape(-1).tolist()

        self.print0(f"Loss Report for Epoch #{epoch}")
        self.print0(f"Batchwise Loss for {label} Data Partition")
        self.current_valid_metric['loss'] = 0.
        for idx, loss in enumerate(rankwise_losses):
            self.print0(f"Batchwise Loss Term Index {idx+1}: {loss:.3f}")
            wandb_dict[f'{label}/loss/idx{idx+1}'] = loss
            if label == 'valid':
                self.current_valid_metric['loss'] += loss
        self.print0("")

        if label != 'train':
            rankwise_df_faiss_arms = [None for _ in range(num_ranks)]
            if self.ddp_mode:
                dist.all_gather_object(rankwise_df_faiss_arms, df_faiss_arms)
                rankwise_df_faiss_arms = pd.concat(rankwise_df_faiss_arms, axis=0)
            else:
                rankwise_df_faiss_arms = df_faiss_arms

            if self.rank0:
                for score_col in ['MRR', 'Hit@5', 'Hit@10', 'Hit@20', 'Hit@50', 'Hit@100', 'Hit@500', 'Hit@1000']:
                    wandb_dict[f'{label}/faiss_arms/{score_col}'] = rankwise_df_faiss_arms[score_col].mean()
                    self.current_valid_metric[score_col]          = rankwise_df_faiss_arms[score_col].mean()
                if label == 'test':
                    self.print0("Saving R-Group Retrieval Results as .CSV File...")
                    rankwise_df_faiss_arms.to_csv(self.checkpoint_path + f'/{label}_rgroup_retrieval_results_faiss_{self.task_name}.csv')
            
        if self.rank0 and self.wandb is not None:
            self.wandb.log(wandb_dict)

        return

    def calculate_losses(self, batch):
        total_loss = []
        graph_contrastive  = ContrastiveLoss(**self.args_graph_contrastive)
        linker_contrastive = ContrastiveLoss(**self.args_linker_contrastive)
        rgroup_contrastive = ContrastiveLoss(**self.args_rgroup_contrastive)

        total_loss.append(graph_contrastive(*batch['graph_contrastive']))
        coef = 0.1 if not self.negation_linker_constrastive else 0.0
        total_loss.append(coef*linker_contrastive(*batch['linker_contrastive']))
        coef = 1.0 if not self.negation_rgroup_constrastive else 0.0
        total_loss.append(coef*rgroup_contrastive(*batch['rgroup_contrastive']))

        return total_loss

    @torch.no_grad()
    def invoke_faiss_molpla_arms(self, model):
        vlib_arms = self.make_vector_library_arms(model)
        pred_arms = np.vstack(self.lookup_vectors['pred_arms_vector'])
        true_arms = self.lookup_vectors['true_arms_smiles']
        data_ids  = self.lookup_vectors['instance_ids']
        if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vlib_arms)
            faiss.normalize_L2(pred_arms)

        res = faiss.StandardGpuResources()
        self.print0("Invoking FAISS-GPU")
        if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(self.vector_dimension)  # the other index
        else:
            index = faiss.IndexFlatL2(self.vector_dimension)
        index = faiss.index_cpu_to_gpu(res, self.rank, index)
        index.add(vlib_arms)

        N = min(1000, len(self.list_arms_smiles)) 
        D, I = index.search(pred_arms, N)
        def get_smiles(x):
            return self.list_arms_smiles[x] if x != -1 else '...'
        get_smiles_vectorized = np.vectorize(get_smiles)

        df = pd.DataFrame(
            get_smiles_vectorized(I), 
            index=data_ids, 
            columns=[f'Rank #{i+1}' for i in range(N)])
        df['Actual Molecular Arms'] = true_arms

        def get_reciprocal_rank(z):
            X = z[:N]
            Y = z[N]
            rr_list = [0.]

            try:
                return 1 / (X.tolist().index(Y)+1)
            except:
                return 0.

        def get_mean_tanimoto_similarity(x, *args):
            K = args[0]
            target = x[N]
            candid = x[:K].tolist()
            from rdkit import DataStructs
            from rdkit import Chem
            from rdkit.Chem import AllChem
            fpgen       = AllChem.GetRDKitFPGenerator()
            target_mol  = Chem.MolFromSmiles(target.strip('.'),sanitize=False)
            target_fp   = fpgen.GetFingerprint(target_mol)
            candid_mols = [Chem.MolFromSmiles(c.strip('.'),sanitize=False) for c in candid]
            candid_fps  = [fpgen.GetFingerprint(m) for m in candid_mols]
            tanimoto_sims = [DataStructs.TanimotoSimilarity(target_mol, cfp) for cfp in candid_fps]

            return np.mean(tanimoto_sims)

        def get_hit(x, *args):
            K = args[0]
            target = x[N]
            candid = x[:K].tolist()

            return 1 if target in candid else 0

        mrr = np.apply_along_axis(get_reciprocal_rank, 1, df.values).reshape(-1,1)
        score_columns = ['MRR']
        score_data = [mrr]
        for K in [5, 10, 20, 50, 100, 500, 1000]:
            score_data.append(np.apply_along_axis(get_hit, 1, df.values, K).reshape(-1,1))
            score_columns.append(f'Hit@{K}')

        score_data = np.hstack(score_data)
        score_df   = pd.DataFrame(score_data, index=data_ids, columns=score_columns)

        return pd.concat([df, score_df], axis=1)

    def train_step(self, model, data, epoch):
        model.train()
        if self.ddp_mode:
            torch.distributed.barrier()
        batchwise_loss = [] 
        start = timeit.default_timer()
        #
        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)
            loss = self.calculate_losses(batch)
            sum(loss).backward()
            self.optimizer.step()
            if self.num_warmups < epoch+1 and self.sche_step == 'batch':
                self.scheduler.step()
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_vectors(batch)
        if self.num_warmups < epoch+1 and self.sche_step == 'epoch':
            self.scheduler.step()
        #
        stop  = timeit.default_timer()
        delta = str((stop - start)//60)
        self.print0(f"Training Time for Epoch {epoch} is {delta} minutes")
        df_faiss_arms = None

        return np.array(batchwise_loss).mean(0), model, df_faiss_arms

    @torch.no_grad()
    def eval_step(self, model, data):
        model.eval()
        batchwise_loss = []

        for idx, batch in enumerate(data):
            batch = model(batch)            
            loss  = self.calculate_losses(batch)
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_vectors(batch)

        df_faiss_arms = self.invoke_faiss_molpla_arms(model)

        return np.array(batchwise_loss).mean(0), model, df_faiss_arms

    def train_valid_test(self, model, train_loader, valid_loader, test_loader):
        self.rank0 = self.rank == 0
        num_ranks = torch.cuda.device_count()
        print(f"RANK: {self.rank+1} | Training Batches: {len(train_loader)}, Validation Batches: {len(valid_loader)}")
        num_batches = len(train_loader)
        EARLY_STOPPING = False

        model = model.to(self.rank)
        model = DDP(model, device_ids=[self.rank]) if self.ddp_mode else model
        model = FSDP(model) if self.ddp_mode else model

        if not self.only_test_mode:
            self.optimizer, self.scheduler = self.get_optimizer_scheduler(model.parameters(), num_batches)

            for epoch in range(self.num_epochs):
                self.print0(f"Training at Epoch #{epoch}")
                # Train Step
                train_loss, model, df_faiss_arms = self.train_step(model, train_loader, epoch)
                self.wandb_lookup_values('train', epoch, train_loss, df_faiss_arms) 
                self.reset_lookup_vectors()

                # Validation Step
                valid_loss, _, df_faiss_arms = self.eval_step(model, valid_loader)
                self.wandb_lookup_values('valid', epoch, valid_loss, df_faiss_arms)
                self.reset_lookup_vectors()

                if not self.check_improvement(model):
                    if self.early_patience > 0: self.early_patience -= 1
                    else: EARLY_STOPPING = True
                if EARLY_STOPPING: break

                if self.rank0: 
                    self.print0("Saving Current Epoch Checkpoint...")
                    torch.save(model.state_dict(), self.checkpoint_path + '/{:03}_epoch.mdl'.format(epoch))
                    torch.save(model.state_dict(), self.checkpoint_path + '/last_epoch.mdl')
            
            if self.ddp_mode:
                torch.distributed.barrier()
        
        # Testing Step
        print(f"RANK: {self.rank+1} | Test Batches: {len(test_loader)}")
        if 'geom' not in self.checkpoint_path:
            self.checkpoint_path = self.checkpoint_path.replace('drugbank', 'geom')
        model.load_state_dict(torch.load(self.checkpoint_path + f'/best_{self.best_metric}.mdl'))
        test_loss, _, df_faiss_arms = self.eval_step(model, test_loader)
        self.wandb_lookup_values('test', 0, test_loss, df_faiss_arms)
        self.reset_lookup_vectors()

        if self.rank0: 
            self.print0(f"RANK: {self.rank+1} | Saving Vector Library of R-Groups...")
            rgroup_library_path = f'{self.checkpoint_path}/rgroup_library.npy'
            rgroup_vocab_path   = f'{self.checkpoint_path}/rgroup.vocab'
            np.save(rgroup_library_path, self.make_vector_library_arms(model))
            pickle.dump(self.list_arms_smiles, open(rgroup_vocab_path, 'wb'))

        if self.ddp_mode:
            torch.distributed.barrier()

        if self.rank0:
            self.wandb.finish()

        return model


class TrainerBench(Trainer):
    def __init__(self, conf, rank, wandb_run, ddp_mode):
        super(TrainerBench, self).__init__(conf, rank, wandb_run, ddp_mode)
        self.checkpoint_path = os.path.join(
                            conf.path.checkpoint, 
                            f'benchmark_{conf.dataprep.dataset}_{conf.dataprep.version}_fold{conf.experiment.fold_num}')
        if not conf.experiment.testing_mode:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.output_dim    = conf.model_params.output_dim

        if self.learn_type == 'reg':
            self.lookup_values = {'true_reg':[],'pred_reg':[]}
        elif self.learn_type == 'clf':
            self.lookup_values = dict()
            for i in range(self.output_dim):
                self.lookup_values[f'true_clf_{i+1}'] = []
                self.lookup_values[f'pred_clf_{i+1}'] = []
        else:
            raise

        self.best_valid_metric = {
            'reg/mae':       np.Inf,
            'reg/rmse':      np.Inf,
            'reg/pearson':  -np.Inf,
            'reg/spearman': -np.Inf,
            'reg/ci':       -np.Inf,
            'reg/r2':       -np.Inf,
            'clf/auroc':    -np.Inf,
            'clf/auprc':    -np.Inf,
            'clf/f1score':  -np.Inf,
            'clf/accuracy': -np.Inf,
            'loss':          np.Inf}

        self.current_valid_metric = {
            'reg/mae':       np.Inf,
            'reg/rmse':      np.Inf,
            'reg/pearson':  -np.Inf,
            'reg/spearman': -np.Inf,
            'reg/ci':       -np.Inf,
            'reg/r2':       -np.Inf,
            'clf/auroc':    -np.Inf,
            'clf/auprc':    -np.Inf,
            'clf/f1score':  -np.Inf,
            'clf/accuracy': -np.Inf,
            'loss':          np.Inf}

    def reset_lookup_values(self):
        if self.learn_type == 'reg':
            self.lookup_values = {'true_reg':[],'pred_reg':[]}
        elif self.learn_type == 'clf':
            self.lookup_values = dict()
            for i in range(self.output_dim):
                self.lookup_values[f'true_clf_{i+1}'] = []
                self.lookup_values[f'pred_clf_{i+1}'] = []
        else:
            raise

    def store_lookup_values(self, batch):
        if self.learn_type == 'reg':
            self.lookup_values['true_reg'].extend(numpify(batch[f'{self.task_name}/true'].view(-1)))
            self.lookup_values['pred_reg'].extend(numpify(batch[f'{self.task_name}/pred'].view(-1)))
        else: # clf            
            true_clf = numpify(batch[f'{self.task_name}/true'])
            pred_clf = numpify(batch[f'{self.task_name}/pred'])
            if self.output_dim > 1:
                for i in range(self.output_dim):
                    self.lookup_values[f'true_clf_{i+1}'].extend(true_clf[:,i])
                    self.lookup_values[f'pred_clf_{i+1}'].extend(pred_clf[:,i])
            else:
                self.lookup_values[f'true_clf_1'].extend(true_clf.reshape(-1))
                self.lookup_values[f'pred_clf_1'].extend(pred_clf.reshape(-1))


    def wandb_lookup_values(self, label, epoch, losses):
        num_ranks = torch.cuda.device_count()
        wandb_dict = {f'{label}/step': epoch}

        rankwise_losses = [None for _ in range(num_ranks)]
        if self.ddp_mode:
            dist.all_gather_object(rankwise_losses, losses)
            rankwise_losses = np.vstack(rankwise_losses).mean(0).tolist()
        else:
            rankwise_losses = losses.reshape(-1).tolist()

        self.print0(f"Loss Report for Epoch #{epoch}")
        self.print0(f"Batchwise Loss for {label} Data Partition")
        self.current_valid_metric['loss'] = 0.
        for idx, loss in enumerate(rankwise_losses):
            self.print0(f"Batchwise Loss Term Index {idx+1}: {loss:.3f}")
            wandb_dict[f'{label}/loss/idx{idx+1}'] = loss
            if label == 'valid':
                self.current_valid_metric['loss'] += loss
        self.print0("")

        if self.learn_type == 'clf':
            auc_list, aup_list, f1_list, acc_list = [], [], [], []
            for i in range(self.output_dim):
                y, yhat = [None for _ in range(num_ranks)], [None for _ in range(num_ranks)]
                if self.ddp_mode:
                    dist.all_gather_object(y,    self.lookup_values[f'true_clf_{i+1}'])
                    dist.all_gather_object(yhat, self.lookup_values[f'pred_clf_{i+1}'])
                    y, yhat = np.array(sum(y, [])), np.array(sum(yhat, []))
                else:
                    y       = np.array(self.lookup_values[f'true_clf_{i+1}'])
                    yhat    = np.array(self.lookup_values[f'pred_clf_{i+1}'])

                isvalid = y**2>0.
                if (y==1).sum() > 0 and (y==-1).sum() > 0:
                    auc_list.append(auc((y[isvalid]+1)/2,yhat[isvalid]))
                    aup_list.append(aup((y[isvalid]+1)/2,yhat[isvalid]))
                    yhat =  1/(1 + np.exp(-yhat))
                    yhat = (yhat > 0.5).reshape(-1) 
                    f1_list.append(f1((y[isvalid]+1)/2,yhat[isvalid]))
                    acc_list.append(acc((y[isvalid]+1)/2,yhat[isvalid]))
                else:
                    self.print0("All ground truth labels are equivalent, skipping metric calculation...")
                    self.print0("")

            wandb_dict[f'{label}/clf/auroc']    = np.mean(auc_list)
            wandb_dict[f'{label}/clf/auprc']    = np.mean(aup_list)
            wandb_dict[f'{label}/clf/f1score']  = np.mean(f1_list)
            wandb_dict[f'{label}/clf/accuracy'] = np.mean(acc_list)

        elif self.learn_type == 'reg':
            y, yhat = [None for _ in range(num_ranks)], [None for _ in range(num_ranks)]
            if self.ddp_mode:
                dist.all_gather_object(y,    self.lookup_values[f'true_reg'])
                dist.all_gather_object(yhat, self.lookup_values[f'pred_reg'])
                y, yhat = np.hstack(y), np.hstack(yhat)
            else:
                y       = np.array(self.lookup_values[f'true_reg'])
                yhat    = np.array(self.lookup_values[f'pred_reg'])


            wandb_dict[f'{label}/reg/mae']      = mae(y,yhat)
            wandb_dict[f'{label}/reg/rmse']     = mse(y,yhat) ** .5
            pearson = pearsonr(y,yhat)[0]
            spearman = spearmanr(y,yhat)[0]
            if np.isnan(pearson): pearson = 0
            if np.isnan(spearman): spearman = 0  
            wandb_dict[f'{label}/reg/pearson']  = pearson
            wandb_dict[f'{label}/reg/spearman'] = spearman
            wandb_dict[f'{label}/reg/ci']       = c_index(y,yhat)
            wandb_dict[f'{label}/reg/r2']       = r2(y,yhat)

        else:
            raise

        if label == 'valid':
            for k,v in wandb_dict.items():
                self.current_valid_metric[k.split(label+'/')[1]] = v

        if self.rank == 0: self.wandb.log(wandb_dict)

    def calculate_losses(self, batch):
        total_loss = []
        reg_criterion = RegressionLoss(rank=self.rank,task=self.task_name)
        clf_criterion = ClassificationLoss(rank=self.rank,task=self.task_name)
        mlf_criterion = MultiClassificationLoss(rank=self.rank,task=self.task_name)
        if self.learn_type == 'reg':
            total_loss.append(reg_criterion(batch))
        else:
            if self.output_dim > 1: total_loss.append(mlf_criterion(batch))
            else:                   total_loss.append(clf_criterion(batch))

        return total_loss 

    def train_step(self, model, data, epoch):
        model.train()
        if self.ddp_mode:
            torch.distributed.barrier()
        batchwise_loss = [] 

        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)
            loss = self.calculate_losses(batch)
            sum(loss).backward()
            self.optimizer.step()
            if self.num_warmups < epoch+1 and self.sche_step == 'batch':
                self.scheduler.step()
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_values(batch)
        if self.num_warmups < epoch+1 and self.sche_step == 'epoch':
            self.scheduler.step()

        return np.array(batchwise_loss).mean(0), model

    @torch.no_grad()
    def eval_step(self, model, data):
        model.eval()
        batchwise_loss = []

        for idx, batch in enumerate(data):
            batch = model(batch)         
            loss = self.calculate_losses(batch)
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_values(batch)

        return np.array(batchwise_loss).mean(0), model

    def train_valid(self, model, train_loader, valid_loader):
        num_ranks = torch.cuda.device_count()
        print(f"RANK: {self.rank+1} | Training Batches: {len(train_loader)}, Validation Batches: {len(valid_loader)}")
        EARLY_STOPPING = False
        num_batches = len(train_loader)

        model = model.to(self.rank)
        if self.ddp_mode: model = DDP(model, device_ids=[self.rank])
        # if self.ddp_mode: model = FSDP(model) 
        self.optimizer, self.scheduler = self.get_optimizer_scheduler(model.parameters(), num_batches)

        for epoch in range(self.num_epochs):
            train_loss, model = self.train_step(model, train_loader, epoch)
            self.wandb_lookup_values('train', epoch, train_loss) 
            self.reset_lookup_values()

            valid_loss, _ = self.eval_step(model, valid_loader)
            self.wandb_lookup_values('valid', epoch, valid_loss)
            self.reset_lookup_values()

            if not self.check_improvement(model):
                if self.early_patience > 0: self.early_patience -= 1
                else: EARLY_STOPPING = True
            if EARLY_STOPPING: break

        if self.rank == 0: 
            torch.save(model.state_dict(), self.checkpoint_path + '/last_epoch.mdl')
            self.wandb.finish()

        return model

    @torch.no_grad()
    def test(self, model, test):
        print(f"RANK: {self.rank} | Test Batches: {len(test)}")
        model = model.to(self.rank)
        if self.ddp_mode: model = DDP(model, device_ids=[self.rank])
        # if self.ddp_mode: model = FSDP(model) 

        # self.best_metric = 'loss'
        best_model_path = f'{self.checkpoint_path}/best_{self.best_metric}.mdl'
        print("Loading Best Trained Model for Final Testing: ", best_model_path)
        chkpt = torch.load(best_model_path, map_location=f"cuda:{self.rank}") # need change!
        model.load_state_dict(chkpt)

        print("Testing Model on RANK: ", self.rank)
        eval_loss, _ = self.eval_step(model, test)
        self.wandb_lookup_values('test', 0, eval_loss)
        self.reset_lookup_values()

        if self.rank == 0: 
            self.wandb.finish()

        return model
