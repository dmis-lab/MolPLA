import pickle
import pandas as pd
import numpy as np
from rdkit import Chem 
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS, Recap
from torch_geometric.data.collate import collate
from rdkit.Chem.rdmolops import FastFindRings
RDLogger.DisableLog('rdApp.*')
import os
import sys
import torch
from torch.utils.data import Dataset 
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
import time
import itertools
from collections import defaultdict
import math
from rdkit.Chem.Scaffolds.MurckoScaffold import *
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.BRICS import BRICSDecompose
from functools import reduce
import random
import torch.nn.functional as F
from copy import deepcopy, copy
from collections import namedtuple, Counter
from datetime import datetime
import pdb
import gc
from tqdm import tqdm

# from torch_geometric.data import Data, InMemoryDataset, Dataset
from typing import Callable, Optional, Tuple, List, Union, Any, Literal

from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import add_self_loops, scatter
from torch_geometric.utils import from_networkx
import networkx as nx

from itertools import compress

from thermo import functional_groups as fgs
import multiprocessing as mp


def linker_to_boolean(gdata):
    gdata.is_linker      = gdata.is_linker.bool()
    gdata.edge_is_linker = gdata.edge_is_linker.bool()

    return gdata
  
def get_molecular_functional_groups(mol):
    binary_vector = [
        fgs.is_organic(mol),
        fgs.is_inorganic(mol),
        False, # fgs.is_radionuclide(mol), 

        fgs.is_hydrocarbon(mol),
        fgs.is_alkane(mol),
        fgs.is_cycloalkane(mol),
        fgs.is_branched_alkane(mol),
        fgs.is_alkene(mol),
        fgs.is_alkyne(mol),
        fgs.is_aromatic(mol),

        fgs.is_alcohol(mol),
        fgs.is_polyol(mol),
        fgs.is_ketone(mol),
        fgs.is_aldehyde(mol),
        fgs.is_carboxylic_acid(mol),
        fgs.is_ether(mol),
        fgs.is_phenol(mol),
        fgs.is_ester(mol),
        fgs.is_anhydride(mol),
        fgs.is_acyl_halide(mol),
        fgs.is_carbonate(mol),
        fgs.is_carboxylate(mol),
        fgs.is_hydroperoxide(mol),
        fgs.is_peroxide(mol),
        fgs.is_orthoester(mol),
        fgs.is_methylenedioxy(mol),
        fgs.is_orthocarbonate_ester(mol),
        fgs.is_carboxylic_anhydride(mol),

        fgs.is_amide(mol),
        fgs.is_amidine(mol),
        fgs.is_amine(mol),
        fgs.is_primary_amine(mol),
        fgs.is_secondary_amine(mol),
        fgs.is_tertiary_amine(mol),
        fgs.is_quat(mol),
        fgs.is_imine(mol),
        fgs.is_primary_ketimine(mol),
        fgs.is_secondary_ketimine(mol),
        fgs.is_primary_aldimine(mol),
        fgs.is_secondary_aldimine(mol),
        fgs.is_imide(mol),
        fgs.is_azide(mol),
        fgs.is_azo(mol),
        fgs.is_cyanate(mol),
        fgs.is_isocyanate(mol),
        fgs.is_nitrate(mol),
        fgs.is_nitrile(mol),
        fgs.is_isonitrile(mol),
        fgs.is_nitrite(mol),
        fgs.is_nitro(mol),
        fgs.is_nitroso(mol),
        fgs.is_oxime(mol),
        fgs.is_pyridyl(mol),
        fgs.is_carbamate(mol),
        False, # fgs.is_cyanide(mol),

        fgs.is_mercaptan(mol),
        fgs.is_sulfide(mol),
        fgs.is_disulfide(mol),
        fgs.is_sulfoxide(mol),
        fgs.is_sulfone(mol),
        fgs.is_sulfinic_acid(mol),
        fgs.is_sulfonic_acid(mol),
        fgs.is_sulfonate_ester(mol),
        fgs.is_thiocyanate(mol),
        fgs.is_isothiocyanate(mol),
        fgs.is_thioketone(mol),
        fgs.is_thial(mol),
        fgs.is_carbothioic_s_acid(mol),
        fgs.is_carbothioic_o_acid(mol),
        fgs.is_thiolester(mol),
        fgs.is_thionoester(mol),
        fgs.is_carbodithioic_acid(mol),
        fgs.is_carbodithio(mol),

        fgs.is_siloxane(mol),
        fgs.is_silyl_ether(mol),

        fgs.is_boronic_acid(mol),
        fgs.is_boronic_ester(mol),
        fgs.is_borinic_acid(mol),
        fgs.is_borinic_ester(mol),

        fgs.is_phosphine(mol),
        fgs.is_phosphonic_acid(mol),
        fgs.is_phosphodiester(mol),
        fgs.is_phosphate(mol),

        fgs.is_haloalkane(mol),
        fgs.is_fluoroalkane(mol),
        fgs.is_chloroalkane(mol),
        fgs.is_bromoalkane(mol),
        fgs.is_iodoalkane(mol)
    ]

    return np.array(binary_vector).reshape(1,-1).astype(int)


def task_to_dim(task_name):
    task2dim = {
        'tox21'   : 12,
        'hiv'     : 1,
        'bace'    : 1,
        'bbbp'    : 1,
        'muv'     : 17,
        'toxcast' : 617,
        'sider'   : 27,
        'pcba'    : 128,
        'clintox' : 2
    }

    return task2dim[task_name]

def mol_to_nx_molpla(mol, linker_indices=[]):
    rdkit_features = {
        'atomic_num':       list(range(0, 128)),
        'formal_charge':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'chiral_tag':       list(Chem.rdchem.ChiralType.names.values()),
        'hybridization':    list(Chem.rdchem.HybridizationType.names.values()),
        'num_explicit_hs':  [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'is_aromatic':      [False, True],
        'bond_type':        list(Chem.rdchem.BondType.names.values()),
        'is_conjugated':    [False, True],
        'is_linker':        [False, True],
        'bond_dir':         list(Chem.rdchem.BondDir.names.values()),
        'bond_stereo':      list(Chem.rdchem.BondStereo.names.values())
    }
    # mol to networkx graph object
    G = nx.Graph()
    for atom in mol.GetAtoms():
        IsLinker = True if atom.GetIdx() in linker_indices else False
        G.add_node(atom.GetIdx(),
                   atomic_num=rdkit_features['atomic_num'].index(atom.GetAtomicNum()),
                   formal_charge=rdkit_features['formal_charge'].index(atom.GetFormalCharge()),
                   chiral_tag=rdkit_features['chiral_tag'].index(atom.GetChiralTag()),
                   hybridization=rdkit_features['hybridization'].index(atom.GetHybridization()),
                   num_explicit_hs=rdkit_features['num_explicit_hs'].index(atom.GetNumExplicitHs()),
                   is_aromatic=rdkit_features['is_aromatic'].index(atom.GetIsAromatic()),
                   is_linker=rdkit_features['is_linker'].index(IsLinker))

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in linker_indices:
            IsLinker = True
        elif bond.GetEndAtomIdx() in linker_indices:
            IsLinker = True
        else:
            IsLinker = False
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=rdkit_features['bond_type'].index(bond.GetBondType()),
                   is_aromatic=rdkit_features['is_aromatic'].index(bond.GetIsAromatic()),
                   is_conjugated=rdkit_features['is_conjugated'].index(bond.GetIsConjugated()),
                   is_linker=rdkit_features['is_linker'].index(IsLinker),
                   bond_dir = rdkit_features['bond_dir'].index(bond.GetBondDir()),
                   bond_stereo = rdkit_features['bond_stereo'].index(bond.GetStereo()))
        
    return G

def batch_extract_to_list(list_batch_dicts: List[dict], key: str):

    return [bdict[key] for bdict in list_batch_dicts]


def get_number_functional_arms(df):
    all_arms = []
    for x, y in zip(df['smiles_arms'].values.tolist(), df['data_instance_id'].values.tolist()):
        nonlinked_arms = []
        candidate_arms = x.split(' & ')
        islinked_arms  = y.split('-')[-1]
        for arm, islinked in zip(candidate_arms, islinked_arms):
            if int(islinked) == 0:
                all_arms.append(arm)

    return all_arms, dict(Counter(all_arms))


def get_nonlinked_arms(x):
    result         = []
    candidate_arms = x['smiles_arms'].split(' & ')
    islinked_arms  = x['data_instance_id'].split('-')[-1]
    for arm, islinked in zip(candidate_arms, islinked_arms):
        if int(islinked) == 0:
            result.append(arm)

    return ' & '.join(result)


def get_condition_vector(shared_dict, x):
    mol_arm = Chem.MolFromSmiles(x.split('.')[-1])
    try:
        shared_dict[x] = get_molecular_functional_groups(mol_arm)
    except:
        shared_dict[x] = np.zeros((1, 88))


def _wrap_dummy_nodes(x: Data, attr: str, dummy_value: int):
    x[attr][x['is_linker'].bool()] = dummy_value

    return x

def _wrap_dummy_edges(x: Data, attr: str, dummy_value: int):
    x[attr][x['edge_is_linker'].bool()] = dummy_value

    return x

def _wrap_dummy_data(x: Data):
    x = _wrap_dummy_nodes(x, 'atomic_num', 128)
    x = _wrap_dummy_nodes(x, 'formal_charge', 11)
    x = _wrap_dummy_nodes(x, 'chiral_tag', 9)
    x = _wrap_dummy_nodes(x, 'hybridization', 9)
    x = _wrap_dummy_nodes(x, 'num_explicit_hs', 9)
    x = _wrap_dummy_nodes(x, 'is_aromatic', 2)
    x = _wrap_dummy_edges(x, 'is_conjugated', 2)
    x = _wrap_dummy_edges(x, 'edge_is_aromatic', 2)
    x = _wrap_dummy_edges(x, 'bond_type', 22)
    x = _wrap_dummy_edges(x, 'bond_dir', 7)
    x = _wrap_dummy_edges(x, 'bond_stereo', 6)

    return x

class DataMolPLA(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        # elif key == 'G_edge_index':
        #     return self.G_markers.sum()
        # elif key == 'P_edge_index':
        #     return self.P_markers.sum()
        # elif key == 'R_edge_index':
        #     return self.R_markers.sum()
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif 'R_indices' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        else:
            return 0

        return super().__inc__(key, value, *args, **kwargs)

def convert_molpla_data(data, rcond_settings, rgroup2count):
    G_data                 = linker_to_boolean(from_networkx(data['complete_mol_nx']))
    P_data                 = linker_to_boolean(from_networkx(data['incomplete_mol_nx']))
    G_data.linker_info     = sum(G_data.linker_info, [])
    P_data.linker_info     = sum(P_data.linker_info, [])

    core_linkers           = (P_data.is_linker==True).nonzero().reshape(-1).numpy().tolist()
    islinked_rgroups       = [bool(int(x)) for x in data['islinked_rgroups']]
    R_data, R_data_indices, R_idx, list_R_smiles = [], [], 0., []

    for i, x in enumerate(G_data.linker_info):
        if x is not None:
            G_data.is_linker[i] = True

    if len(core_linkers) != len(data['disjoint_rgroups_nx']):
        import pdb; pdb.set_trace()

    for islinked, r_data_nx, core_linker, r_smiles in zip(islinked_rgroups, 
                                                          data['disjoint_rgroups_nx'], 
                                                          core_linkers, 
                                                          data['smiles_masked_rgroups']):
        if not islinked:
            r_data               = linker_to_boolean(from_networkx(r_data_nx))
            r_data.linker_info   = sum(r_data.linker_info, [])
            r_data.is_linker_mod = r_data.is_linker.clone()
            R_data.append(r_data)
            R_data_indices.extend([R_idx for _ in range(r_data.num_nodes)])
            R_idx+=1
            list_R_smiles.append(r_smiles)
        else:
            P_data.is_linker[core_linker] = False
            for idx in range(len(P_data.edge_is_linker)):
                if P_data.edge_index[0,idx] == core_linker or P_data.edge_index[1,idx] == core_linker:
                    P_data.edge_is_linker[idx] = False

    if len(core_linkers) != len(islinked_rgroups):
        import pdb; pdb.set_trace()

    if len(R_data_indices) == 0:
        import pdb; pdb.set_trace()

    if P_data.is_linker.sum() != len(R_data):
        import pdb; pdb.set_trace()

    # Markers
    G_markers = [True for _ in range(G_data.num_nodes)] + [False for _ in range(P_data.num_nodes)]
    for r_data in R_data:
        G_markers.extend([False for _ in range(r_data.num_nodes)])
    P_markers = [False for _ in range(G_data.num_nodes)] + [True for _ in range(P_data.num_nodes)]
    for r_data in R_data:
        P_markers.extend([False for _ in range(r_data.num_nodes)])
    R_markers = [False for _ in range(G_data.num_nodes)] + [False for _ in range(P_data.num_nodes)]
    for r_data in R_data:
        R_markers.extend([True for _ in range(r_data.num_nodes)])

    # Edge Markers
    G_markers_edge = [True for _ in range(G_data.num_edges)] + [False for _ in range(P_data.num_edges)]
    for r_data in R_data:
        G_markers_edge.extend([False for _ in range(r_data.num_edges)])
    P_markers_edge = [False for _ in range(G_data.num_edges)] + [True for _ in range(P_data.num_edges)]
    for r_data in R_data:
        P_markers_edge.extend([False for _ in range(r_data.num_edges)])
    R_markers_edge = [False for _ in range(G_data.num_edges)] + [False for _ in range(P_data.num_edges)]
    for r_data in R_data:
        R_markers_edge.extend([True for _ in range(r_data.num_edges)])

    # 
    G_data.is_linker_mod  = G_data.is_linker.clone()
    linker_info = [x for x in G_data.linker_info if x is not None] 
    islinked    = [True if x == '1' else False for x  in data['islinked_rgroups']]
    linker_tuples = sorted(zip(linker_info, islinked), key=lambda pair: int(pair[0].split('.')[0].split(':')[-1]), reverse=False)
    for linker_tuple in linker_tuples:
        if linker_tuple[1]:
            pos = int(linker_tuple[0].split(':')[0])
            G_data.is_linker_mod[pos] = False
    
    G_data.is_linker      = G_data.is_linker      & False
    G_data.edge_is_linker = G_data.edge_is_linker & False
    P_data.is_linker_mod  = P_data.is_linker.clone()

    # Make whole geometric data
    R_data                = collate(DataMolPLA, data_list=R_data,                   increment=True,  add_batch=False)[0]
    W_data                = collate(DataMolPLA, data_list=[G_data, P_data, R_data], increment=True, add_batch=False)[0]
    W_data                = _wrap_dummy_data(W_data)

    W_data.G_markers      = torch.BoolTensor(G_markers)
    W_data.P_markers      = torch.BoolTensor(P_markers)
    W_data.R_markers      = torch.BoolTensor(R_markers)
    W_data.R_indices      = torch.LongTensor(R_data_indices)
    # W_data.G_markers_edge = torch.BoolTensor(G_markers_edge)
    # W_data.P_markers_edge = torch.BoolTensor(P_markers_edge)
    # W_data.R_markers_edge = torch.BoolTensor(R_markers_edge)

    # Seperate edge index
    # W_data.G_edge_index     = G_data.edge_index
    # W_data.P_edge_index     = P_data.edge_index
    # W_data.R_edge_index     = R_data.edge_index

    # Add condition vector and Rgroup counts
    rgroup_selected      = [x for x in list_R_smiles]
    rgroup_conditioned   = rcond_settings['rgroup_conditioned']
    rgroup2condvec       = rcond_settings['rgroup2condvec']
    condvec = np.zeros((len(rgroup_selected), 88))
    if rgroup_conditioned == 'rgroups':
        condvec = np.vstack( [rgroup2condvec[x] for x in rgroup_selected])
    W_data.condvec      = torch.FloatTensor(condvec)
    W_data.rgroup_count = torch.FloatTensor([rgroup2count[x] for x in rgroup_selected])

    # Add meta information
    data_instances_extended = [data['data_instance_id']+f'[{i}]' for i in rgroup_selected]
    W_data.data_instance_id = data_instances_extended
    W_data.G_smiles         = data['smiles_original']
    W_data.C_smiles         = data['smiles_masked_core']
    W_data.R_smiles         = list_R_smiles
    W_data.P_smiles         = data['smiles_partially_assembled']

    return W_data


###############################################
#                                             #
#              Dataset Base Class             #
#                                             #
###############################################
class DatasetBase(GeometricDataset):
    def __init__(self, conf):
        self.data_instances = []
        self.meta_instances = []
        self.dim_outputs    = 1
        self.kfold_splits   = []
        self.large_dataset  = False
        self.atomfg_dict    = None

        if conf.dataprep.dataset not in ['malaria', 'lipophilicity', 'esol', 'geom', 'pubchem', 'freesolv', 'drugbank']:
            self.output_dim = task_to_dim(conf.dataprep.dataset)
        else:
            self.output_dim = 1

        if conf.dataprep.dataset in ['geom', 'drugbank']:
            self.path_parquet   = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}.parquet.gzip')
            self.path_moldict   = f'{conf.path.dataset}/geom_{conf.dataprep.version}'
            self.dataframe      = pd.read_parquet(self.path_parquet)
            self.filter_double_joint_cases()
            self.data_instances = self.dataframe['data_instance_id'].values.tolist()
            self.unique_arms    = self.make_list_usable_unique_arms()

            self.path_vocab     = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}_arms_vocab.pickle')
            if not os.path.isfile(self.path_vocab):
                self.make_arms_vocab_dict()

            self.path_count     = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}_arms_count.pickle')
            if not os.path.isfile(self.path_count):
                self.make_arms_count_dict()

            if conf.dataprep.subsample:
                print("Before Subsampling Data Instances", self.dataframe.shape[0])
                self.dataframe      = self.dataframe.sample(frac=conf.dataprep.subsample, replace=False)
                print("After  Subsampling Data Instances", self.dataframe.shape[0])
                self.data_instances = self.dataframe['data_instance_id'].values.tolist()

            if conf.dataprep.filter_rare_arms:
                print("Before Removing Rare R-Subgraphs", self.dataframe.shape[0])
                self.filter_rare_functional_arms()
                print("After  Removing Rare R-Subgraphs", self.dataframe.shape[0])
                self.data_instances = self.dataframe['data_instance_id'].values.tolist()

            if conf.dataprep.filter_partial_mols:
                print("Before Removing Partially Assembled Molecular Graphs", self.dataframe.shape[0])
                self.filter_partially_assembled_mols()
                print("After  Removing Partially Assembled Molecular Graphs", self.dataframe.shape[0])
                self.data_instances = self.dataframe['data_instance_id'].values.tolist()

            self.path_condv     = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}_arms_condv.pickle')
            if not os.path.isfile(self.path_condv):
                self.make_arms_condv_dict()

            self.rgroup2count     = pickle.load(open(self.path_count, 'rb'))
            self.prop_conditioned = conf.model_params.prop_conditioned
            self.arm2condvector = pickle.load(open(self.path_condv, 'rb'))
            self.rcond_settings = dict(rgroup_conditioned=conf.model_params.prop_conditioned,
                                       rgroup2condvec=self.arm2condvector)

            if conf.dev_mode.toy_test or conf.dev_mode.debugging:
                self.data_instances = self.data_instances[:10000]

            if conf.dataprep.dataset == 'drugbank':
                print("EXTERNAL DATASET: DRUGBANK")

                self.path_parquet      = os.path.join(conf.path.dataset, f'drugbank_{conf.dataprep.version}.parquet.gzip')
                self.path_moldict      = f'{conf.path.dataset}/drugbank_{conf.dataprep.version}'
                self.dataframe         = pd.read_parquet(self.path_parquet)
                self.filter_double_joint_cases()
                self.data_instances    = self.dataframe['data_instance_id'].values.tolist()
                self.added_unique_arms = self.update_list_usable_unique_arms()

                self.path_vocab        = os.path.join(conf.path.dataset, f'drugbank_{conf.dataprep.version}_arms_vocab.pickle')
                if not os.path.isfile(self.path_vocab):
                    self.update_arms_vocab_dict(conf)

                self.path_condv        = os.path.join(conf.path.dataset, f'drugbank_{conf.dataprep.version}_arms_condv.pickle')
                if not os.path.isfile(self.path_condv):
                    self.make_arms_condv_dict()
                
                self.prop_conditioned  = conf.model_params.prop_conditioned
                self.arm2condvector    = pickle.load(open(self.path_condv, 'rb'))
                self.rcond_settings    = dict(rgroup_conditioned=conf.model_params.prop_conditioned,
                                              rgroup2condvec=self.arm2condvector)

            super().__init__(root=self.path_moldict, 
                             transform=None, 
                             pre_transform=convert_molpla_data, 
                             pre_filter=None)

            self.data_indices = list(range(len(self.data_instances)))




        else:
            self.task_benchmark = conf.dataprep.dataset
            self.path_benchmark = os.path.join(conf.path.dataset,   f'benchmark/{conf.dataprep.dataset}/raw')
            self.path_benchmark = os.path.join(self.path_benchmark, f'{conf.dataprep.dataset}.csv')
            self.data_instances = self.make_downstream_dataset()


    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def filter_double_joint_cases(self):
        def is_double_joint(x):
            return len(x['smiles_core'].split('.')) != len(x['smiles_arms'].split(' & '))+1
        self.dataframe['is_double'] = self.dataframe.apply(is_double_joint, axis=1)
        self.dataframe = self.dataframe[~self.dataframe['is_double']]

    def filter_partially_assembled_mols(self):
        self.dataframe['is_partial'] = self.dataframe.apply(lambda x: '1' in x['data_instance_id'].split('-')[-1], axis=1)
        self.dataframe = self.dataframe[~self.dataframe['is_partial']]
        
    def filter_rare_functional_arms(self):
        total_arms, counter_arms = get_number_functional_arms(self.dataframe)
        rare_arms = set([k for k,v in counter_arms.items() if v < 2])
        self.dataframe['nonlinked_arms'] = self.dataframe.apply(get_nonlinked_arms, axis=1)
        self.dataframe['filter'] = self.dataframe.apply(lambda x: len(set(x['nonlinked_arms'].split(' & ')) & rare_arms) > 0, axis=1)
        self.dataframe = self.dataframe[~self.dataframe['filter']]

    def make_list_usable_unique_arms(self):
        unique_arms = []
        for x, y in zip(self.dataframe['smiles_arms'].values.tolist(), self.dataframe['data_instance_id'].values.tolist()):
            candidate_arms = x.split(' & ')
            islinked_arms  = y.split('-')[-1]
            for arm, islinked in zip(candidate_arms, islinked_arms):
                if int(islinked) == 0:
                    unique_arms.append(arm)
        unique_arms = list(set(unique_arms))
        print("Number of Recommendable Molecular Arms", len(unique_arms))

        return unique_arms

    def update_list_usable_unique_arms(self):
        unique_arms = self.unique_arms
        for x, y in zip(self.dataframe['smiles_arms'].values.tolist(), self.dataframe['data_instance_id'].values.tolist()):
            candidate_arms = x.split(' & ')
            islinked_arms  = y.split('-')[-1]
            for arm, islinked in zip(candidate_arms, islinked_arms):
                if int(islinked) == 0:
                    unique_arms.append(arm)
        unique_arms = list(set(unique_arms))
        print("Number of Recommendable Molecular Arms", len(unique_arms))

        return unique_arms

    def make_arms_vocab_dict(self):
        print("Making Arms Vocab Dictionary...")
        d = dict()
        for _id in tqdm(self.data_instances):
            molpla_item  = pickle.load(open(os.path.join(self.path_moldict+'/raw', _id+'.pickle'), 'rb'))
            arms_nxgraph = molpla_item['disjoint_rgroups_nx']
            arms_smiles  = molpla_item['smiles_masked_rgroups']
            assert len(arms_nxgraph) == len(arms_smiles)
            for sm, nx in zip(molpla_item['smiles_masked_rgroups'],molpla_item['disjoint_rgroups_nx']):
                if nx != None:
                    d[sm] = nx
        pickle.dump(d, open(self.path_vocab, 'wb'))
        print("Size of Arm Vocab Dictionary: ", len(d))

    def update_arms_vocab_dict(self, conf):
        print("Updating Arms Vocab Dictionary...")
        arms_vocab_path = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}_arms_vocab.pickle')
        d               = pickle.load(open(arms_vocab_path, 'rb'))
        for _id in tqdm(self.data_instances):
            molpla_item  = pickle.load(open(os.path.join(self.path_moldict+'/raw', _id+'.pickle'), 'rb'))
            arms_nxgraph = molpla_item['disjoint_rgroups_nx']
            arms_smiles  = molpla_item['smiles_masked_rgroups']
            assert len(arms_nxgraph) == len(arms_smiles)
            for sm, nx in zip(molpla_item['smiles_masked_rgroups'],molpla_item['disjoint_rgroups_nx']):
                if nx != None:
                    d[sm] = nx
        pickle.dump(d, open(self.path_vocab, 'wb'))
        print("Size of Arm Vocab Dictionary: ", len(d))

    def make_arms_count_dict(self):
        print("Making Arms Count Dictionary...")
        list_functional_arms = []
        for v in tqdm(self.dataframe.smiles_arms.values.tolist()):
            list_functional_arms.extend(v.split(' & '))
        arm2count = Counter(list_functional_arms) 
        pickle.dump(arm2count, open(self.path_count, 'wb'))
        print("Size of Arm Count Dictionary: ", len(arm2count))

    def make_arms_condv_dict(self):
        print("Making Arms Condition Vector Dictionary...")
        with mp.Manager() as manager:
            arm2condvector = manager.dict()
            with manager.Pool(32) as pool:
                pool.starmap(get_condition_vector, [(arm2condvector, x) for x in self.unique_arms])
            arm2condvector = dict(arm2condvector)    
        pickle.dump(arm2condvector, open(self.path_condv, 'wb'))
        print("Size of Arm Condition Vector Dictionary: ", len(arm2condvector))

    def update_arms_condv_dict(self):
        print("Updating Arms Condition Vector Dictionary...")
        with mp.Manager() as manager:
            arm2condvector = manager.dict()
            with manager.Pool(32) as pool:
                pool.starmap(get_condition_vector, [(arm2condvector, x) for x in self.added_unique_arms])
            arm2condvector = dict(arm2condvector)    
        pickle.dump(arm2condvector, open(self.path_condv, 'wb'))
        print("Size of Arm Condition Vector Dictionary: ", len(arm2condvector))


    def make_random_splits(self):
        print("Making Random Splits")
        kf = KFold(n_splits=5, shuffle=True)
        for train_indices, test_indices in kf.split(self.data_indices):
            train_indices, valid_indices = train_test_split(train_indices, test_size=0.0625)
            assert len(set(train_indices) & set(valid_indices)) == 0
            assert len(set(valid_indices) & set(test_indices)) == 0
            assert len(set(train_indices) & set(test_indices)) == 0
            self.kfold_splits.append((train_indices.tolist(), 
                                      valid_indices.tolist(), test_indices.tolist()))

    def make_random_splits_single(self):
        print("Making Random Splits")
        train_indices, test_indices  = train_test_split(self.data_indices, test_size=0.1)    
        train_indices, valid_indices = train_test_split(train_indices,     test_size=0.001)
        assert len(set(train_indices) & set(valid_indices)) == 0
        assert len(set(valid_indices) & set(test_indices)) == 0
        assert len(set(train_indices) & set(test_indices)) == 0
        self.kfold_splits.append((train_indices, valid_indices, test_indices))

        print("Number of Training   Data Instances: ", len(train_indices))
        print("Number of Validation Data Instances: ", len(valid_indices))
        print("Number of Test       Data Instances: ", len(test_indices))

    def make_zeroshot_testing(self):
        print("ZERO-SHOT TESTING!!!!")
        train_indices = [self.data_indices[0]]
        valid_indices = [self.data_indices[1]]
        test_indices  = self.data_indices
        self.kfold_splits.append((train_indices, valid_indices, test_indices))

        print("Number of Training   Data Instances: ", len(train_indices))
        print("Number of Validation Data Instances: ", len(valid_indices))
        print("Number of Test       Data Instances: ", len(test_indices))
   
    def make_scaffold_splits(self):
        print("Making Scaffold Splits")
        kf = KFold(n_splits=5, shuffle=True)
        list_scaffold_smiles = [MurckoScaffoldSmiles(mol=x[1]) for x in self.data_instances]
        list_scaffold_smiles = np.array(list(set(list_scaffold_smiles)))
        for train_indices, test_indices in kf.split(list_scaffold_smiles):
            train_indices, valid_indices = train_test_split(train_indices, test_size=0.0625)
            assert len(set(train_indices) & set(valid_indices)) == 0
            assert len(set(valid_indices) & set(test_indices)) == 0
            assert len(set(train_indices) & set(test_indices)) == 0
            train_scaffolds = list_scaffold_smiles[train_indices]
            valid_scaffolds = list_scaffold_smiles[valid_indices]
            test_scaffolds  = list_scaffold_smiles[test_indices]

            train_indices, valid_indices, test_indices = [], [], []
            for idx, data_tuple in enumerate(self.data_instances):
                if MurckoScaffoldSmiles(mol=data_tuple[1]) in train_scaffolds:
                    train_indices.append(idx)
                if MurckoScaffoldSmiles(mol=data_tuple[1]) in valid_scaffolds:
                    valid_indices.append(idx)
                if MurckoScaffoldSmiles(mol=data_tuple[1]) in test_scaffolds:
                    test_indices.append(idx)
            assert len(set(train_indices) & set(valid_indices)) == 0
            assert len(set(valid_indices) & set(test_indices)) == 0
            assert len(set(train_indices) & set(test_indices)) == 0

            self.kfold_splits.append((train_indices, valid_indices, test_indices))

    def make_scaffold_splits_mk2(self, frac_train=0.8, frac_valid=0.1, frac_test=0.1, random_seed=0):
        # must be changed!
        def generate_scaffold(smiles, include_chirality=False):
            scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
            return scaffold

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        smiles_list = [x[0] for x in self.data_instances]

        non_null = np.ones(len(self.data_instances)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

        rng = np.random.RandomState(random_seed)

        scaffold2indices = defaultdict(list)
        for ind, smiles in smiles_list:
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffold2indices[scaffold].append(ind)

        scaffold_sets = rng.permutation(list(scaffold2indices.keys()))

        n_total_valid = int(np.floor(frac_valid * len(self.data_instances)))
        n_total_test  = int(np.floor(frac_test  * len(self.data_instances)))

        train_indices = []
        valid_indices = []
        test_indices  = []

        for scaffold_set in scaffold_sets:
            if len(valid_indices) + len(scaffold_set) <= n_total_valid:
                valid_indices.extend(scaffold2indices[scaffold_set])
            elif len(test_indices) + len(scaffold_set) <= n_total_test:
                test_indices.extend(scaffold2indices[scaffold_set])
            else:
                train_indices.extend(scaffold2indices[scaffold_set])

        assert len(set(train_indices) & set(valid_indices)) == 0
        assert len(set(valid_indices) & set(test_indices)) == 0
        assert len(set(train_indices) & set(test_indices)) == 0
        self.kfold_splits.append((train_indices, valid_indices, test_indices))

    def make_scaffold_splits_mk3(self, frac_train=0.8, frac_valid=0.1, frac_test=0.1, random_seed=0):
        def generate_scaffold(smiles, include_chirality=False):
            scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
            return scaffold

        smiles_list = [x[0] for x in self.data_instances]
        all_scaffolds = {}
        for i, smiles in enumerate(smiles_list):
            try:
                scaffold = generate_scaffold(smiles, include_chirality=True)
                if scaffold not in all_scaffolds:
                    all_scaffolds[scaffold] = [i]
                else:
                    all_scaffolds[scaffold].append(i)
            except ValueError:
                pass

        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        train_cutoff = frac_train * len(smiles_list)
        valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0
        self.kfold_splits.append((train_idx, valid_idx, test_idx))

    def make_downstream_dataset(self):
        if self.task_benchmark == "tox21":
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.path_benchmark)

        elif self.task_benchmark == "hiv":
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(self.path_benchmark)

        elif self.task_benchmark == "bace":
            smiles_list, rdkit_mol_objs, labels = _load_bace_dataset(self.path_benchmark)

        elif self.task_benchmark == "bbbp":
            smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(self.path_benchmark)

        elif self.task_benchmark == "clintox":
            smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(self.path_benchmark)

        elif self.task_benchmark == "esol":
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(self.path_benchmark)

        elif self.task_benchmark == "freesolv":
            smiles_list, rdkit_mol_objs, labels = _load_freesolv_dataset(self.path_benchmark)

        elif self.task_benchmark == "lipophilicity":
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(self.path_benchmark)

        elif self.task_benchmark == "muv":
            smiles_list, rdkit_mol_objs, labels = _load_muv_dataset(self.path_benchmark)

        elif self.task_benchmark == "sider":
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.path_benchmark)

        elif self.task_benchmark == "toxcast":
            smiles_list, rdkit_mol_objs, labels = _load_toxcast_dataset(self.path_benchmark)

        else:
            raise ValidError("Invalid Benchmark Dataset")

        if type(smiles_list) == pd.core.series.Series:
            smiles_list = smiles_list.to_list()
        return list(zip(smiles_list, rdkit_mol_objs, labels))



def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    # convert 0 to -1, nan to 0
    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["HIV_active"]
    labels = labels.replace(0, -1)
    # convert 0 to -1, there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["mol"], input_df["Class"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1, assuming there are no nans
    labels = labels.replace(0, -1)
    folds = input_df["Model"]
    folds = folds.replace("Train", 0)  # 0 -> train
    folds = folds.replace("Valid", 1)  # 1 -> valid
    folds = folds.replace("Test", 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)

    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["p_np"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]

    # convert 0 to -1, there are no nans
    labels = labels.replace(0, -1)

    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    # drop NoneType, have this issue when re-generate with PyG 2.x
    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]
    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc].reshape(-1, 1),
    )


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = ["FDA_APPROVED", "CT_TOX"]
    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]
    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc],
    )


def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["measured log solubility in mols per litre"]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_freesolv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["expt"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_lipophilicity_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["exp"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_muv_dataset(input_path):
    tasks = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df[tasks]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1, then nan to 0
    # so MUV has three values, -1, 0, 1
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_sider_dataset(input_path):
    tasks = [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]

    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df[tasks]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_toxcast_dataset(input_path):
    # Note: some examples have multiple species,
    #   some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1, then nan to 0
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]

    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc],
    )