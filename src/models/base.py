import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_sparse import SparseTensor
# from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import MessagePassing, PNAConv, GCNConv, GATConv, GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import to_dense_batch, from_networkx
from torch_geometric.data.collate import collate
from torch_geometric.nn.norm.batch_norm import HeteroBatchNorm
from torch_geometric.nn.norm.layer_norm import HeteroLayerNorm
from torch_geometric.data import Data
# from torch_scatter import scatter_add
import pdb
import pickle
import os

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.data import Data, InMemoryDataset
from typing import Callable, Optional, Tuple

from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import add_self_loops, scatter
from typing import Callable, Optional, Tuple, List, Union

import faiss
import networkx as nx

ATOM_FEATURES = ['atomic_num', 'formal_charge', 'chiral_tag', 
                 'hybridization', 'num_explicit_hs', 'is_aromatic']
BOND_FEATURES = ['is_conjugated', 'edge_is_aromatic', 'bond_type', 'bond_dir', 'bond_stereo']

NUM_MOL_PROPS =88

def linker_to_boolean(gdata):
    gdata.is_linker      = gdata.is_linker.bool()
    gdata.edge_is_linker = gdata.edge_is_linker.bool()

    return gdata

def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [element.detach().cpu().numpy() for element in tensor]
    else:
        return tensor

def load_to_cuda(batch: dict):
    new_batch = dict()
    for k, v in batch.items():
        if isinstance(v, Data) or isinstance(v, torch.Tensor):
            v = v.to('cuda')
        new_batch[k] = v

    return new_batch

def load_graph_encoder(conf):
    if conf.model_params.graph_encoder == 'GNN':
        return GNN(
            emb_dim=conf.model_params.hidden_dim,
            aggr=conf.model_params.gnn_params.aggr,
            JK=conf.model_params.gnn_params.JK,
            dropout_ratio=conf.model_params.dropout_rate,
            gnn_type=conf.model_params.gnn_params.gnn_type,
            num_layer=conf.model_params.gnn_params.num_layer,
            norm_type=conf.model_params.gnn_params.norm_type,
            graph_pooling=None)
    else:
        raise ValueError("Invalid Shared Graph Encoder.")

def load_graph_pooling(conf):
    graph_pooling = conf.model_params.graph_pooling
    if graph_pooling == 'sum':
        return global_add_pool
    elif graph_pooling == 'add':
        return global_add_pool
    elif graph_pooling == "mean":
        return global_mean_pool
    elif graph_pooling == "avg":
        return global_mean_pool
    elif graph_pooling == "max":
        return global_max_pool
    else:
        raise ValueError("Invalid graph pooling type.")


def load_node_projector(conf):
    if conf.model_params.node_projector == 'ff':
        return nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim)
    elif conf.model_params.node_projector == 'mlp':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.node_projector == 'softplus':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.Softplus(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.node_projector == 'layernorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LayerNorm(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.node_projector == 'batchnorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.BatchNorm1d(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    else:
        raise ValueError("Invalid Node Projection Head")

def load_graph_projector(conf):
    if conf.model_params.graph_projector == 'ff':
        return nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim)
    elif conf.model_params.graph_projector == 'mlp':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.graph_projector == 'softplus':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.Softplus(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.graph_projector == 'layernorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LayerNorm(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.graph_projector == 'batchnorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.BatchNorm1d(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    else:
        raise ValueError("Invalid Graph Projection Head")

def load_link_decoder(conf):
    if conf.model_params.link_decoder == 'ff':
        return nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim)
    elif conf.model_params.link_decoder == 'mlp':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.link_decoder == 'softplus':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.Softplus(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.link_decoder == 'layernorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.LayerNorm(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.link_decoder == 'batchnorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim, conf.model_params.hidden_dim//2),
            nn.BatchNorm1d(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    else:
        raise ValueError("Invalid Link decoder")

def load_link_decoder_conditioned(conf):
    prop_dim = NUM_MOL_PROPS if conf.model_params.prop_conditioned else 0
    
    if conf.model_params.link_decoder == 'ff':
        return nn.Linear(conf.model_params.hidden_dim+prop_dim, conf.model_params.hidden_dim)
    elif conf.model_params.link_decoder == 'mlp':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim+prop_dim, conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.link_decoder == 'layernorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim+prop_dim, conf.model_params.hidden_dim//2),
            nn.LayerNorm(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    elif conf.model_params.link_decoder == 'batchnorm':
        return nn.Sequential(
            nn.Linear(conf.model_params.hidden_dim+prop_dim, conf.model_params.hidden_dim//2),
            nn.BatchNorm1d(conf.model_params.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(conf.model_params.dropout_rate),
            nn.Linear(conf.model_params.hidden_dim//2, conf.model_params.hidden_dim))
    else:
        raise ValueError("Invalid Link decoder")


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge info by concatenation.

    See https://arxiv.org/abs/1810.00826"""

    def __init__(self, emb_dim, aggr="add"):
        """
        :param emb_dim: int, dimensionality of embeddings for nodes and edges.
        :param aggr: aggregation method, option: "add" or "mean" or "max"
        """
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), 
            nn.ReLU(), 
            nn.Linear(2 * emb_dim, emb_dim))

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def forward(self, x, edge_index, edge_attr):
        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # in message,
        # x_j: (1514, 300)
        # edge_attr: (1514, 300)
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def forward(self, x, edge_index, edge_attr):
        # # add self loops in the edge space
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding['is_conjugated'](edge_attr[:,0]) +
        #                   self.edge_embedding['edge_is_aromatic'](edge_attr[:,1]) +
        #                   self.edge_embedding['bond_type'](edge_attr[:,2])

        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        norm = self.norm(edge_index[0], x.size(0), x.dtype)
        x = self.linear(x)

        # return self.propagate(
        #     self.aggr, edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm
        # )
        return self.propagate(
            edge_index=edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GINConv_Ext(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        self.aggr = aggr
        super(GINConv_Ext, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim))

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def forward(self, x, edge_index, edge_attr):
        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv_Ext(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr="add"):
        super(GCNConv_Ext, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # # add self loops in the edge space
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding['is_conjugated'](edge_attr[:,0]) +
        #                   self.edge_embedding['edge_is_aromatic'](edge_attr[:,1]) +
        #                   self.edge_embedding['bond_type'](edge_attr[:,2])
        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        norm = self.norm(edge_index, x.size(0), x.dtype)

        # https://github.com/THUDM/GraphMAE/blob/6d2636e942f6597d70f438e66ce876f80f9ca9e0/chem/model.py#L103
        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

        self.reset_parameters()

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # # add self loops in the edge space
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding['is_conjugated'](edge_attr[:,0]) +
        #                   self.edge_embedding['edge_is_aromatic'](edge_attr[:,1]) +
        #                   self.edge_embedding['bond_type'](edge_attr[:,2])
        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(aggr=self.aggr, edge_index=edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)

        self.edge_embedding  = nn.ModuleDict()
        self.edge_embedding['is_conjugated']    = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['edge_is_aromatic'] = nn.Embedding(3,  emb_dim, padding_idx=2)
        self.edge_embedding['bond_type']        = nn.Embedding(23, emb_dim, padding_idx=22)
        self.edge_embedding['bond_dir']         = nn.Embedding(8,  emb_dim, padding_idx=7)
        self.edge_embedding['bond_stereo']      = nn.Embedding(7,  emb_dim, padding_idx=6)

        for _, v in self.edge_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

    @staticmethod
    def add_self_loops(x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = dict()

        for i, k in enumerate(BOND_FEATURES):
            self_loop_attr[k] = torch.cuda.LongTensor([0 for _ in range(x.size(0))])

        return edge_index, self_loop_attr

    def forward(self, x, edge_index, edge_attr):
        # # add self loops in the edge space
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding['is_conjugated'](edge_attr[:,0]) +
        #                   self.edge_embedding['edge_is_aromatic'](edge_attr[:,1]) +
        #                   self.edge_embedding['bond_type'](edge_attr[:,2])
        edge_index, self_attr = self.add_self_loops(x, edge_index, edge_attr)
        edge_embeddings = sum([self.edge_embedding[k](edge_attr[k]) for k in BOND_FEATURES])
        self_embeddings = sum([self.edge_embedding[k](self_attr[k]) for k in BOND_FEATURES])
        edge_embeddings = torch.cat([edge_embeddings, self_embeddings], dim=0)

        x = self.linear(x)

        return self.propagate(aggr=self.aggr, edge_index=edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(nn.Module):
    """Wrapper of GNN models"""
    def __init__(self, emb_dim: int, aggr: str, JK: str, dropout_ratio: float, 
                       gnn_type: str, num_layer: int, norm_type: str, graph_pooling: Union[str, None]):
        """
        :param config: include following parameters:
            drop_ratio (float): dropout rate in the {}
            num_layer (int): the number of GNN layers
            JK (str): node reprs, "last", "concat", "max" or "sum"
            gnn_type (str): "gin", "gcn", "graphsage" or "gat" """
        super(GNN, self).__init__()
        self.drop_ratio = dropout_ratio
        self.num_layer  = num_layer
        self.JK         = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding  = nn.ModuleDict()
        self.x_embedding['atomic_num']      = nn.Embedding(129, emb_dim, padding_idx=128)
        self.x_embedding['formal_charge']   = nn.Embedding(12,  emb_dim, padding_idx=11)
        self.x_embedding['chiral_tag']      = nn.Embedding(10,  emb_dim, padding_idx=9)
        self.x_embedding['hybridization']   = nn.Embedding(10,  emb_dim, padding_idx=9)
        self.x_embedding['num_explicit_hs'] = nn.Embedding(10,  emb_dim, padding_idx=9)
        self.x_embedding['is_aromatic']     = nn.Embedding(3,   emb_dim, padding_idx=2)

        for _, v in self.x_embedding.items():
            nn.init.xavier_uniform_(v.weight[:-1,:].data)

        # Graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == None:
            self.pool = None
        else:
            raise ValueError("Invalid graph pooling type.")

        # stacking GNN layers
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr=aggr))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # adding batchnorms
        self.norms = nn.ModuleList()
        for _ in range(num_layer):
            if norm_type == "batchnorm":
                self.norms.append(nn.BatchNorm1d(emb_dim))
            elif norm_type == "layernorm":
                self.norms.append(nn.LayerNorm(emb_dim))
            elif norm_type == "nonorm":
                self.norms.append(nn.Identity())
            else:
                raise ValueError

        if self.JK == "concat":
            self.jk_concat = nn.Linear(emb_dim*(num_layer+1), emb_dim)

    def forward(self, data: Data):
        # Custom Torch Geometric Data Object
        edge_index = data.edge_index
        edge_attr  = {k: data[k] for k in BOND_FEATURES}
        x = sum([self.x_embedding[f](data[f]) for f in ATOM_FEATURES])

        h_list = [x]  # list(x)
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr) 
            h = self.norms[layer](h)
            # h = F.relu(h)
            # h = F.dropout(h, self.drop_ratio, training=self.training)
            if layer == self.num_layer - 1:
                # remove relu in the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
            node_representation = self.jk_concat(node_representation)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        else:
            raise NotImplementedError
            
        return node_representation

    def get_embeddings(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_embeddings = self.forward(x, edge_index, edge_attr)
        graph_embeddings = self.pool(node_embeddings, batch)
        return node_embeddings, graph_embeddings


# def _collate(list_data):
#     batch_data = collate(list_data[0].__class__, data_list=list_data)
#     batch_data = batch_data[0]

#     return batch_data
    

# def _wrap_dummy_nodes(x: Data, attr: str, dummy_value: int):
#     x[attr][x['is_linker'].bool()] = dummy_value

#     return x

# def _wrap_dummy_edges(x: Data, attr: str, dummy_value: int):
#     x[attr][x['edge_is_linker'].bool()] = dummy_value

#     return x

# def _wrap_dummy_data(x: Data):
#     x = _wrap_dummy_nodes(x, 'atomic_num', 128)
#     x = _wrap_dummy_nodes(x, 'formal_charge', 11)
#     x = _wrap_dummy_nodes(x, 'chiral_tag', 9)
#     x = _wrap_dummy_nodes(x, 'hybridization', 9)
#     x = _wrap_dummy_nodes(x, 'num_explicit_hs', 9)
#     x = _wrap_dummy_nodes(x, 'is_aromatic', 2)
#     x = _wrap_dummy_edges(x, 'is_conjugated', 2)
#     x = _wrap_dummy_edges(x, 'edge_is_aromatic', 2)
#     x = _wrap_dummy_edges(x, 'bond_type', 22)
#     x = _wrap_dummy_edges(x, 'bond_dir', 7)
#     x = _wrap_dummy_edges(x, 'bond_stereo', 6)

#     return x

def convert_molpla_data_rgroup_retrieval(data, target_linkers, condvec):
    Q_data                 = linker_to_boolean(from_networkx(data))
    Q_data.linker_info     = sum(Q_data.linker_info, [])
    core_linkers           = (Q_data.is_linker==True).nonzero().reshape(-1).numpy().tolist()

    for core_linker in core_linkers:
        if core_linker not in target_linkers:
            Q_data.is_linker[core_linker] = False
            for idx in range(len(Q_data.edge_is_linker)):
                if Q_data.edge_index[0,idx] == core_linker or Q_data.edge_index[1,idx] == core_linker:
                    Q_data.edge_is_linker[idx] = False

    Q_data = _wrap_dummy_data(Q_data)
    if condvec is None:
        Q_data.condvec = torch.zeros((Q_data.is_linker.sum(),88))
    else:
        Q_data.condvec = torch.FloatTensor(condvec)

    return Q_data

def convert_molpla_data_dummified(data):
    def __wrap_dummy_nodes(x: Data, attr: str, dummy_value: int):
        x[attr][:] = dummy_value

        return x

    def __wrap_dummy_edges(x: Data, attr: str, dummy_value: int):
        x[attr][:] = dummy_value

        return x

    def __wrap_dummy_data(x: Data):
        x = __wrap_dummy_nodes(x, 'atomic_num', 128)
        x = __wrap_dummy_nodes(x, 'formal_charge', 11)
        x = __wrap_dummy_nodes(x, 'chiral_tag', 9)
        x = __wrap_dummy_nodes(x, 'hybridization', 9)
        x = __wrap_dummy_nodes(x, 'num_explicit_hs', 9)
        x = __wrap_dummy_nodes(x, 'is_aromatic', 2)
        x = __wrap_dummy_edges(x, 'is_conjugated', 2)
        x = __wrap_dummy_edges(x, 'edge_is_aromatic', 2)
        x = __wrap_dummy_edges(x, 'bond_type', 22)
        x = __wrap_dummy_edges(x, 'bond_dir', 7)
        x = __wrap_dummy_edges(x, 'bond_stereo', 6)

        return x

    G_data = linker_to_boolean(from_networkx(data))
    G_data.condvec = torch.zeros((G_data.is_linker.sum(),88))
    # G_data = __wrap_dummy_data(G_data)
    
    return G_data

def collate_arms(list_arms_graph: List[nx.Graph]):
    list_arms_gdata = [from_networkx(x) for x in list_arms_graph]
    batch_arms_gdata = _collate(list_arms_gdata)

    return _wrap_dummy_data(batch_arms_gdata)