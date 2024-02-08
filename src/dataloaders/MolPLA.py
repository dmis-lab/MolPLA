from .base import *

rdkit_features = {
    # node attributes
    'atomic_num':         list(range(0, 128)), 
    'formal_charge':      [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chiral_tag':         list(Chem.rdchem.ChiralType.names.values()),
    'hybridization':      list(Chem.rdchem.HybridizationType.names.values()),
    'num_explicit_hs':    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'is_linker':          [False, True],
    # edge attributes
    'is_conjugated': [False, True],
    'edge_is_aromatic':   [False, True],
    'bond_type':          list(Chem.rdchem.BondType.names.values()),
    'edge_is_linker':     [False, True],
    'bond_dir':           list(Chem.rdchem.BondDir.names.values()),
    'bond_stereo':        list(Chem.rdchem.BondStereo.names.values())
}


class DatasetCore(DatasetBase):
    def __init__(self, conf):
        super().__init__(conf)
        print("Number of Data Instances", len(self.data_instances))

    @property 
    def raw_file_names(self):

        return 

    @property
    def processed_file_names(self):

        return 

    def process(self):
        
        return

    def __getitem__(self, idx):
        
        return self.get(idx)

    def get(self):

        return self.data_instances[idx]

    def len(self):

        return len(self.data_instances)

    def __len__(self):

        return self.len()

class DatasetPretraining(DatasetCore):
    def __init__(self, conf):
        super().__init__(conf)

    @property 
    def raw_file_names(self):

        return [f'{x}.pickle' for x in self.data_instances]

    @property
    def processed_file_names(self):

        return [f'{x}.pt' for x in self.data_instances]

    def process(self):
        raw_paths_removed      = []
        data_instances_removed = []
        idx = 0 
        for raw_path in tqdm(self.raw_paths):
            data = pickle.load(open(raw_path, 'rb'))
            
            if self.pre_filter is not None and not self.pre_filter(data):
                raw_paths_removed.append(raw_path)
                data_instances_removed.append(self.data_instances[idx])
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data, self.rcond_settings, self.rgroup2count)
            
            torch.save(data, os.path.join(self.processed_dir, f'{self.data_instances[idx]}.pt'))
            idx += 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.data_instances[idx]}.pt'))

        return data

    def __getitem__(self, idx):

        return self.get(idx)

    def len(self):

        return len(self.processed_file_names)

    def __len__(self):

        return self.len()

class DatasetMolPLAArms(Dataset):
    def __init__(self, conf, unique_arms):
        super().__init__()
        arms_vocab_path     = os.path.join(conf.path.dataset, f'geom_{conf.dataprep.version}_arms_vocab.pickle')
        self.smiles2nxgraph = pickle.load(open(arms_vocab_path, 'rb'))
        if unique_arms:
            self.get_subset_arms_vocab(unique_arms)

        self.data_instances      = list(self.smiles2nxgraph.keys())
        self.geometric_instances = [from_networkx(self.smiles2nxgraph[x]) for x in tqdm(self.data_instances)]

    def get_subset_arms_vocab(self, unique_arms):
        temp = dict()
        for k,v in self.smiles2nxgraph.items():
            if k in unique_arms:
                temp[k] = v
        self.smiles2nxgraph = temp

    def __getitem__(self, idx):

        return self.geometric_instances[idx]

    def __len__(self):

        return len(self.geometric_instances)

def collate_arms(list_arms_gdata: List[nx.Graph]):
    def _collate(list_data):
        batch_data = collate(list_data[0].__class__, data_list=list_data)
        batch_data = batch_data[0]
        return batch_data

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

    batch_arms_gdata = _collate(list_arms_gdata)

    return _wrap_dummy_data(batch_arms_gdata)

def collate_arms_nomask(list_arms_graph: List[nx.Graph]):
    def _collate(list_data):
        batch_data = collate(list_data[0].__class__, data_list=list_data)
        batch_data = batch_data[0]
        return batch_data

    list_arms_gdata = [from_networkx(x) for x in list_arms_graph]
    batch_arms_gdata = _collate(list_arms_gdata)

    return batch_arms_gdata


class DatasetBenchmark(DatasetCore):
    def __init__(self, conf):
        super().__init__(conf)


    def get(self, idx):
        smiles, rdkit_mol, label = self.data_instances[idx]
        gdata                    = from_networkx(mol_to_nx_molpla(rdkit_mol))

        if not hasattr(gdata, 'bond_type'):
            gdata.bond_type        = torch.LongTensor([]) # torch.LongTensor([22])
            gdata.bond_dir         = torch.LongTensor([]) # torch.LongTensor([7])
            gdata.bond_stereo      = torch.LongTensor([]) # torch.LongTensor([6])
            gdata.edge_is_linker   = torch.LongTensor([]) # torch.LongTensor([2])
            gdata.edge_is_aromatic = torch.LongTensor([]) # torch.LongTensor([2])
            gdata.is_conjugated    = torch.LongTensor([]) # torch.LongTensor([2])

        molpla_dict = {
            'data_instance_id': idx,
            'smiles_original':  smiles,
            'complete_gdata':   gdata,
            'ground_truths':    label
        }

        return molpla_dict


def collate_fn(batch: List[dict]):
    batch_dict = dict()
    batch_dict['full_gdata']       = batch_extract_to_list(batch, 'complete_gdata')
    batch_dict['data_instance_id'] = batch_extract_to_list(batch, 'data_instance_id')
    batch_dict['smiles_original']  = batch_extract_to_list(batch, 'smiles_original')
    batch_dict['ground_truths']    = batch_extract_to_list(batch, 'ground_truths')
    batch_dict['ground_truths']    = torch.cuda.FloatTensor(np.vstack(batch_dict['ground_truths']))

    def _collate(list_data):
        batch_data = collate(list_data[0].__class__, data_list=list_data)
        batch_data = batch_data[0]

        return batch_data.cuda()

    # Collate the Molecular Full Graph Data
    batch_dict['full_gdata'] = _collate(batch_dict['full_gdata'])

    return batch_dict





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--session_name', '-sn', default='development', type=str)

    args = parser.parse_args()
    conf = OmegaConf.load(f'./settings.yaml')[args.session_name]    

    dataset = DatasetAutoload(conf)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)