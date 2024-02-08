from .base import *

class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.pcond                          = conf.model_params.prop_conditioned

        self.sg_P                           = conf.model_params.stop_gradient_P
        self.sg_R                           = conf.model_params.stop_gradient_R
        self.sg_Q                           = conf.model_params.stop_gradient_Q

        self.moduledict                     = nn.ModuleDict()
        self.moduledict['graph_encoder']    = load_graph_encoder(conf)
        self.moduledict['node_projector']   = load_node_projector(conf)
        self.moduledict['graph_projector']  = load_graph_projector(conf)

        self.moduledict['query_projector']  = load_link_decoder_conditioned(conf)
        self.moduledict['rgroup_projector'] = load_link_decoder(conf)
        self.graph_pooling                  = load_graph_pooling(conf)

    def set_default_hp(self, trainer):

        return trainer

    def forward(self, data: Data):
        data               = data.cuda()
        # G, P, R            = get_decomposed(data)    

        W_node_features      = self.moduledict['graph_encoder'](data)
        G_node_features      = W_node_features[data.G_markers]
        P_node_features      = W_node_features[data.P_markers]
        R_node_features      = W_node_features[data.R_markers]
        # P_node_features    = self.moduledict['graph_encoder'](P)
        # R_node_features    = self.moduledict['graph_encoder'](R)
        P_node_features      = P_node_features.detach() if self.sg_P else P_node_features
        R_node_features      = R_node_features.detach() if self.sg_R else R_node_features
        Q_node_features      = torch.cat([P_node_features, R_node_features], dim=0)

        # For Loss #2
        G_node_features_lj   = G_node_features[data.is_linker_mod[data.G_markers]]
        P_node_features_lj   = P_node_features[data.is_linker_mod[data.P_markers]]
        R_node_features_lj   = R_node_features[data.is_linker_mod[data.R_markers]]
        Q_node_features_lj   = P_node_features_lj + R_node_features_lj

        G_node_projection_lj = self.moduledict['node_projector'](G_node_features_lj)
        Q_node_projection_lj = self.moduledict['node_projector'](Q_node_features_lj)
        Q_node_projection_lj = Q_node_projection_lj.detach() if self.sg_Q else Q_node_projection_lj
        
        # For Loss #1
        G_node_batch_index   = data.batch[data.G_markers]
        P_node_batch_index   = data.batch[data.P_markers]
        R_node_batch_index   = data.batch[data.R_markers]
        Q_node_batch_index   = torch.cat([P_node_batch_index, R_node_batch_index])

        G_graph_pooled       = self.graph_pooling(G_node_features, G_node_batch_index)
        Q_graph_pooled       = self.graph_pooling(Q_node_features, Q_node_batch_index)

        G_graph_projection   = self.moduledict['graph_projector'](G_graph_pooled)
        Q_graph_projection   = self.moduledict['graph_projector'](Q_graph_pooled)
        Q_graph_projection   = Q_graph_projection.detach() if self.sg_Q else Q_graph_projection

        # For Loss #3
        k_is_linker_masks    = data.is_linker[data.P_markers]
        k_node_features      = P_node_features[k_is_linker_masks]
        k_node_features      = torch.cat([k_node_features, data.condvec], dim=1) if self.pcond != None else k_node_features
        R_graph_pooled       = self.graph_pooling(R_node_features, data.R_indices)

        P_query_projection   = self.moduledict['query_projector'](k_node_features)
        R_graph_projection   = self.moduledict['rgroup_projector'](R_graph_pooled)

        return dict(
            graph_contrastive=(G_graph_projection, Q_graph_projection, None),
            linker_contrastive=(G_node_features_lj, Q_node_features_lj, None),
            rgroup_contrastive=(P_query_projection, R_graph_projection, None),
            data_instance_id=sum(data.data_instance_id, []),
            true_arms_smiles=sum(data.R_smiles, []))

    @torch.no_grad()
    def get_rgroup_library(self, batch_gdata, rank: int):
        arms_node_features = self.moduledict['graph_encoder'](batch_gdata.to(rank))
        arms_linkers       = self.graph_pooling(arms_node_features, batch_gdata.batch)

        return self.moduledict['rgroup_projector'](arms_linkers)

    @torch.no_grad()
    def get_query_projection(self, input_data: nx.Graph, linker_indices: list, condvec: Union[Data, np.ndarray, None]):  
        input_data         = convert_molpla_data_rgroup_retrieval(input_data, linker_indices, condvec).cuda()
        P_node_features    = self.moduledict['graph_encoder'](input_data)
        k_node_features    = P_node_features[input_data.is_linker]
        k_node_features    = torch.cat([k_node_features, input_data.condvec], dim=1)
        linker_projection  = self.moduledict['query_projector'](k_node_features)

        return numpify(linker_projection)

    @torch.no_grad()
    def get_node_embeddings(self, input_data: nx.Graph):
        input_data         = convert_molpla_data_dummified(input_data).cuda()
        G_node_features    = self.moduledict['graph_encoder'](input_data)

        return numpify(G_node_features)

class NetBench(nn.Module):
    def __init__(self, conf):
        super(NetBench, self).__init__()
        self.moduledict = nn.ModuleDict()
        self.task_name  = conf.dataprep.dataset 
        print(self.task_name)
        output_dim      = conf.model_params.output_dim

        self.moduledict                    = nn.ModuleDict()
        self.moduledict['graph_encoder']   = load_graph_encoder(conf)
        # self.moduledict['graph_projector'] = load_graph_projector(conf)
        self.moduledict['task_predictor']  = nn.Linear(conf.model_params.hidden_dim, output_dim)
        self.graph_pooling                 = load_graph_pooling(conf)

        # Loading the Pretrained Parameters from Encoder Module
        if conf.train_params.finetuning.from_pretrained and not conf.train_params.finetuning.from_scratch:
            pretrained_checkpoint = os.path.join(conf.path.checkpoint, conf.train_params.finetuning.from_pretrained)
            pretrained_checkpoint = os.path.join(pretrained_checkpoint, 'last_epoch.mdl')
            print(f"Loading Pretrained Parameters from {pretrained_checkpoint} to Encoder Module")
            print("")
            model_state_dict    = torch.load(pretrained_checkpoint, map_location='cpu')
            state_dict_selected = dict()
            for k,v in model_state_dict.items():
                k = k.split('module.')[1]
                if 'graph_encoder' in k: state_dict_selected[k] = v
                # if 'graph_projector' in k: state_dict_selected[k] = v
            self.load_state_dict(state_dict_selected, strict=False)
            del model_state_dict, state_dict_selected
            torch.cuda.empty_cache() # Lingering GPU

            # Finetuning? Freezing?
            if conf.train_params.finetuning.freeze_pretrained:
                for n,p in self.moduledict['graph_encoder'].named_parameters():
                    p.requires_grad = False
                # for n,p in self.moduledict['graph_projector'].named_parameters():
                #     p.requires_grad = False
            if conf.train_params.finetuning.freeze_normalization:
                for n,p in self.moduledict['graph_encoder'].named_parameters():
                    if 'norm' in n:
                        p.requires_grad = False

        torch.nn.init.zeros_(self.moduledict['task_predictor'].bias)
        torch.nn.init.xavier_normal_(self.moduledict['task_predictor'].weight)

    def set_default_hp(self, trainer):

        return trainer

    def forward(self, batch: dict):
        full_node_features = self.moduledict['graph_encoder'](batch['full_gdata'])
        full_graph_view    = self.graph_pooling(full_node_features, batch['full_gdata'].batch)
        predictions        = self.moduledict['task_predictor'](full_graph_view)

        return {
            'graph_view1'            : full_graph_view,
            f'{self.task_name}/pred' : predictions,
            f'{self.task_name}/true' : batch['ground_truths'],
            'data_instance_id'       : batch['data_instance_id'],
            'input_smiles'           : batch['smiles_original']
        }