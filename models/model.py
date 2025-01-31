import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_min,scatter_max
from torch_scatter.composite import scatter_softmax

class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, idx, dim_size):
        layer1_act = F.relu(self.layer1(embeddings))
        layer2_act = self.layer2(layer1_act)
        attention = scatter_softmax(src=layer2_act, index=idx, dim=0)
        visit_embedding = scatter_sum(attention * embeddings, index=idx, dim=0, dim_size=dim_size)
        return visit_embedding


class GraphConv(nn.Module):
    def __init__(self, emb_size, n_hops, n_visits, n_ccss, n_icds, device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.emb_size = emb_size
        self.n_visits = n_visits
        self.n_icds = n_icds
        self.n_ccss = n_ccss
        self.n_nodes = self.n_visits + self.n_ccss + self.n_icds
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device

        idx = torch.arange(self.n_visits).to(self.device)
        #self.visit_union_idx = torch.cat([idx, idx], dim=0)
        self.visit_union_idx = torch.cat([idx, idx, idx, idx, idx], dim=0)

        idx = torch.arange(self.n_ccss + self.n_icds + self.n_visits).to(self.device)
        self.all_union_idx = torch.cat([idx, idx], dim=0)


        self.center_net = Attention(self.emb_size)
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        self.visit_time = torch.load('mimic3_0.05/visit_time.pt')

        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def offset_net(self, embeddings, idx, dim_size):
        return scatter_max(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]


    def forward(self, visit_center, visit_offset, ccs_center, ccs_offset, icd_center, icd_offset, graph1, graph2):

        all_center = torch.cat([visit_center, ccs_center, icd_center], dim=0)
        all_offset = F.relu(torch.cat([visit_offset, ccs_offset, icd_offset], dim=0))


        # all history icd ccs
        _indices = graph1._indices()
        head, tail = _indices[0, :], _indices[1, :]

        #center
        history_embs = all_center[tail]
        visit_center_agg1 = self.center_net(history_embs, head, self.n_nodes)
        visit_center_agg1 = visit_center_agg1[:self.n_visits]
        visit_center_agg1 = F.normalize(visit_center_agg1)

        ## visit offset
        ### visit-ccs, intersect
        inter_visit_ordinal = (head < self.n_visits) & (tail >= self.n_visits) & (tail < self.n_visits + self.n_ccss)
        inter_visit_history_offset = F.relu(all_offset[tail[inter_visit_ordinal]])
        inter_visit_offset_emb1 = self.offset_net(inter_visit_history_offset, head[inter_visit_ordinal], self.n_nodes)
        inter_visit_offset_emb1 = inter_visit_offset_emb1[:self.n_visits]

        ### visit-icd, union
        visit_icd_ordinal = (head < self.n_visits) & (tail >= self.n_visits + self.n_ccss)
        visit_icd_history_offset = F.relu(all_offset[tail[visit_icd_ordinal]])
        ut_visit_offset_emb1 = self.offset_net(visit_icd_history_offset, head[visit_icd_ordinal], self.n_nodes)
        ut_visit_offset_emb1 = ut_visit_offset_emb1[:self.n_visits]



        #-------------------------

        # v-v
        _indices = graph2._indices()
        head, tail = _indices[0, :], _indices[1, :]


        entity_visit_ordinal = (head < self.n_visits) & (tail >= self.n_visits)
        history_embs = all_center[tail[entity_visit_ordinal]]
        # icd ccs
        agg_emb1 = self.center_net(history_embs, head[entity_visit_ordinal], self.n_visits)

        time_embedding = torch.tensor(self.visit_time, dtype = torch.float32, device=self.device).view(-1,1)
        time_embedding = 1.0/time_embedding
        layer1_act = F.relu(self.layer1(time_embedding))
        layer2_act = self.layer2(layer1_act)
        time_embedding = F.softmax(layer2_act, dim=0)
        agg_emb2 = agg_emb1 * time_embedding

        # visit
        visit_visit_ordinal =  (head < self.n_visits) & (tail < self.n_visits)
        history_embs = agg_emb2[tail[visit_visit_ordinal]]
        visit_center_agg2 = self.center_net(history_embs, head[visit_visit_ordinal], self.n_visits)
        visit_center_agg2 = F.normalize(visit_center_agg2)

        lambda1 = 1.0
        visit_final_emb = lambda1*visit_center_agg1 + (1.0-lambda1)*visit_center_agg2


        ## visit offset
        ### visit-ccs, intersect
        inter_visit_ordinal = (head < self.n_visits) & (tail >= self.n_visits) & (tail < self.n_visits + self.n_ccss)
        inter_visit_history_offset = F.relu(all_offset[tail[inter_visit_ordinal]])
        inter_visit_offset_emb2 = self.offset_net(inter_visit_history_offset, head[inter_visit_ordinal], self.n_nodes)
        inter_visit_offset_emb2 = inter_visit_offset_emb2[:self.n_visits]

        ### visit-icd, union
        visit_icd_ordinal = (head < self.n_visits) & (tail >= self.n_visits + self.n_ccss)
        visit_icd_history_offset = F.relu(all_offset[tail[visit_icd_ordinal]])
        ut_visit_offset_emb2 = self.offset_net(visit_icd_history_offset, head[visit_icd_ordinal], self.n_nodes)
        ut_visit_offset_emb2 = ut_visit_offset_emb2[:self.n_visits]

        ### visit-visit, union
        visit_visit_ordinal =  (head < self.n_visits) & (tail < self.n_visits)
        visit_visit_history_offset = F.relu(all_offset[tail[visit_visit_ordinal]])
        visit_visit_offset_emb = self.offset_net(visit_visit_history_offset, head[visit_visit_ordinal], self.n_visits)


        ### union two part
        visit_offset = torch.cat([inter_visit_offset_emb1, ut_visit_offset_emb1], dim=0)
        visit_offset = torch.cat([inter_visit_offset_emb1, ut_visit_offset_emb1, inter_visit_offset_emb2, ut_visit_offset_emb2, visit_visit_offset_emb], dim=0)
        visit_final_offset = F.relu(self.offset_net(visit_offset, self.visit_union_idx, inter_visit_offset_emb1.shape[0]))



        return visit_final_emb, visit_final_offset



class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=64):
        super(MLP, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)


class BoxLM(nn.Module):
    def __init__(self, args, data_stat, adj_mat11, adj_mat12, adj_mat21, adj_mat22):
        super(BoxLM, self).__init__()

        self.beta = args.beta
        self.n_visits = data_stat['n_visits']
        self.n_ccss = data_stat['n_ccss']
        self.n_icds = data_stat["n_icds"]
        self.n_nodes = data_stat['n_nodes']  # n_visits + n_ccss + n_icds
        self.n_entities = data_stat['n_ccss'] + data_stat['n_icds']
        self.decay = args.l2
        self.emb_size = args.dim
        self.n_layers = args.context_hops
        self.logit_cal = args.logit_cal
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")


        self._init_weight()
        self.tau = args.tau
        self.gcn = self._init_model()

        input_dim = 768
        hidden_dim = 128
        output_dim = 16



        self.center_mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.offset_mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)



        self.adj_mat11 = adj_mat11[args.graph_type]
        self.graph11 = self._convert_sp_mat_to_sp_tensor(self.adj_mat11).to(self.device)
        self.graph11 = self.add_residual(self.graph11)

        self.adj_mat12 = adj_mat12[args.graph_type]
        self.graph12 = self._convert_sp_mat_to_sp_tensor(self.adj_mat12).to(self.device)
        self.graph12 = self.add_residual(self.graph12)

        self.adj_mat21 = adj_mat21[args.graph_type]
        self.graph21 = self._convert_sp_mat_to_sp_tensor(self.adj_mat21).to(self.device)
        self.graph21 = self.add_residual(self.graph21)

        self.adj_mat22 = adj_mat22[args.graph_type]
        self.graph22 = self._convert_sp_mat_to_sp_tensor(self.adj_mat22).to(self.device)
        self.graph22 = self.add_residual(self.graph22)


        ccs_embedding = torch.load('mimic3_0.05/ccs_embeddings.pt')
        self.ccs_embedding = torch.from_numpy(ccs_embedding).to(self.device)
        icd_embedding = torch.load('mimic3_0.05/icd_embeddings.pt')
        self.icd_embedding = torch.from_numpy(icd_embedding).to(self.device)

        self.adj1 = torch.load('mimic3_0.05/adj-1.pt').to(self.device)
        self.adj2 = torch.load('mimic3_0.05/adj-2.pt').to(self.device)

        self.w1 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w1.data)
        self.w2 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w2.data)
        self.w3 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w3.data)


    def _init_model(self):
        return GraphConv(emb_size=self.emb_size,
                         n_hops=self.n_layers,
                         n_visits=self.n_visits,
                         n_ccss=self.n_ccss,
                         n_icds=self.n_icds,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_visits + self.n_entities, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.all_offset = initializer(torch.empty([self.n_visits + self.n_entities, self.emb_size])) 
        self.all_offset = nn.Parameter(self.all_offset)



    def add_residual(self, graph):
        residual_node = torch.arange(self.n_nodes).to(self.device)
        row, col = graph._indices()
        row = torch.cat([row, residual_node], dim=0)
        col = torch.cat([col, residual_node], dim=0)
        val = torch.cat([graph._values(), torch.ones_like(residual_node)])

        return torch.sparse.FloatTensor(torch.stack([row, col]), val, graph.shape).to(self.device)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    
    def cal_logit_box(self, visit_center_embedding, visit_offset_embedding, ccs_center_embedding, ccs_offset_embedding, training=True):
        if self.logit_cal == "box":
            gumbel_beta = self.beta
            t1z, t1Z = visit_center_embedding - visit_offset_embedding, visit_center_embedding + visit_offset_embedding
            t2z, t2Z = ccs_center_embedding - ccs_offset_embedding, ccs_center_embedding + ccs_offset_embedding
            z = gumbel_beta * torch.logaddexp(
                    t1z / gumbel_beta, t2z / gumbel_beta
                )
            z = torch.max(z, torch.max(t1z, t2z))
            # z =  torch.max(t1z, t2z)
            Z = -gumbel_beta * torch.logaddexp(
                -t1Z / gumbel_beta, -t2Z / gumbel_beta
            )
            Z = torch.min(Z, torch.min(t1Z, t2Z))
            # Z = torch.min(t1Z, t2Z)
            euler_gamma = 0.57721566490153286060

            return torch.sum(
                torch.log(
                    F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=1 / gumbel_beta) + 1e-23
                ),
                dim=-1,
            )


    def lightgcn(self, ccs_center, icd_center):
        embedding = torch.cat((ccs_center, icd_center), dim=0)


        emb1 = torch.spmm(self.adj1, embedding)
        emb1 = torch.mm(emb1, self.w1)
        emb2 = torch.spmm(self.adj2, embedding)
        emb2 = torch.mm(emb2, self.w2)
        emb3 = embedding
        emb3 = torch.mm(emb3, self.w3)

        embs = emb1 * (1/3) + emb2 * (1/3) + emb3 * (1/3)

        return embs[:ccs_center.shape[0]], embs[ccs_center.shape[0]:]


    def train_generate(self):
        visit_emb, ccs_emb, icd_emb = torch.split(self.all_embed, [self.n_visits, self.n_ccss, self.n_icds])
        visit_offset, ccs_offset, icd_offset = torch.split(self.all_offset, [self.n_visits, self.n_ccss, self.n_icds])
        
        ccs_emb = self.center_mlp(self.ccs_embedding)
        icd_emb = self.center_mlp(self.icd_embedding)
        ccs_offset, icd_offset = self.lightgcn(ccs_emb, icd_emb)


        visit_agg_embs, visit_agg_offset = self.gcn(visit_emb, visit_offset, ccs_emb, ccs_offset, icd_emb, icd_offset, self.graph11, self.graph21)

        ccs_agg_embs = ccs_emb
        icd_agg_emb = icd_emb
        ccs_agg_offset = ccs_offset
        icd_agg_offset = icd_offset

        
        visit_embs = torch.cat([visit_agg_embs, visit_agg_offset], axis=-1)
        ccs_embs = torch.cat([ccs_agg_embs, ccs_agg_offset], axis=-1)

        return visit_embs, ccs_embs

    def test_generate(self):
        visit_emb, ccs_emb, icd_emb = torch.split(self.all_embed, [self.n_visits, self.n_ccss, self.n_icds])
        visit_offset, ccs_offset, icd_offset = torch.split(self.all_offset, [self.n_visits, self.n_ccss, self.n_icds])

        ccs_emb = self.center_mlp(self.ccs_embedding)
        icd_emb = self.center_mlp(self.icd_embedding)
        ccs_offset, icd_offset = self.lightgcn(ccs_emb, icd_emb)

        visit_agg_embs, visit_agg_offset = self.gcn(visit_emb, visit_offset, ccs_emb, ccs_offset, icd_emb, icd_offset,
                                                  self.graph12, self.graph22)

        ccs_agg_embs = ccs_emb
        icd_agg_emb = icd_emb
        ccs_agg_offset = ccs_offset
        icd_agg_offset = icd_offset

        visit_embs = torch.cat([visit_agg_embs, visit_agg_offset], axis=-1)
        ccs_embs = torch.cat([ccs_agg_embs, ccs_agg_offset], axis=-1)

        return visit_embs, ccs_embs


    def rating(self, visit_embs, entity_embs, same_dim=False):
        if same_dim:
            visit_agg_embs, visit_agg_offset = torch.split(visit_embs, [self.emb_size, self.emb_size], dim=-1)
            entity_agg_embs, entity_agg_offset = torch.split(entity_embs, [self.emb_size, self.emb_size], dim=-1)
            return self.cal_logit_box(visit_agg_embs, visit_agg_offset, entity_agg_embs, entity_agg_offset)
        else:
            n_visits = visit_embs.shape[0]
            n_entities = entity_embs.shape[0]
            visit_embs = visit_embs.unsqueeze(1).expand(n_visits, n_entities,  self.emb_size * 2)
            visit_agg_embs, visit_agg_offset = torch.split(visit_embs, [self.emb_size, self.emb_size], dim=-1)

            entity_embs = entity_embs.unsqueeze(0).expand(n_visits, n_entities,  self.emb_size * 2)
            entity_agg_embs, entity_agg_offset = torch.split(entity_embs, [self.emb_size, self.emb_size], dim=-1)

            return self.cal_logit_box(visit_agg_embs, visit_agg_offset, entity_agg_embs, entity_agg_offset)
