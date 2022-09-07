import torch
from torch import nn
import numpy as np
from GNN import GNN

class CM_HGNN(nn.Module):
    def __init__(self,args, num_item, num_cat, cat4item) -> None:
        super().__init__()

        self.item_embedding = nn.Embedding(num_item, args.dim)
        self.cat_embedding = nn.Embedding(num_cat, args.dim)
        self.pos_emb = nn.Embedding(200, args.dim)

        self.cat4item = nn.parameter.Parameter(torch.LongTensor(cat4item),requires_grad=False)
        
        self.layer_num = args.layer_num

        self.cat_item_gnn = nn.ModuleList(
                GNN(args.dim,2)  for _ in range(self.layer_num)
        )

        self.item_item_gnn = GNN(args.dim, 4)

        self.cat_cat_gnn = GNN(args.dim,4)
        
        self.item_cat_gnn =nn.ModuleList(
             GNN(args.dim,2) for _ in range(self.layer_num)
        )

        self.alpha1 = nn.parameter.Parameter(torch.zeros(1),requires_grad=False)
        self.alpha2 = nn.parameter.Parameter(torch.zeros(1),requires_grad=False)

        self.w1 = nn.Linear(3*args.dim,2*args.dim)

        self.q = nn.Linear(args.dim*2,1)
        self.w2 = nn.Linear(2*args.dim,2*args.dim)
        self.w3 = nn.Linear(2*args.dim,2*args.dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=stdv)
        nn.init.normal_(self.item_embedding.weight,std=stdv)
        nn.init.normal_(self.cat_embedding.weight,std=stdv)
        nn.init.normal_(self.pos_emb.weight,std=stdv)
        
        nn.init.normal_(self.alpha1,std=stdv)
        nn.init.normal_(self.alpha2,std=stdv)

    def forward(self, batch_data):

        item_emb = self.item_embedding(batch_data['items'])
        cat_emb = self.cat_embedding(batch_data['cats'])

        per_item_emb = torch.split(item_emb, batch_data['item_num']) # split tensor by session length
        per_cat_emb = torch.split(cat_emb, batch_data['cat_num']) # split tensor by session length

        # concat session1[item,cat], session2[item,cat]
        all_emb = torch.concat([torch.concat([i_e,c_e],dim=0) for i_e, c_e in zip(per_item_emb,per_cat_emb)],dim=0)

        item_i = self.item_item_gnn(all_emb, batch_data['item_item_edge'], batch_data['item_item_edge_type'])
        cat_c = self.cat_cat_gnn(all_emb, batch_data['cat_cat_edge'], batch_data['cat_cat_edge_type'])


        for cat_item, item_cat in zip(self.cat_item_gnn, self.item_cat_gnn):
            item_c = cat_item(all_emb, batch_data['cat_item_edge'], batch_data['cat_item_edge_type']) 
            cat_i = item_cat(all_emb, batch_data['item_cat_edge'],  batch_data['item_cat_edge_type'])

            item_c[~batch_data['is_item']] = 0.0
            cat_i[batch_data['is_item']] = 0.0
            all_emb = item_c + cat_i


        cat_emb = cat_i + self.alpha2*cat_c 

        item_emb = item_c + self.alpha1*item_i 

        # convert to sequence
        item_emb = item_emb[batch_data['item2idx']]
        cat_emb = cat_emb[batch_data['cat2idx']]

        hs = torch.concat([item_emb, cat_emb],dim=-1)

        # pos
        pos_emb = self.pos_emb(batch_data['pos_idx'])
        ms = torch.tanh(self.w1(torch.concat([hs,pos_emb],dim=-1)))

        # last item index
        hn = hs[batch_data['last_idx']]

        beta = self.q(torch.sigmoid(self.w2(ms) + self.w3(hn)))

        sess_emb = torch.split(hs*beta,batch_data['sess_len']) # split tensor by session length

        hs = torch.stack([embs.sum(0) for embs in sess_emb],dim=0)

        all_item = torch.concat([self.item_embedding.weight, self.cat_embedding(self.cat4item)],dim=-1)
        return torch.matmul(hs, all_item.T)
