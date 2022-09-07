import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np

def load_data(args):
    dataset_folder = os.path.abspath('dataset')

    train_set = SeqDataset(dataset_folder,'train')
    test_set = SeqDataset(dataset_folder,'test')

    train_loader = DataLoader(train_set,args.batch_size,  num_workers=0,
                              shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_set,args.batch_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)

    id_maps = pickle.load(open(os.path.join( dataset_folder,'idmap.pkl'), 'rb'))
    item_num = max(id_maps[0].values())+1
    cat_num = max(id_maps[1].values())+1

    # get category of item
    cat4item = [0]*item_num
    for d in pickle.load(open(os.path.join(dataset_folder, 'train.pkl'),'rb')):
        for i,c in zip(d['items'],d['cats']):
            cat4item[i] = c
    
    return train_loader, test_loader, item_num, cat_num, cat4item

class SeqDataset(Dataset):
    def __init__(self, datafolder, file='train',max_len=50) -> None:
        super().__init__()
        self.max_len = max_len
        data_file = os.path.join(datafolder, file+'.pkl')

        self.data = pickle.load(open(data_file,'rb')) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        # get raw data
        session = self.data[index]['items'][-self.max_len:]
        cats = self.data[index]['cats'][-self.max_len:]
        target = self.data[index]['target']

        sess_len = len(session)

        item_node = np.unique(session)
        item_id_map = dict([(i,idx) for idx,i in enumerate(item_node)]) # item id to index in item_node
        sess_item_num = len(item_id_map)

        cat_node = np.unique(cats)
        cat_id_map = dict([(i,idx+sess_item_num) for idx,i in enumerate(cat_node)]) # cat index followed by item index
        sess_cat_num = len(cat_id_map)

        # restructure session from item_node and cat_node
        item2idx = [item_id_map[i] for i in session] 
        cat2idx = [cat_id_map[i] for i in cats]

            # we will concat item_node and cat_node embedding in model.py
        # item-cat and cat-item edge 
        item_cat_edge = [[],[]] # from item to cat  # item
        item_cat_edge_type = []

        cat_item_edge = [[],[]] # from cat to item 
        cat_item_edge_type = []

        for i,c in zip(session,cats): # 
            item_cat_edge[0].append(item_id_map[i])
            item_cat_edge[1].append(cat_id_map[c])
            item_cat_edge_type.append(0)

            # 3.2.1item embedding 
            cat_item_edge[0].append(cat_id_map[c])
            cat_item_edge[1].append(item_id_map[i])
            cat_item_edge_type.append(0)

        for h, t in zip(session[:-1],session[1:]): 
            # 3.2.1item embedding 
            cat_item_edge[0].append(item_id_map[h])
            cat_item_edge[0].append(item_id_map[t])
            cat_item_edge[1].append(item_id_map[t])
            cat_item_edge[1].append(item_id_map[h])
            cat_item_edge_type.append(1)
            cat_item_edge_type.append(1)


        for h,t in zip(cats[:-1],cats[1:]):
            # different to above, we use a cat_cat_edge because eq14.
            item_cat_edge[0].append(cat_id_map[h])
            item_cat_edge[0].append(cat_id_map[t])
            item_cat_edge[1].append(cat_id_map[t])
            item_cat_edge[1].append(cat_id_map[h])
            item_cat_edge_type.append(1)
            item_cat_edge_type.append(1)
      
        # item-item
        #four edge type --> to, from, to&from ,self
        edge_type = dict()
        for h,t in zip(session[:-1],session[1:]):
            if edge_type.get('%d-%d'%(h,t),-1) >= 1: # 
                # exist reversed edge
                edge_type['%d-%d'%(t,h)] = edge_type['%d-%d'%(h,t)] = 2
            elif edge_type.get('%d-%d'%(h,t),-1) == 0:
                # exist this edge 
                continue
            else:
                # not exist
                edge_type['%d-%d'%(t,h)] = 1
                edge_type['%d-%d'%(h,t)] = 0
        for i in item_id_map.keys():
            edge_type['%d-%d'%(i,i)] = 3


        item_item_edge = [[],[]]
        item_item_edge_type = []
        for e, tp in edge_type.items():
            h,t = e.split('-')
            h = item_id_map[eval(h)]
            t = item_id_map[eval(t)]
            item_item_edge[0].append(h)
            item_item_edge[1].append(t)
            item_item_edge_type.append(tp)

        # cat-cat 
        edge_type = dict() 
        for h,t in zip(cats[:-1],cats[1:]):
            if edge_type.get('%d-%d'%(h,t),-1) >= 1: # 
                # exist reversed edge
                edge_type['%d-%d'%(t,h)] = edge_type['%d-%d'%(h,t)] = 2
            elif edge_type.get('%d-%d'%(h,t),-1) == 0:
                # exist this edge 
                continue
            else:
                # not exist
                edge_type['%d-%d'%(t,h)] = 1
                edge_type['%d-%d'%(h,t)] = 0

        for c in cat_id_map.keys():
            edge_type['%d-%d'%(c,c)] = 3

        cat_cat_edge = [[],[]]
        cat_cat_edge_type = []
        for e, tp in edge_type.items():
            h,t = e.split('-')
            h = cat_id_map[eval(h)]
            t = cat_id_map[eval(t)]
            cat_cat_edge[0].append(h)
            cat_cat_edge[1].append(t)
            cat_cat_edge_type.append(tp)

        return (item_node, cat_node, item2idx, cat2idx,  
                cat_item_edge, cat_item_edge_type, item_item_edge, item_item_edge_type,
                item_cat_edge, item_cat_edge_type, cat_cat_edge, cat_cat_edge_type,
                sess_item_num, sess_cat_num, sess_len, target)


def collate_fn(batch_data):

    batch_item_nodes = [] # 1d 
    batch_cat_nodes = [] # 1d
    batch_item2idx = [] # 1d
    batch_cat2idx = [] # 1d

    batch_cat_item_edge = []
    batch_cat_item_edge_type = []

    batch_item_item_edge = []
    batch_item_item_edge_type = []

    batch_item_cat_edge = []
    batch_item_cat_edge_type = []

    batch_cat_cat_edge = []
    batch_cat_cat_edge_type = []

    batch_sess_item_num = []
    batch_sess_cat_num = []
    batch_sess_len = []
    batch_target = []

    now_idx = 0

    for d in batch_data:
        (item_node, cat_node, item2idx, cat2idx,  
            cat_item_edge, cat_item_edge_type, item_item_edge, item_item_edge_type,
            item_cat_edge, item_cat_edge_type, cat_cat_edge,cat_cat_edge_type,
            sess_item_num, sess_cat_num, sess_len, target) = d

        batch_item_nodes.append(torch.LongTensor(item_node))
        batch_cat_nodes.append(torch.LongTensor(cat_node))

        batch_item2idx.append(torch.LongTensor(item2idx) + now_idx)
        batch_cat2idx.append(torch.LongTensor(cat2idx) + now_idx)

        batch_cat_item_edge.append(torch.LongTensor(cat_item_edge)+now_idx)
        batch_cat_item_edge_type.append(torch.LongTensor(cat_item_edge_type))

        batch_item_item_edge.append(torch.LongTensor(item_item_edge)+now_idx)
        batch_item_item_edge_type.append(torch.LongTensor(item_item_edge_type))

        batch_item_cat_edge.append(torch.LongTensor(item_cat_edge)+now_idx)
        batch_item_cat_edge_type.append(torch.LongTensor(item_cat_edge_type))

        batch_cat_cat_edge.append(torch.LongTensor(cat_cat_edge)+now_idx)
        batch_cat_cat_edge_type.append(torch.LongTensor(cat_cat_edge_type))

        batch_sess_item_num.append(sess_item_num)
        batch_sess_cat_num.append(sess_cat_num)

        batch_sess_len.append(sess_len)
        batch_target.append(target)

        now_idx += sess_item_num
        now_idx += sess_cat_num

    batch_item_nodes = torch.concat(batch_item_nodes,-1)
    batch_cat_nodes = torch.concat(batch_cat_nodes,-1)

    batch_item2idx = torch.concat(batch_item2idx,-1)
    batch_cat2idx = torch.concat(batch_cat2idx,-1)

    batch_cat_item_edge = torch.concat(batch_cat_item_edge,-1)
    batch_cat_item_edge_type = torch.concat(batch_cat_item_edge_type,-1)

    batch_item_item_edge = torch.concat(batch_item_item_edge,-1)
    batch_item_item_edge_type = torch.concat(batch_item_item_edge_type,-1)

    batch_item_cat_edge = torch.concat(batch_item_cat_edge,-1)
    batch_item_cat_edge_type = torch.concat(batch_item_cat_edge_type,-1)

    batch_cat_cat_edge = torch.concat(batch_cat_cat_edge,-1)
    batch_cat_cat_edge_type = torch.concat(batch_cat_cat_edge_type,-1)

    batch_target = torch.LongTensor(batch_target)

    batch_data = {}
    batch_data['items'] = batch_item_nodes
    batch_data['cats'] = batch_cat_nodes

    batch_data['item2idx'] = batch_item2idx
    batch_data['cat2idx'] = batch_cat2idx

    batch_data['item_item_edge'] = batch_item_item_edge
    batch_data['item_item_edge_type'] = batch_item_item_edge_type

    batch_data['cat_item_edge'] = batch_cat_item_edge
    batch_data['cat_item_edge_type'] = batch_cat_item_edge_type

    batch_data['item_cat_edge'] = batch_item_cat_edge
    batch_data['item_cat_edge_type'] = batch_item_cat_edge_type

    batch_data['cat_cat_edge'] = batch_cat_cat_edge
    batch_data['cat_cat_edge_type'] = batch_cat_cat_edge_type

    batch_data['item_num'] = batch_sess_item_num
    batch_data['cat_num'] = batch_sess_cat_num

    batch_data['sess_len'] = batch_sess_len
    batch_data['target'] = batch_target

    # pos  embedding idx
    pos_idx = [np.arange(i-1,-1,-1) for i in batch_data['sess_len']]
    batch_data['pos_idx'] = torch.LongTensor(np.concatenate(pos_idx))

    # last embedding idx
    batch_last_idx = [] # 
    now_idx = -1
    for i in batch_data['sess_len']:
        now_idx += i
        batch_last_idx.extend([now_idx]*i)
    batch_data['last_idx'] = torch.LongTensor(batch_last_idx)

    # is item embedding ? in all embedding
    batch_is_item = []
    for i,c in zip(batch_sess_item_num,batch_sess_cat_num):
        batch_is_item.extend([1]*i)
        batch_is_item.extend([0]*c)
    batch_data['is_item'] = torch.BoolTensor(batch_is_item)

    return batch_data
