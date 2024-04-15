import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup,  get_polynomial_decay_schedule_with_warmup, BertGenerationConfig, BertGenerationDecoder
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import DictConfig, OmegaConf
import hydra
import torchmetrics
from typing import List, Dict
from pytorch_lightning.loggers import WandbLogger
from pandas import DataFrame as df
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd
import esm
import esm.inverse_folding
import torch_geometric
from GPUtil import showUtilization as gpu_usage
import time
import os
import json
import wandb
import csv
import random
import torch
import pickle
import itertools
class CSVDataset(Dataset):
    def __init__(self, csv_path, split, split_seed=42, supervised=True, plddt=85, clust=False):
        super().__init__()
        """
        args:
            csv_path: path to data
            split: one of "train" "val" "test"
            split_seed: used for future work, not yet
            supervised: if True drop all samples without a bind site 
            plddt: plddt cutoff for alphafold confidence
        """
        print('start of data')
        self.df = pd.read_csv(csv_path)
        #import pdb; pdb.set_trace()
        if supervised:
            self.df = self.df.dropna()
        
        #import pdb; pdb.set_trace()       
        
        def alpha(ids):
            return os.path.isfile(f'/vast/og2114/rebase/20220519/output/{ids}/ranked_0.pdb') and (max(json.load(open(f'/vast/og2114/rebase/20220519/output/{ids}/ranking_debug.json'))['plddts'].values()) >= plddt)

        self.df = self.df[self.df['id'] != 'Csp7507ORF4224P']
        
        spl = self.split(split)
        print("pre filter",len(spl))
        #spl = spl[spl['id'].apply(alpha) ==True ]
        print("post filter",len(spl))
        self.data = spl[['seq','bind', 'id', 'cluster']].to_dict('records')
        print(len(self.data))
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.clustered_data = {}
        tmp_clust = self.df.cluster.unique()
        self.cluster_idxs =[]
        for cluster in tmp_clust:
            t = spl[spl['cluster'] == cluster][['seq','bind', 'id', 'cluster']].to_dict('records')

            if len(t) != 0:
                self.clustered_data[cluster]= spl[spl['cluster'] == cluster][['seq','bind', 'id', 'cluster']].to_dict('records')
                self.cluster_idxs.append(cluster)
        self.use_cluster=clust
        #import pdb;pdb.set_trace()
        print('initialized', self.__len__())
    def __getitem__(self, idx):
        if self.use_cluster == False:
            return self.data[idx]
        else:
            return self.clustered_data[self.cluster_idxs[idx]][random.randint(0, (len(self.clustered_data[self.cluster_idxs[idx]])-1))]

    def __len__(self):
        if self.use_cluster== False:
            return len(self.data)
        else:
            return len(self.cluster_idxs)

    def split(self, split):
        '''
        splits data on train/val/test

        args:
            split: One of "train" "val" "test"
        
        returns:
            subsection of data included in the train/val/test split
        '''
        if split.lower() == 'train':
            tmp = self.df[self.df['split'] != 1]
            return tmp[tmp['split'] != 2]
        
        elif split.lower() == 'val':
            return self.df[self.df['split'] == 1]

        elif split.lower() == 'test':
            return self.df[self.df['split'] == 2]


class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens.   will usually have eos already applied,
        but won't have bos. Hence the defaults here.
        
        args:
            dataset: CSVDataset of data
            dictionary: esmif1 dictionary used in code
        '''

        super().__init__(dataset)
        self.dictionary = dictionary
        self.apply_bos = apply_bos
        self.apply_eos = apply_eos
        ''' 
        batchConverter git line 217 - https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py
        '''
        self.batch_converter_coords = esm.inverse_folding.util.CoordBatchConverter(self.dictionary)
        
    def __getitem__(self, idx):
        '''
        Get item from dataset:
        returns:
        {
            'bind': torch.tensor (bind site)
            'coords': esm.inverse_folding.util.extract_coords_from_structure(structure) output
            'seq': torch.tensor sequence
        } to be post-proccessed in self.collate_dicts()

        

        '''
        structure = esm.inverse_folding.util.load_structure(f"/vast/og2114/rebase/20220519/output/{self.dataset[idx]['id']}/ranked_0.pdb", 'A')
        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        return {
            'bind':torch.tensor( self.dictionary.encode(self.dataset[idx]['bind'])),
            'coords': coords,
            'cluster': self.dataset[idx]['cluster'],
            'seq': torch.tensor(self.dictionary.encode(seq))

        }

    def __len__(self):
        return len(self.dataset)

    def collate_tensors(self, batch: List[torch.tensor], bos=None, eos=None):
        '''
        utility for collating tensors together, applying eos and bos if needed, 
        padding samples with self.dictionary.padding_idx as neccesary for length
        
        input:
            batch: [
                torch.tensor shape[l1],
                torch.tensor shape[l2],  
                ...  
            ]
            bos: bool, apply bos (defaults to class init settings) - !!!BOS is practically <af2>, idx 34!!!
            eos: bool, apply eos (defaults to class init settings)
        output:
            torch.tensor shape[len(input), max(l1, l2, ...)+bos+eos]
        '''
        if bos == None:
            bos = self.apply_bos
        if eos == None:
            eos = self.apply_eos
        
        batch_size = len(batch)
        max_len = max(el.size(0) for el in batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(bos) + int(eos) # eos and bos
            ),
            dtype=torch.int64,
        ).fill_(self.dictionary.padding_idx)

        if bos:
            tokens[:, 0] = self.dictionary.get_idx('<af2>')

        for idx, el in enumerate(batch):
            tokens[idx, int(bos):(el.size(0) + int(bos))] = el
            if eos:
                tokens[idx, el.size(0) + int(bos)] = self.dictionary.eos_idx
        
        return tokens
        
            

    def collater(self, batch):
        if isinstance(batch, list) and torch.is_tensor(batch[0]):
            return self.collate_tensors(batch)
        else:
            return self.collate_dicts(batch)
    def collate_dicts(self, batch: List[Dict[str, torch.tensor]]):
        '''
        combine sequences of the form
        [
            {
                'bind': torch.tensor (bind site)
                'coords': esm.inverse_folding.util.extract_coords_from_structure(structure) output
                'seq': torch.tensor sequence
            },
            {
                'bind': torch.tensor (bind site)
                'coords': esm.inverse_folding.util.extract_coords_from_structure(structure) output
                'seq': torch.tensor sequence
            },
        ]
        into a collated form:
        {
            'bind': torch.tensor (bind site)
            'bos_bind': torch.tensor (bos+bind site)
            'coords': torch.tensor (coords input to esm if)
            'seq': torch.tensor (protein sequence)
            'bos_seq': torch.tensor (bos+protein sequence)
            'coord_conf': torch.tensor(confidence input to esmif encoder)
            'coord_pad' torch.tensor (padding_mask input to esm if encoder)
        }
        !!!BOS is practiaclly '<af2>', idx 34!!!
        applying the padding correctly to capture different lengths
        '''

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]
        
        
        pre_proccessed_coords = self.batch_converter_coords.from_lists(select_by_key(batch, 'coords'))
        
        post_proccessed = {
            'bind': self.collate_tensors(select_by_key(batch, 'bind'), bos=False, eos=True),
            'bos_bind': self.collate_tensors(select_by_key(batch, 'bind'), bos=True, eos=True),
            'coords': pre_proccessed_coords[0],
            'seq': self.collate_tensors(select_by_key(batch, 'seq'), bos=False, eos=True),
            'bos_seq': self.collate_tensors(select_by_key(batch, 'seq'), bos=True, eos=True),
            'coord_conf': pre_proccessed_coords[1],
            'coord_pad': pre_proccessed_coords[4],
            'lens': [len(l) for l in select_by_key(batch, 'seq')],
            'bind_lens': [len(l) for l in select_by_key(batch, 'bind')], 
            'cluster': select_by_key(batch, 'cluster')
        }
        return post_proccessed



def enable_cpu_offloading(model):
    """
    Enable CPU offloading for a PyTorch model to manage GPU memory.
    """

    def forward_hook(module, inputs, outputs):
        if torch.cuda.memory_allocated() > torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory * 0.9:
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu()
            elif isinstance(outputs, tuple):
                outputs = tuple(out.cpu() if isinstance(out, torch.Tensor) else out for out in outputs)
            elif isinstance(outputs, list):
                outputs = [out.cpu() if isinstance(out, torch.Tensor) else out for out in outputs]
        return outputs

    # Here you could log or track gradient sizes, but modifying them directly is not advised
    def backward_hook(module, grad_input, grad_output):
        # Example of logging or processing without direct modification
        pass

    for module in model.modules():
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    return model

class EncoderDataset(Dataset):
    def __init__(self, dataset, batch_size, device, path, cluster=True, eos=True):

        super().__init__()
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.cluster = cluster
        self.dictionary = self.dataset.dictionary
        
        #check if file exists at path, if so load it, if not create it
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
            for i in range(len(self.data)):
                self.data[i]['seq_enc'] = self.data[i]['seq_enc'].to(torch.device('cpu'))
                
        else:
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater)
            self.data = []
            self.eos = eos
            print(f'creating embeddings saving to {path}')
            print(f'batch size: {self.batch_size}')
            self.ifmodel, self.ifalphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            self.ifmodel = self.ifmodel.to(self.device)
            self.ifmodel = self.ifmodel.eval()
            #self.ifmodel = enable_cpu_offloading(self.ifmodel)
            for step, batch in enumerate(self.dataloader):
                #predict the encoder output using self.ifmodel.encoder.forward(batch['coords'].to(self.device), batch['coord_pad'].to(self.device), batch['coord_conf'].to(self.device), return_all_hiddens=False). 
                # if GPU runs out of memory, use sharding
                
                encoder_out = self.ifmodel.encoder.forward(batch['coords'].to(self.device), batch['coord_pad'].to(self.device), batch['coord_conf'].to(self.device), return_all_hiddens=False)
                # remove beginning and end (bos and eos tokens)
                embeddings = encoder_out['encoder_out'][0].transpose(0, 1)[:, 1:-1, :]
                #import pdb; pdb.set_trace()
                for i in range(batch['seq'].shape[0]):
                    self.data.append({
                        'seq': batch['seq'][i][int(self.eos):batch['lens'][i]+int(self.eos)],
                        'bind': batch['bind'][i][int(self.eos):batch['bind_lens'][i]+int(self.eos)],
                        'coords': batch['coords'][i][:batch['lens'][i]],
                        'seq_enc': embeddings[i, :batch['lens'][i], :].to(torch.device('cpu')), 
                        'cluster': batch['cluster'][i]
                    })
                #augment self.data from form list[dict[..., cluster]] to list[list`dict[..., cluster]]], where the inner list is a list of dicts with the same cluster
                
            #save self.data to path and create a function to load it back to self.data
            with open(path, 'wb') as f:
                pickle.dump(self.data, f)
            self.path = path
            print('embeddings created', len(self.data))
        self.clustered_data = [list(group) for key, group in itertools.groupby(self.data, lambda x: x['cluster'])]
        print('clustered_data clusters present:', len(self.clustered_data))

        
    def __len__(self):
        if self.cluster:
            return len(self.clustered_data)
        return len(self.data)
    def __getitem__(self, idx):
        if self.cluster:
            return self.clustered_data[idx][random.randint(0, (len(self.clustered_data[idx])-1))]
        return self.data[idx]
    def collate_tensors(self, batch: List[torch.tensor], bos=None, eos=None):
        if bos == None:
            bos = self.dataset.apply_bos
        if eos == None:
            eos = self.dataset.apply_eos
        
        batch_size = len(batch)
        beos = int(bos) + int(eos)
        max_shape = [max(el.size(i) for el in batch)  for i in range(len(batch[0].shape))]
        max_shape[0] = max_shape[0] + beos
        tokens = torch.empty(
            (
                batch_size, 
                *max_shape
            ),
            dtype=torch.int64,

        ).fill_(self.dictionary.padding_idx)


        if bos:
            tokens[:, 0] = self.dictionary.get_idx('<af2>')
        print('batch', len(batch))
        print(tokens.shape)
        print(range(len(batch)))
        print(batch[0].device)
        print(tokens.device)

        for idx in range(len(batch)):
            print('idx', idx)
            tokens[idx, int(bos):(batch[idx].size(0) + int(bos))] = batch[idx]
            if eos:
                tokens[idx, batch[idx].size(0) + int(bos)] = self.dictionary.eos_idx
        print(tokens.shape)
        return tokens
        
            

    def collater(self, batch):
        if isinstance(batch, list) and torch.is_tensor(batch[0]):
            return self.collate_tensors(batch)
        else:
            return self.collate_dicts(batch)
    def collate_dicts(self, batch: List[Dict[str, torch.tensor]]):

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]
        print(batch[0]['seq_enc'][0])   
        post_proccessed = {
            'bind': self.collate_tensors(select_by_key(batch, 'bind'), bos=False, eos=True),
            'seq': self.collate_tensors(select_by_key(batch, 'seq'), bos=False, eos=True),
            'lens': [len(l) for l in select_by_key(batch, 'seq')],
            'bind_lens': [len(l) for l in select_by_key(batch, 'bind')], 
            'cluster': select_by_key(batch, 'cluster'),
            'seq_enc': self.collate_tensors(select_by_key(batch, 'seq_enc')) #shape (B, l, emb)
        }
        return post_proccessed