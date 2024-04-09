import torch
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization, neucleotides
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
import torch

import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path, split, split_seed=42, supervised=True):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # print(self.df)
        # print(self.df['seq'][0])
        if supervised:
            self.df = self.df.dropna()
        # print(self.df['seq'].apply(len).mean())
        # quit()
        def cat(x):
            return (x[:1023] if len(x) > 1024 else x)
        self.df['seq'] = self.df['seq'].apply(cat)
        self.data = self.split(split)[['seq', 'bind', 'cluster']].to_dict('records')
    
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.data =self.data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def split(self, split):
        if split.lower() == 'train':
            # print(len(self.df[0:int(.7*len(self.df))]))
            # return self.df[0:int(.7*len(self.df))]
            return self.df[self.df['split'] == 0]
        elif split.lower() == 'val':
            # print(len(self.df[int(.7*len(self.df)):int(.85*len(self.df))]))
            
            # return self.df[int(.7*len(self.df)):int(.85*len(self.df))]
            return self.df[self.df['split'] == 1]
        elif split.lower() == 'test':
            
            # return self.df[int(.85*len(self.df)):]
            return self.df[self.df['split'] == 2]



class SupervisedRebaseDataset(BaseWrapperDataset):
    '''
    Filters a rebased dataset for entries that have supervised labels
    '''
    def __init__(self, dataset: FastaDataset):
        super().__init__(dataset)
        # print(len(dataset))
        self.filtered_indices = []

        # example desc: ['>AacAA1ORF2951P', 'GATATC', '280', 'aa']
        self.dna_element = 1 # element in desc corresponding to the DNA

        def encodes_as_dna(s: str):
            for c in s:
                if c not in list(neucleotides.keys()):
                    return False
            return True

        # filter indicies which don't have supervised labels
        for idx, (desc, seq) in enumerate(dataset):
            # if len(desc.split()) == 4 and encodes_as_dna(desc.split()[self.dna_element]):
            if len(desc.split(' ')) >= 2 and encodes_as_dna(desc.split(' ')[self.dna_element]):
                self.filtered_indices.append(idx)
        # print(len(self.dataset[0]))
        # print(self.dataset[0])
        # print('size:', len(self.filtered_indices))

    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # translate to our filtered indices
        new_idx = self.filtered_indices[idx]
        desc, seq = self.dataset[new_idx]
        try:
            return {
                'seq': self.dataset[new_idx][1].replace(' ', ''),
                'bind': self.dataset[new_idx][0].split(' ')[self.dna_element],
                'cluster': self.dataset[new_idx][2],
                'seq_len': len(self.dataset[new_idx][1].replace(' ', '')),
                'bind_len': len(self.dataset[new_idx][0].split(' ')[self.dna_element]),
            }     
        except IndexError:
            # print(new_idx)
            # print({
            #     'protein': self.dataset[new_idx][1].replace(' ', ''),
            #     'dna': self.dataset[new_idx][0].split(' ')[self.dna_element]
            # })
            return {
                'seq': self.dataset[new_idx][1].replace(' ', ''),
                'bind': self.dataset[new_idx][0].split(' ')[self.dna_element],
                'cluster': self.dataset[new_idx][2],
                'seq_len': len(self.dataset[new_idx][1].replace(' ', '')),
                'bind_len': len(self.dataset[new_idx][0].split(' ')[self.dna_element])
            }
            # quit()
            


class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens.   will usually have eos already applied,
        but won't have bos. Hence the defaults here.
        '''
        super().__init__(dataset)
        self.dictionary = dictionary
        self.apply_bos = apply_bos
        self.apply_eos = apply_eos

    def __getitem__(self, idx):
        # desc, seq = self.dataset[idx]

        return {
            k: self.dictionary.encode_line(v, line_tokenizer=list, append_eos=False, add_if_not_exist=False).long()
            for k, v in self.dataset[idx].items()
        }
    def __len__(self):
        return len(self.dataset)
    def collate_tensors(self, batch: List[torch.tensor]):
        batch_size = len(batch)
        max_len = max(el.size(0) for el in batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.apply_bos) + int(self.apply_eos) # eos and bos
            ),
            dtype=torch.int64,
        ).fill_(self.dictionary.pad())

        if self.apply_bos:
            tokens[:, 0] = self.dictionary.bos()

        for idx, el in enumerate(batch):
            tokens[idx, int(self.apply_bos):(el.size(0) + int(self.apply_bos))] = el

            # import pdb; pdb.set_trace()
            if self.apply_eos:
                tokens[idx, el.size(0) + int(self.apply_bos)] = self.dictionary.eos()
        
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
                'key1': torch.tensor,
                'key2': torch.tensor
            },
            {
                'key1': torch.tensor,
                'key2': torch.tensor
            },
        ]
        into a collated form:
        {
            'key1': torch.tensor,
            'key2': torch.tensor,
        }
        applying the padding correctly to capture different lengths
        '''

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]

        return {
            'seq': self.collate_tensors(select_by_key(batch, 'seq')),
            'bind': self.collate_tensors(select_by_key(batch, 'bind')),
            'cluster': list(select_by_key(batch, 'cluster')),
            'seq_len': list(select_by_key(batch, 'seq_len')),
            'bind_len': list(select_by_key(batch, 'bind_len'))
        }
            
        
class InlineDictionary(Dictionary):
    @classmethod
    def from_list(cls, lst: List[str]):
        d = cls()
        for idx, word in enumerate(lst):
            count = len(lst) - idx
            d.add_symbol(word, n=count, overwrite=False)
        return d


class EncoderDataset(Dataset):
    def __init__(self, dataset, batch_size, device, path, cluster=True, eos=True):

        super().__init__()
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        #check if file exists at path, if so load it, if not create it
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
            return
        else:
            self.dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater), device=self.device)
            self.data = []
            self.eos = eos
            print(f'creating embeddings saving to {path}')
            self.650m, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.3b, _ = esm.pretrained.esm2_t36_3B_UR50D()
            self.150m, _ = esm.pretrained.esm2_t30_150M_UR50D()
            
            for step, batch in enumerate(self.dataloader):
                with torch.no_grad()):
                    b3 = self.3b(batch['seq'], repr_layers=[36])['representations'][-1]
                    m650 = self.650m(batch['seq'], repr_layers=[33])['representations'][-1]
                    m150 = self.150m(batch['seq'], repr_layers=[30])['representations'][-1]
                for i in range(len(batch['seq'].shape[0])):
                    self.data.append({
                        'seq': batch['seq'][i][int(self.eos):batch['seq_len'][i]+int(self.eos)],
                        'bind': batch['bind'][i][int(self.eos):batch['bind_len'][i]+int(self.eos)],
                        '3b_enc': b3[i, :batch['seq_len'][i], :], 
                        '650m_enc': m650[i, :batch['seq_len'][i], :],
                        '150m_enc': m150[i, :batch['seq_len'][i], :],
                        'cluster': batch['cluster'][i]
                    })
                #augment self.data from form list[dict[..., cluster]] to list[list`dict[..., cluster]]], where the inner list is a list of dicts with the same cluster
                
            #save self.data to path and create a function to load it back to self.data
            with open(path, 'wb') as f:
                pickle.dump(self.data, f)
            self.path = path
        if cluster:
            self.clustered_data = [list(group) for key, group in itertools.groupby(self.data, lambda x: x['cluster'])]
    def __len__(self):
        if cluster:
            return len(self.clustered_data)
        return len(self.data)
    def __getitem__(self, idx):
        if cluster:
            return self.clustered_data[idx][self.cluster_idxs[idx]][random.randint(0, (len(self.clustered_data[idx])-1))]
        return self.data[idx]
    def collate_tensors(self, batch: List[torch.tensor], bos=None, eos=None):
        if bos == None:
            bos = self.apply_bos
        if eos == None:
            eos = self.apply_eos
        
        batch_size = len(batch)
        beos = int(bos) + int(eos)
        max_shape = [max(el.size(i) for el in batch) + beos for i in range(len(batch[0].shape))]
        tokens = torch.empty(
            (
                batch_size, 
                *max_shape
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

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]
        
        post_proccessed = {
            'bind': self.collate_tensors(select_by_key(batch, 'bind'), bos=False, eos=True),
            'seq': self.collate_tensors(select_by_key(batch, 'seq'), bos=False, eos=True),
            'lens': [len(l) for l in select_by_key(batch, 'seq')],
            'bind_lens': [len(l) for l in select_by_key(batch, 'bind')], 
            'cluster': select_by_key(batch, 'cluster'),
            '650m_enc': self.collate_tensors(select_by_key(batch, '650m_enc')) #shape (B, l, emb)
            '150m_enc': self.collate_tensors(select_by_key(batch, '150m_enc')) #shape (B, l, emb)
            '3b_enc': self.collate_tensors(select_by_key(batch, '3b_enc')) #shape (B, l, emb)
        }
        return post_proccessed