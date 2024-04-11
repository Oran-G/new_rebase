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
        self.data = self.split(split)[['id', 'seq', 'bind', 'cluster']].to_dict('records')
    
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.data =self.data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def split(self, split):
        if split.lower() == 'train':
            return self.df[self.df['split'] == 0]
        elif split.lower() == 'val':
            return self.df[self.df['split'] == 1]
        elif split.lower() == 'test':
            return self.df[self.df['split'] == 2]



class EmbeddedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EmbeddedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, model, embed_path, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens.   will usually have eos already applied,
        but won't have bos. Hence the defaults here.
        '''
        super().__init__(dataset)
        self.dictionary = dictionary
        self.apply_bos = apply_bos
        self.apply_eos = apply_eos
        self.model_name = model
        self.embed_path = embed_path
        #check if f'{embed_path}/{model_name}/' exists as a directory and is populated
        if not os.path.isdir(f'{embed_path}/{model_name}/') or  len(os.listdir(f'{embed_path}/{model_name}/')) == 0:        
            os.system(f'python3 fasta_preprocess.py --model_name [{model_name}]')

    def __getitem__(self, idx):
        # desc, seq = self.dataset[idx]
        return {
            'id': self.dataset[idx]['id']
            'seq': self.dictionary.encode_line(self.dataset[new_idx]['seq'].replace(' ', ''), line_tokenizer=list, append_eos=False, add_if_not_exist=False).long(),
            'bind': self.dictionary.encode_line(self.dataset[new_idx]['bind'], line_tokenizer=list, append_eos=False, add_if_not_exist=False).long(),
            'cluster': self.dataset[new_idx]['cluster'],
            'embedding':torch.load(f'{self.embed_path}/{self.model_name}/{self.dataset[new_idx]["id"]}.pt')
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
            'embedding': self.collate_tensors(select_by_key(batch, 'embedding')),
            'cluster': list(select_by_key(batch, 'cluster')),
            'id': list(select_by_key(batch, 'id')),
        }
            
        
class InlineDictionary(Dictionary):
    @classmethod
    def from_list(cls, lst: List[str]):
        d = cls()
        for idx, word in enumerate(lst):
            count = len(lst) - idx
            d.add_symbol(word, n=count, overwrite=False)
        return d
