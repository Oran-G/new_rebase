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
import esm.inverse_folding
import esm
import torch_geometric
from GPUtil import showUtilization as gpu_usage
import time
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
import os
import json
import wandb
import csv
import random
from rfdiffusion.kinematics import xyz_to_t2d
from rfdiffusion.util import get_torsions
from rfdiffusion.inference.utils import Denoise
from rfdiffusion.inference.utils import process_target
from .diffusor_utils import PARAMS
class RFdict():
    def __init__(self):
        self.one_letter = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "?", "-"] #20 AA, unk, mask. this is taken from rfidffusion.chemical, and will also be used for Neuc
        self.letter_idx = {self.one_letter[i]:i for i in range(len(self.one_letter))}
        self.pad_idx = 22
        self.eos_idx = 23
        self.padding_idx = 22
    def encode(self, line):
        idx = []
        for char in line:
            if char in self.one_letter:
                idx.append(self.letter_idx[char])
            else:
                idx.append(20)
        return idx
    def pad(self, line, length):
        idx = line
        for i in range(length - len(idx)):
            idx.append(self.pad_idx)
        return idx
    def eos(self, line):
        return line.append(eos_idx)
        



def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

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
        if supervised:
            self.df = self.df.dropna()

        print("pre filter",len(self.df))

        def alpha(ids):
            return os.path.isfile(f'/vast/og2114/rebase/20220519/output/{ids}/ranked_0.pdb') and (max(json.load(open(f'/vast/og2114/rebase/20220519/output/{ids}/ranking_debug.json'))['plddts'].values()) >= plddt)
        #self.df  = self.df[self.df['id'].apply(alpha) ==True ]
        self.df = self.df[self.df['id'] != 'Csp7507ORF4224P']
        print("post filter",len(self.df))

        spl = self.split(split)
        self.data = spl[['seq','bind', 'id']].to_dict('records')
        print(len(self.data))
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.clustered_data = {}
        tmp_clust = self.df.cluster.unique()
        self.cluster_idxs =[]
        for cluster in tmp_clust:
            t = spl[spl['cluster'] == cluster][['seq','bind', 'id']].to_dict('records')
            if len(t) != 0:
                self.clustered_data[cluster]= spl[spl['cluster'] == cluster][['seq','bind', 'id']].to_dict('records')
                self.cluster_idxs.append(cluster)
        self.use_cluster=clust
        
        num = 0
        for data in self.data:
            
            if len(data['seq']) > 376:
                num += 1
        print("SEQUENCES OVER LENGTH 376: ", num)
        
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
            'xyz_27': rf.inference.utils.process_target coords output
            'mask_27': rf.inference.utils.process_target mask output
            'seq': torch.tensor sequence
        } to be post-proccessed in self.collate_dicts()

        https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/inference/utils.py#L613
        '''

    
        proccessed = process_target(f"/vast/og2114/rebase/20220519/output/{self.dataset[idx]['id']}/ranked_0.pdb")
        print(proccessed.keys())
        MAXLEN = 271
        if torch.tensor(proccessed['xyz_27']).shape[0] >= MAXLEN:
            start = random.randint(0, torch.tensor(proccessed['xyz_27']).shape[0] - (MAXLEN+1))
            end = start + MAXLEN
        else:
            start = 0
            end = -1
        #import pdb; pdb.set_trace()
        return {
            'bind':torch.tensor( self.dictionary.encode(self.dataset[idx]['bind'])),
            'xyz_27': torch.tensor(proccessed['xyz_27'])[start: end], #tensor of atomic coords
            'mask_27': torch.tensor(proccessed['mask_27'][start: end]), #tensor of true/false for if the atoms exist
            'seq': torch.tensor(proccessed['seq'][start:end]), #tensor of idx
            'idx_pdb': proccessed['pdb_idx'][start:end],

        }

    def __len__(self):
        return len(self.dataset)

    def collate_tensors(self, batch: List[torch.tensor], bos=False, eos=False):
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
        
        if eos == None:
            eos = self.dictionary.eos()

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

            # import pdb; pdb.set_trace()
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
                'xyz_27': rf.inference.utils.process_target coords output
                'mask_27': rf.inference.utils.process_target mask output
                'seq': torch.tensor sequence
                'idx_pdb': proccessed['idx_pdb'],
            },
            {
                'bind': torch.tensor (bind site)
                'xyz_27': rf.inference.utils.process_target coords output
                'mask_27': rf.inference.utils.process_target mask output
                'seq': torch.tensor sequence
                'idx_pdb': proccessed['idx_pdb'],
            },
        ]
        into a collated form:
        {
            'bind': torch.tensor (bind site)
            'bos_bind': torch.tensor (bos+bind site)
            
            'seq': torch.tensor (protein sequence)
            'xyz_27': torch.tensor (coords input to rf)
            'mask_27' torch.tensor (mask input to rf)
            'idx_pdb': list of idx_pdb in form [(chain, i) for i in len]
        }
        
        applying the padding correctly to capture different lengths
        '''

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]



        post_proccessed = {
            'bind': self.collate_tensors(select_by_key(batch, 'bind'), eos=False),
            
            'seq': self.collate_tensors(select_by_key(batch, 'seq')),
            'xyz_27': torch.stack(select_by_key(batch, 'xyz_27'), dim=0),
            'mask_27': torch.stack(select_by_key(batch, 'mask_27'), dim=0),
            'idx_pdb': select_by_key(batch, 'idx_pdb'),
            
        }
        return post_proccessed





class RebaseT5(pl.LightningModule):
    def __init__(self, cfg):
        '''
        Main class
        pl.LightningModule git - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/module.py
        '''

        super(RebaseT5, self).__init__()

        self.cfg = cfg
        try:
            self.cfg['slurm'] = str(os.environ.get('SLURM_JOB_ID'))
        except:
            pass
        self.save_hyperparameters(self.cfg)
        self.batch_size = self.hparams.model.batch_size
        print("Argument hparams: ", self.hparams)
        print('batch size', self.hparams.model.batch_size)


        
        self.ifalphabet= RFdict()
        
        '''
        used to record validation data for logging
        '''
        self.val_data = []
        print('initialized')
        
        self.params = PARAMS
        self.T = PARAMS['T']
        from rfdiffusion.inference.model_runners  import Sampler
        self.sampler = Sampler(conf=OmegaConf.load('/vast/og2114/RFdiffusion/config/inference/base.yaml'))
        self.model = self.sampler.model.train() #ROSETTAFold Model created by sampler
        self.model.to(self.device)
        from rfdiffusion.diffusion import Diffuser
        
        self.diffuser = Diffuser(T=self.T,
		    b_0=PARAMS['Bt0'], 
		    b_T=PARAMS['Bt0'], 
		    min_sigma=0.01, #IGSO3 docs say to do this 
            max_sigma=PARAMS['Bt0']+((PARAMS['BtT'] - PARAMS['Bt0'])/2), #see page 12
            min_b=PARAMS['Bt0'], 
            max_b=PARAMS['BtT'], 
            schedule_type='linear', # this gets fed into the Euclidean diffuser class —> default for that is linear 
            so3_schedule_type='linear',  # same but for IGS03 class
            so3_type='igs03', #this is literally not stored or used anywhere in the init function,  
            crd_scale=PARAMS["crd_scale"], #  I think it has to do with centering the coordinates in space to prevent data leakage as of rfdiffusion.diffusion.py line 641
            )
        



        



    def training_step(self, batch, batch_idx):
        '''
        Training step
        input:
            batch: output of EncodedFastaDatasetWrapper.collate_dicts {
            'bind': torch.tensor (bind site)
            'bos_bind': torch.tensor (bos+bind site)
            
            'seq': torch.tensor (protein sequence)
            'xyz_27': torch.tensor (coords input to rf)
            'mask_27' torch.tensor (mask input to rf)
        }

        output:
            loss for pl,
            batch size for pl
        '''


        '''step lr scheduler'''
        if self.global_step  != 0:
            self.lr_schedulers().step()
        torch.cuda.empty_cache()
        start_time  = time.time()


        '''
        TRAIN STEP FORM!!!
        1. Generate a list of random timestep "T"s from 1-100 of len(batch size) (might replace with one singular t for whole batch for simplicity)
        2. Diffuse the protein, select diffused timestep "T", and "T-1"
        3. Feed bind site through transformer NN
        4. Using transofrmer output, and diffused_to_t, use rfdiffusion model to get t-1 coords. 
        5. take loss between t-1_pred, and t-1_truth 


        DENOISING:
            50% of time we use self conditioning. how self-conditioning works, 
            there will be a place in the model input to feed in the sef conditioning. 
            self conditioning is when you run the model on x_t+1, and get x_o-pred(condition). 
            you now feed x_o-pred along with x_t to rf. this essentially I guess says what changes would be 
            made from timestep t+1 to t, and what needs to be changed now. 
            good literature on this in chen et. al. https://arxiv.org/abs/2208.04202


            If Training with self conditioning -> noise to timestep t+1, then have x_t = denoise(x_t+1) 
        '''

        
        
        t = random.randint(0, self.T)
        if random.randint(0, 1) == 0 or t == self.T:
            poses = self.diffuser.diffuse_pose(batch['xyz_27'][0][:, :14, :].to('cpu'), batch['seq'][0].to('cpu'), batch['mask_27'][0][:, :14].to('cpu'), t_list=[t])
            pose_t = poses[0][0]
        
            #from model_runnsers Sampler._preprocess mostly
            seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
            print('shape:', seq.shape)
            L = seq.shape[0]
            print('LENGTH = ', L)
            msa_masked = torch.zeros((1, 1, L, 48))
            msa_masked[:, :, :, :22] = seq[None, None]
            msa_masked[:, :, :, 22:44] = seq[None, None]
            msa_masked[:, :, 0, 46] = 1.0
            msa_masked[:, :, -1, 47] = 1.0
            msa_full = torch.zeros((1, 1, L, 25))
            msa_full[:, :, :, :22] = seq[None, None]
            msa_full[:, :, 0, 23] = 1.0
            msa_full[:, :, -1, 24] = 1.0
            t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))
            t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)
            t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
            seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
            
            alpha, _, alpha_mask, _ = get_torsions(pose_t.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
            alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
            alpha[torch.isnan(alpha)] = 1.0
            alpha = alpha.reshape(1, -1, L, 10, 2)
            alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
            alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
            idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
            seq_in = torch.zeros((L, 22))
            seq_in[:, 21] = 1.0
            seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
            seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
            mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
            print(seq_in.shape)

            pose_t = pose_t.to(batch['bind'].device)
            t1d = t1d.to(batch['bind'].device)
            t2d = t2d.to(batch['bind'].device)
            alpha_t = alpha_t.to(batch['bind'].device)
            pose_t = pose_t.to(batch['bind'].device)
            seq_in = seq_in.to(batch['bind'].device)
            seq = seq.to(batch['bind'].device)
            idx_pdb = idx_pdb.to(batch['bind'].device)
            msa_masked = msa_masked.to(batch['bind'].device)
            msa_full = msa_full.to(batch['bind'].device)

            xyz_t = torch.clone(pose_t)
            xyz_t = xyz_t[None, None]
            xyz_t = torch.cat((xyz_t[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
            print("msa_masked = ", msa_masked.shape)
            print(msa_full.shape)
            print(seq_in.shape)
            print(xyz_t.squeeze(dim=0).shape)
            print(idx_pdb.shape)
            print(t1d.shape)
            print(t2d.shape)
            print(xyz_t.shape)
            print(alpha_t.shape)
            print('MASK_27 = ', mask.shape) 
            #import pdb; pdb.set_trace()
            #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
            output = self.model(
                    msa_masked,
                    msa_full,
                    seq_in,
                    xyz_t.squeeze(dim=0),#[:, :14, :]
                    #pose_t,
                    idx_pdb,
                    t1d=t1d,
                    t2d=t2d,
                    #xyz_t=pose_t.unsqueeze(0),
                    xyz_t=xyz_t, 
                    alpha_t=alpha_t,
                    msa_prev=None,
                    pair_prev=None,
                    state_prev=None,
                    t=torch.tensor(t),
                    motif_mask=mask,
                    return_infer=True,
                    )
            _, px0 = self.sampler.allatom(torch.argmax(seq_in, dim=-1), output[2], output[4])

            alpha_0, _, alpha_mask_0, _ = get_torsions(batch['xyz_27'].reshape(-1, L, 27, 3), batch['seq'], torch.full((22, 4, 4), 0).to(batch['bind'].device), torch.full((22, 10), False, dtype=torch.bool).to(batch['bind'].device), torch.ones((22, 3, 2)).to(batch['bind'].device)) #these wierd tensors are from rfdiffusion.utils
            alpha_mask_0 = torch.logical_and(alpha_mask_0, ~torch.isnan(alpha_0[...,0]))
            alpha_0[torch.isnan(alpha_0)] = 1.0
            alpha_0 = alpha_0.reshape(1, -1, L, 10, 2)
            alpha_mask_0 = alpha_mask_0.reshape(1, -1, L, 10, 1)
            alpha_0 = torch.cat((alpha_0, alpha_mask_0), dim=-1).reshape(1, -1, L, 30)
            #import pdb; pdb.set_trace()
            #loss = lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, output[4], .99, 1, 1, t)
            loss = self.lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, torch.cat((output[4].reshape(1, -1, L, 10, 2), alpha_mask_0), dim=-1).reshape(1, -1, L, 30), .99, 1, 1, t)
       
       
       
       
       
        else:
            poses = self.diffuser.diffuse_pose(batch['xyz_27'][0][:, :14, :].to('cpu'), batch['seq'][0].to('cpu'), batch['mask_27'][0][:, :14].to('cpu'), t_list=[t, t+1])
            pose_t = poses[0][0]
            pose_t_1=poses[0][1]
            #from model_runnsers Sampler._preprocess mostly
            seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
            print('shape:', seq.shape)
            L = seq.shape[0]
            print('LENGTH = ', L)
            #import pdb; pdb.set_trace()

            msa_masked = torch.zeros((1, 1, L, 48))
            msa_masked[:, :, :, :22] = seq[None, None]
            msa_masked[:, :, :, 22:44] = seq[None, None]
            msa_masked[:, :, 0, 46] = 1.0
            msa_masked[:, :, -1, 47] = 1.0
            msa_full = torch.zeros((1, 1, L, 25))
            msa_full[:, :, :, :22] = seq[None, None]
            msa_full[:, :, 0, 23] = 1.0
            msa_full[:, :, -1, 24] = 1.0
            t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))

            t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)

            t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
            seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
            alpha, _, alpha_mask, _ = get_torsions(pose_t_1.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
            alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
            alpha[torch.isnan(alpha)] = 1.0
            alpha = alpha.reshape(1, -1, L, 10, 2)
            alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
            alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
            idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
            seq_in = torch.zeros((L, 22))
            seq_in[:, 21] = 1.0
            seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
            seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
            mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
            print(seq_in.shape)


            pose_t_1 = pose_t_1.to(batch['bind'].device)
            t1d = t1d.to(batch['bind'].device)
            t2d = t2d.to(batch['bind'].device)
            alpha_t = alpha_t.to(batch['bind'].device)
            pose_t_1 = pose_t_1.to(batch['bind'].device)
            seq_in = seq_in.to(batch['bind'].device)
            seq = seq.to(batch['bind'].device)
            idx_pdb = idx_pdb.to(batch['bind'].device)
            msa_masked = msa_masked.to(batch['bind'].device)
            msa_full = msa_full.to(batch['bind'].device)

            xyz_t_1 = torch.clone(pose_t_1)
            xyz_t_1 = xyz_t_1[None, None]
            xyz_t_1 = torch.cat((xyz_t_1[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
            print("msa_masked = ", msa_masked.shape)
            print(msa_full.shape)
            print(seq_in.shape)
            print(xyz_t_1.squeeze(dim=0).shape)
            print(idx_pdb.shape)
            print(t1d.shape)
            print(t2d.shape)
            print(xyz_t_1.shape)
            print(alpha_t.shape)
            print('MASK_27 = ', mask.shape) 
            #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
            
            with torch.no_grad():    
                output = self.model(
                        msa_masked,
                        msa_full,
                        seq_in,
                        xyz_t_1.squeeze(dim=0),#[:, :14, :]
                        #pose_t,
                        idx_pdb,
                        t1d=t1d,
                        t2d=t2d,
                        #xyz_t=pose_t.unsqueeze(0),
                        xyz_t=xyz_t_1, 
                        alpha_t=alpha_t,
                        msa_prev=None,
                        pair_prev=None,
                        state_prev=None,
                        t=torch.tensor(t+1),
                        motif_mask=mask,
                        return_infer=True,
                        )

            torch.cuda.empty_cache()
            msa_prev = output[0]
            pair_prev = output[1]
            state_prev = output[3]

            



            
            #from model_runnsers Sampler._preprocess mostly
            seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
            print('shape:', seq.shape)
            L = seq.shape[0]
            print('LENGTH = ', L)
         

            msa_masked = torch.zeros((1, 1, L, 48))
            msa_masked[:, :, :, :22] = seq[None, None]
            msa_masked[:, :, :, 22:44] = seq[None, None]
            msa_masked[:, :, 0, 46] = 1.0
            msa_masked[:, :, -1, 47] = 1.0
            msa_full = torch.zeros((1, 1, L, 25))
            msa_full[:, :, :, :22] = seq[None, None]
            msa_full[:, :, 0, 23] = 1.0
            msa_full[:, :, -1, 24] = 1.0
            t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))
            t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)

            t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
            seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
            
            alpha, _, alpha_mask, _ = get_torsions(pose_t.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
            alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
            alpha[torch.isnan(alpha)] = 1.0
            alpha = alpha.reshape(1, -1, L, 10, 2)
            alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
            alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
            idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
            seq_in = torch.zeros((L, 22))
            seq_in[:, 21] = 1.0
            seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
            seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
            mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
            print(seq_in.shape)
            #import pdb; pdb.set_trace()

            pose_t = pose_t.to(batch['bind'].device)
            t1d = t1d.to(batch['bind'].device)
            t2d = t2d.to(batch['bind'].device)
            alpha_t = alpha_t.to(batch['bind'].device)
            pose_t = pose_t.to(batch['bind'].device)
            seq_in = seq_in.to(batch['bind'].device)
            seq = seq.to(batch['bind'].device)
            idx_pdb = idx_pdb.to(batch['bind'].device)
            msa_masked = msa_masked.to(batch['bind'].device)
            msa_full = msa_full.to(batch['bind'].device)

            xyz_t = torch.clone(pose_t)
            xyz_t = xyz_t[None, None]
            xyz_t = torch.cat((xyz_t[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
            print("msa_masked = ", msa_masked.shape)
            print(msa_full.shape)
            print(seq_in.shape)
            print(xyz_t.squeeze(dim=0).shape)
            print(idx_pdb.shape)
            print(t1d.shape)
            print(t2d.shape)
            print(xyz_t.shape)
            print(alpha_t.shape)
            print('MASK_27 = ', mask.shape) 
            #import pdb; pdb.set_trace()
            #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
            output = self.model(
                    msa_masked,
                    msa_full,
                    seq_in,
                    xyz_t.squeeze(dim=0),#[:, :14, :]
                    #pose_t,
                    idx_pdb,
                    t1d=t1d,
                    t2d=t2d,
                    #xyz_t=pose_t.unsqueeze(0),
                    xyz_t=xyz_t, 
                    alpha_t=alpha_t,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    t=torch.tensor(t),
                    motif_mask=mask,
                    return_infer=True,
                    )
            _, px0 = self.sampler.allatom(torch.argmax(seq_in, dim=-1), output[2], output[4])
            #import pdb; pdb.set_trace()
            alpha_0, _, alpha_mask_0, _ = get_torsions(batch['xyz_27'].reshape(-1, L, 27, 3), batch['seq'], torch.full((22, 4, 4), 0).to(batch['bind'].device), torch.full((22, 10), False, dtype=torch.bool).to(batch['bind'].device), torch.ones((22, 3, 2)).to(batch['bind'].device)) #these wierd tensors are from rfdiffusion.utils
            alpha_mask_0 = torch.logical_and(alpha_mask_0, ~torch.isnan(alpha_0[...,0]))
            alpha_0[torch.isnan(alpha_0)] = 1.0
            alpha_0 = alpha_0.reshape(1, -1, L, 10, 2)
            alpha_mask_0 = alpha_mask_0.reshape(1, -1, L, 10, 1)
            alpha_0 = torch.cat((alpha_0, alpha_mask_0), dim=-1).reshape(1, -1, L, 30)

            #import pdb; pdb.set_trace()
            #loss = lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, output[4], .99, 1, 1, t)
            loss = self.lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, torch.cat((output[4].reshape(1, -1, L, 10, 2), alpha_mask_0), dim=-1).reshape(1, -1, L, 30), .99, 1, 1, t)













  
       
        self.log('train_loss', float(loss.item())/(.99**(self.T - t)), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('train_acc',  , on_step=True, on_epoch=True, prog_bar=False, logger=True) #accuracy using torchmetrics accuracy
        #self.log('length', ,  on_step=True,  logger=True) # length of prediction
        #self.log('train_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True) # step time

        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()

        start_time = time.time()

        torch.cuda.empty_cache()
        with torch.no_grad():
            t = random.randint(0, self.T)
            if random.randint(0, 1) == 0 or t == self.T:
                poses = self.diffuser.diffuse_pose(batch['xyz_27'][0][:, :14, :].to('cpu'), batch['seq'][0].to('cpu'), batch['mask_27'][0][:, :14].to('cpu'), t_list=[t])
                pose_t = poses[0][0]
            
                #from model_runnsers Sampler._preprocess mostly
                seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
                print('shape:', seq.shape)
                #import pdb; pdb.set_trace()
                L = seq.shape[0]
                print('LENGTH = ', L)


                msa_masked = torch.zeros((1, 1, L, 48))
                msa_masked[:, :, :, :22] = seq[None, None]
                msa_masked[:, :, :, 22:44] = seq[None, None]
                msa_masked[:, :, 0, 46] = 1.0
                msa_masked[:, :, -1, 47] = 1.0
                msa_full = torch.zeros((1, 1, L, 25))
                msa_full[:, :, :, :22] = seq[None, None]
                msa_full[:, :, 0, 23] = 1.0
                msa_full[:, :, -1, 24] = 1.0
                t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))
                #import pdb; pdb.set_trace()
                t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)
                #t1d = torch.cat((t1d, torch.zeros((1, 1, L, 5))), dim=-1)
                t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
                seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
                
                alpha, _, alpha_mask, _ = get_torsions(pose_t.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 1.0
                alpha = alpha.reshape(1, -1, L, 10, 2)
                alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
                idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
                seq_in = torch.zeros((L, 22))
                seq_in[:, 21] = 1.0
                seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
                seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
                mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
                print(seq_in.shape)
                #import pdb; pdb.set_trace()

                pose_t = pose_t.to(batch['bind'].device)
                t1d = t1d.to(batch['bind'].device)
                t2d = t2d.to(batch['bind'].device)
                alpha_t = alpha_t.to(batch['bind'].device)
                pose_t = pose_t.to(batch['bind'].device)
                seq_in = seq_in.to(batch['bind'].device)
                seq = seq.to(batch['bind'].device)
                idx_pdb = idx_pdb.to(batch['bind'].device)
                msa_masked = msa_masked.to(batch['bind'].device)
                msa_full = msa_full.to(batch['bind'].device)

                xyz_t = torch.clone(pose_t)
                xyz_t = xyz_t[None, None]
                xyz_t = torch.cat((xyz_t[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
                print("msa_masked = ", msa_masked.shape)
                print(msa_full.shape)
                print(seq_in.shape)
                print(xyz_t.squeeze(dim=0).shape)
                print(idx_pdb.shape)
                print(t1d.shape)
                print(t2d.shape)
                print(xyz_t.shape)
                print(alpha_t.shape)
                print('MASK_27 = ', mask.shape) 
                #import pdb; pdb.set_trace()
                #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                output = self.model(
                        msa_masked,
                        msa_full,
                        seq_in,
                        xyz_t.squeeze(dim=0),#[:, :14, :]
                        #pose_t,
                        idx_pdb,
                        t1d=t1d,
                        t2d=t2d,
                        #xyz_t=pose_t.unsqueeze(0),
                        xyz_t=xyz_t, 
                        alpha_t=alpha_t,
                        msa_prev=None,
                        pair_prev=None,
                        state_prev=None,
                        t=torch.tensor(t),
                        motif_mask=mask,
                        return_infer=True,
                        )
                _, px0 = self.sampler.allatom(torch.argmax(seq_in, dim=-1), output[2], output[4])

                alpha_0, _, alpha_mask_0, _ = get_torsions(batch['xyz_27'].reshape(-1, L, 27, 3), batch['seq'], torch.full((22, 4, 4), 0).to(batch['bind'].device), torch.full((22, 10), False, dtype=torch.bool).to(batch['bind'].device), torch.ones((22, 3, 2)).to(batch['bind'].device)) #these wierd tensors are from rfdiffusion.utils
                alpha_mask_0 = torch.logical_and(alpha_mask_0, ~torch.isnan(alpha_0[...,0]))
                alpha_0[torch.isnan(alpha_0)] = 1.0
                alpha_0 = alpha_0.reshape(1, -1, L, 10, 2)
                alpha_mask_0 = alpha_mask_0.reshape(1, -1, L, 10, 1)
                alpha_0 = torch.cat((alpha_0, alpha_mask_0), dim=-1).reshape(1, -1, L, 30)
                #import pdb; pdb.set_trace()
                #loss = lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, output[4], .99, 1, 1, t)
                loss = self.lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, torch.cat((output[4].reshape(1, -1, L, 10, 2), alpha_mask_0), dim=-1).reshape(1, -1, L, 30), .99, 1, 1, t)
        
        
        
        
        
            else:
                poses = self.diffuser.diffuse_pose(batch['xyz_27'][0][:, :14, :].to('cpu'), batch['seq'][0].to('cpu'), batch['mask_27'][0][:, :14].to('cpu'), t_list=[t, t+1])
                pose_t = poses[0][0]
                pose_t_1=poses[0][1]
                #from model_runnsers Sampler._preprocess mostly
                seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
                print('shape:', seq.shape)
                #import pdb; pdb.set_trace()
                L = seq.shape[0]
                print('LENGTH = ', L)
                #samp_init = self.sampler.sample_init()
                #pre = self.sampler._preprocess(torch.zeros((L, 22)), pose_t_1, t)
                #print(pre)
                #import pdb; pdb.set_trace()

                msa_masked = torch.zeros((1, 1, L, 48))
                msa_masked[:, :, :, :22] = seq[None, None]
                msa_masked[:, :, :, 22:44] = seq[None, None]
                msa_masked[:, :, 0, 46] = 1.0
                msa_masked[:, :, -1, 47] = 1.0
                msa_full = torch.zeros((1, 1, L, 25))
                msa_full[:, :, :, :22] = seq[None, None]
                msa_full[:, :, 0, 23] = 1.0
                msa_full[:, :, -1, 24] = 1.0
                t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))
                #import pdb; pdb.set_trace()
                t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)
                #t1d = torch.cat((t1d, torch.zeros((1, 1, L, 5))), dim=-1)
                t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
                seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
                alpha, _, alpha_mask, _ = get_torsions(pose_t_1.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 1.0
                alpha = alpha.reshape(1, -1, L, 10, 2)
                alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
                idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
                seq_in = torch.zeros((L, 22))
                seq_in[:, 21] = 1.0
                seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
                seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
                mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
                print(seq_in.shape)
                #import pdb; pdb.set_trace()

                pose_t_1 = pose_t_1.to(batch['bind'].device)
                t1d = t1d.to(batch['bind'].device)
                t2d = t2d.to(batch['bind'].device)
                alpha_t = alpha_t.to(batch['bind'].device)
                pose_t_1 = pose_t_1.to(batch['bind'].device)
                seq_in = seq_in.to(batch['bind'].device)
                seq = seq.to(batch['bind'].device)
                idx_pdb = idx_pdb.to(batch['bind'].device)
                msa_masked = msa_masked.to(batch['bind'].device)
                msa_full = msa_full.to(batch['bind'].device)

                xyz_t_1 = torch.clone(pose_t_1)
                xyz_t_1 = xyz_t_1[None, None]
                xyz_t_1 = torch.cat((xyz_t_1[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
                print("msa_masked = ", msa_masked.shape)
                print(msa_full.shape)
                print(seq_in.shape)
                print(xyz_t_1.squeeze(dim=0).shape)
                print(idx_pdb.shape)
                print(t1d.shape)
                print(t2d.shape)
                print(xyz_t_1.shape)
                print(alpha_t.shape)
                print('MASK_27 = ', mask.shape) 
                #import pdb; pdb.set_trace()
                #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                
                with torch.no_grad():    
                    output = self.model(
                            msa_masked,
                            msa_full,
                            seq_in,
                            xyz_t_1.squeeze(dim=0),#[:, :14, :]
                            #pose_t,
                            idx_pdb,
                            t1d=t1d,
                            t2d=t2d,
                            #xyz_t=pose_t.unsqueeze(0),
                            xyz_t=xyz_t_1, 
                            alpha_t=alpha_t,
                            msa_prev=None,
                            pair_prev=None,
                            state_prev=None,
                            t=torch.tensor(t+1),
                            motif_mask=mask,
                            return_infer=True,
                            )

                torch.cuda.empty_cache()
                msa_prev = output[0]
                pair_prev = output[1]
                state_prev = output[3]

                



                
                #from model_runnsers Sampler._preprocess mostly
                seq = torch.nn.functional.one_hot(batch['seq'][0], num_classes=22)
                print('shape:', seq.shape)
                L = seq.shape[0]
                print('LENGTH = ', L)
            

                msa_masked = torch.zeros((1, 1, L, 48))
                msa_masked[:, :, :, :22] = seq[None, None]
                msa_masked[:, :, :, 22:44] = seq[None, None]
                msa_masked[:, :, 0, 46] = 1.0
                msa_masked[:, :, -1, 47] = 1.0
                msa_full = torch.zeros((1, 1, L, 25))
                msa_full[:, :, :, :22] = seq[None, None]
                msa_full[:, :, 0, 23] = 1.0
                msa_full[:, :, -1, 24] = 1.0
                t1d = torch.zeros((1, 1, L - batch['bind'].shape[1], 22))
                t1d = torch.cat((torch.unsqueeze(torch.unsqueeze(torch.nn.functional.one_hot(batch['bind'][0], num_classes=22), 0), 0).to('cpu'), t1d), dim=2)
                #t1d = torch.cat((t1d, torch.zeros((1, 1, L, 5))), dim=-1)
                t2d = xyz_to_t2d(torch.unsqueeze(torch.unsqueeze(pose_t, 0), 0))
                seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
                
                alpha, _, alpha_mask, _ = get_torsions(pose_t.reshape(-1, L, 27, 3), seq_tmp, torch.full((22, 4, 4), 0), torch.full((22, 10), False, dtype=torch.bool), torch.ones((22, 3, 2))) #these wierd tensors are from rfdiffusion.utils
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 1.0
                alpha = alpha.reshape(1, -1, L, 10, 2)
                alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
                idx_pdb =torch.tensor([batch['idx_pdb'][0][i][1]-1 for i in range(len(batch['idx_pdb'][0]))]).unsqueeze(0)
                seq_in = torch.zeros((L, 22))
                seq_in[:, 21] = 1.0
                seq_in = torch.unsqueeze(torch.tensor([21 for i in range(L)]), dim=0)
                seq_in = torch.nn.functional.one_hot(seq_in, num_classes=22).float()
                mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
                print(seq_in.shape)
                #import pdb; pdb.set_trace()

                pose_t = pose_t.to(batch['bind'].device)
                t1d = t1d.to(batch['bind'].device)
                t2d = t2d.to(batch['bind'].device)
                alpha_t = alpha_t.to(batch['bind'].device)
                pose_t = pose_t.to(batch['bind'].device)
                seq_in = seq_in.to(batch['bind'].device)
                seq = seq.to(batch['bind'].device)
                idx_pdb = idx_pdb.to(batch['bind'].device)
                msa_masked = msa_masked.to(batch['bind'].device)
                msa_full = msa_full.to(batch['bind'].device)

                xyz_t = torch.clone(pose_t)
                xyz_t = xyz_t[None, None]
                xyz_t = torch.cat((xyz_t[:, :14, :], torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3)
                print("msa_masked = ", msa_masked.shape)
                print(msa_full.shape)
                print(seq_in.shape)
                print(xyz_t.squeeze(dim=0).shape)
                print(idx_pdb.shape)
                print(t1d.shape)
                print(t2d.shape)
                print(xyz_t.shape)
                print(alpha_t.shape)
                print('MASK_27 = ', mask.shape) 
                #import pdb; pdb.set_trace()
                #msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                output = self.model(
                        msa_masked,
                        msa_full,
                        seq_in,
                        xyz_t.squeeze(dim=0),#[:, :14, :]
                        #pose_t,
                        idx_pdb,
                        t1d=t1d,
                        t2d=t2d,
                        #xyz_t=pose_t.unsqueeze(0),
                        xyz_t=xyz_t, 
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        t=torch.tensor(t),
                        motif_mask=mask,
                        return_infer=True,
                        )
                _, px0 = self.sampler.allatom(torch.argmax(seq_in, dim=-1), output[2], output[4])
                #import pdb; pdb.set_trace()
                alpha_0, _, alpha_mask_0, _ = get_torsions(batch['xyz_27'].reshape(-1, L, 27, 3), batch['seq'], torch.full((22, 4, 4), 0).to(batch['bind'].device), torch.full((22, 10), False, dtype=torch.bool).to(batch['bind'].device), torch.ones((22, 3, 2)).to(batch['bind'].device)) #these wierd tensors are from rfdiffusion.utils
                alpha_mask_0 = torch.logical_and(alpha_mask_0, ~torch.isnan(alpha_0[...,0]))
                alpha_0[torch.isnan(alpha_0)] = 1.0
                alpha_0 = alpha_0.reshape(1, -1, L, 10, 2)
                alpha_mask_0 = alpha_mask_0.reshape(1, -1, L, 10, 1)
                alpha_0 = torch.cat((alpha_0, alpha_mask_0), dim=-1).reshape(1, -1, L, 30)

                #import pdb; pdb.set_trace()
                #loss = lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, output[4], .99, 1, 1, t)
                loss = self.lframe(px0[:, :, :14, :], batch['xyz_27'][:, :, :14, :], alpha_0, torch.cat((output[4].reshape(1, -1, L, 10, 2), alpha_mask_0), dim=-1).reshape(1, -1, L, 30), .99, 1, 1, t)


        
        self.log('val_loss', float(loss.item())/(.99**(self.T - t)), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        #self.log('val_acc', float(self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2), batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        if str(self.cfg.model.seq_identity) == '0.9':
            cs = f'{self.cfg.io.final}-9'
        elif str(self.cfg.model.seq_identity) == '0.7':
            cs = f'{self.cfg.io.final}-7'
        else:
            cs = self.cfg.io.final
        if self.cfg.model.dna_clust == True:
            cs = self.cfg.io.dnafinal
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(cs, 'train', clust=self.cfg.model.sample_by_cluster),

            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )


        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collater), device=self.device)
        #dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collater)
        return dataloader
    
    def val_dataloader(self):
        if str(self.cfg.model.seq_identity)== '0.9':
            cs = f'{self.cfg.io.final}-9'
        elif str(self.cfg.model.seq_identity) == '0.7':
            cs = f'{self.cfg.io.final}-7'
        else:
            cs = self.cfg.io.final

        if self.cfg.model.dna_clust == True:
            cs = self.cfg.io.dnafinal
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(cs, 'val', clust=self.cfg.model.sample_by_cluster),
            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        self.dataset = dataset
        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collater), device=self.device)
        return dataloader
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                #{'params': self.ifmodel.parameters(), 'lr': float(self.hparams.model.lr)/5},
                {'params': self.model.parameters()}],
            lr=float(self.hparams.model.lr))
        # return opt

        # figure out reaosnable number of total steps
        # 100k steps
        # 4% warmup to peak lr
        # linear decay after that

        if self.hparams.model.scheduler:
            return {
                'optimizer': opt,
                'lr_scheduler': get_polynomial_decay_schedule_with_warmup(
                    optimizer=opt,
                    num_training_steps=300000,
                    num_warmup_steps=100, #was 4000
                    power=self.hparams.model.lrpower,
                )
            }
        else:
            return opt


    def decode(self, seq):
        '''
        decode tokens to  string
        input -> [list] type token representation of sequence to be decoded
        output -> [string] of sequence decoded
        '''
        newseq = ''
        for tok in seq:
            newseq += str(self.ifalphabet.get_tok(tok))
        return newseq

    def loss_trans(self, x, x_pred, dclamp=10):
        distances = torch.abs(x - x_pred)
        distances = torch.clamp(distances, max=dclamp)
        distances = torch.square(distances)
        return torch.mean(distances)
    def loss_rot(self, alpha, alpha_pred):
        l = torch.abs(alpha - alpha_pred)
        l = torch.square(l)
        return torch.mean(l)
    def lframe(self, x_pred, x, alpha, alpha_pred, decay, wtrans, wrot, T):
        return torch.tensor(decay**(self.T - T), device=x.device) * (torch.tensor(wtrans, device=x.device) * self.loss_trans(x, x_pred) + torch.tensor(wrot, device = alpha.device) * self.loss_rot(alpha, alpha_pred))
    





@hydra.main(version_base="1.2.0",config_path='/vast/og2114/new_rebase/configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = RebaseT5(cfg)
    gpu = cfg.model.gpu
    cfg = model.hparams
    cfg.model.gpu = gpu
    try:
        os.mkdir(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}")
    except: 
        pass
    try:
        os.mkdir(f"/vast/og2114/rebase/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs")
    except: 
        pass
    wandb.init(settings=wandb.Settings(start_method='thread', code_dir="."))
    wandb_logger = WandbLogger(project="Focus",save_dir=cfg.io.wandb_dir)
    wandb_logger.watch(model)
    wandb.save(os.path.abspath(__file__))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", filename=f'{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True,save_top_k=5)
    acc_callback = ModelCheckpoint(monitor="val_acc_epoch", filename=f'acc-{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True, save_top_k=5)
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print(model.batch_size)
    print('tune: ')
    model.batch_size = 1

    print(int(max(1, cfg.model.batch_size/model.batch_size)))
    trainer = pl.Trainer(
        gpus=-1,
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            acc_callback,
            #swa_callback,
            ],
        # check_val_every_n_epoch=1000,
        # max_epochs=cfg.model.max_epochs,
        default_root_dir=cfg.io.checkpoints,
        #accumulate_grad_batches=8),
        accumulate_grad_batches=4,
        precision=cfg.model.precision,
        strategy='ddp',
        #strategy='cpu',
        log_every_n_steps=5,
        progress_bar_refresh_rate=10,
        max_epochs=-1,
        #limit_train_batches=.1,
        auto_scale_batch_size="power",
        gradient_clip_val=0.3,

        )
    print('ready to train!')
    trainer.fit(model)
    model = model.to(torch.device("cuda:0"))
    trainer.test(model, dataloaders=model.val_dataloader())
    import csv
    dictionaries=model.val_data
    keys = dictionaries[0].keys()
    a_file = open(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/final.csv", "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(dictionaries)
    a_file.close()


if __name__ == '__main__':
    main()


