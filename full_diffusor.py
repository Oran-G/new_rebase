import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, BertGenerationConfig, BertGenerationDecoder
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
import esm
import esm.inverse_folding
from omegaconf import DictConfig, OmegaConf
import hydra
from GPUtil import showUtilization as gpu_usage
from typing import List, Dict
import pandas as pd
from pandas import DataFrame as df
import time
import os
import json
import wandb
import csv
import random

from rfdiffusion.kinematics import xyz_to_t2d
from rfdiffusion.util import get_torsions, make_frame, torsion_indices, torsion_can_flip, reference_angles
from rfdiffusion.inference.utils import Denoise, process_target
from rfdiffusion.chemical import aa2long, aa2longalt, torsions, ideal_coords
from rfdiffusion.inference.model_runners  import Sampler
from rfdiffusion.diffusion import Diffuser

import diffusor_utils
        




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


        #RF encoding for sequence
        self.rfdict= diffusor_utils.RFdict()
        
        #used to record validation data for logging - legacy
        self.val_data = []
        print('initialized')

        # RFdiffusion model stuff
        self.params = diffusor_utils.PARAMS
        self.T = self.params['T']
        #  New config that should load the "Complex_Fold_base_ckpt.pt" model. See line 87 of model_runners
        self.sampler = Sampler(conf=OmegaConf.load('/vast/og2114/RFdiffusion/config/finetune/neuc.yaml')) 
        # ROSETTAFold Model created by sampler. Should be from "Complex_Fold_base_ckpt.pt"
        self.model = self.sampler.model.train().to(self.device)
        # alpha creation dictionaries
        self.torsion_indices, self.torsion_can_flip, self.torsion_ref_angles = torsion_indices, torsion_can_flip, reference_angles
        # Create a diffusor in line with the paper
        self.diffuser = Diffuser(T=self.T,
            b_0=self.params['Bt0'], 
            b_T=self.params['Bt0'], 
            min_sigma=0.01, #IGSO3 docs say to do this 
            max_sigma=self.params['Bt0']+((self.params['BtT'] - self.params['Bt0'])/2), #see page 12
            min_b=self.params['Bt0'], 
            max_b=self.params['BtT'], 
            schedule_type='linear', # this gets fed into the Euclidean diffuser class —> default for that is linear 
            so3_schedule_type='linear',  # same but for IGS03 class
            so3_type='igs03', #this is literally not stored or used anywhere in the init function,  
            crd_scale=self.params["crd_scale"], #  I think it has to do with centering the coordinates in space to prevent data leakage as of rfdiffusion.diffusion.py line 641
            cache_dir = '/vast/og2114/RFdiffusion/schedules/',
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



        
        
        t_global = random.randint(0, self.T) # T
        
        if random.randint(0, 1) == 0 or t_global == self.T: # No self conditioning
            loss, _ = self.ministep(batch, t_global)
        else: # Self conditioning
            with torch.no_grad():
                _, x_prev = self.ministep(batch, t_gloabl + 1)
            torch.cuda.empty_cache()
            loss, _ = self.ministep(batch, t_global, x_prev)
        self.log('train_loss', float(loss.item())/(.99**(self.T - t)), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()

        start_time = time.time()

        torch.cuda.empty_cache()
        with torch.no_grad():
            t_global = random.randint(0, self.T)
            if random.randint(0, 1) == 0 or t_global == self.T:
                loss, _ = self.ministep(batch, t_global)
            else:
                _, x_prev = self.ministep(batch, t_global + 1)
                torch.cuda.empty_cache()
                loss, _ = self.ministep(batch, t_global, x_prev)
                
        self.log('val_loss', float(loss.item())/(.99**(self.T - t)), on_step=True, on_epoch=True, prog_bar=False, logger=True)
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
        dataset = diffusor_utils.EncodedFastaDatasetWrapper(
            diffusor_utils.CSVDataset(cs, 'train', clust=self.cfg.model.sample_by_cluster),

            self.rfdict,
            apply_eos=False,
            apply_bos=False,
        )


        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collater), device=self.device)
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
        dataset = diffusor_utils.EncodedFastaDatasetWrapper(
            diffusor_utils.CSVDataset(cs, 'val', clust=self.cfg.model.sample_by_cluster),
            self.rfdict,
            apply_eos=False,
            apply_bos=False,
        )
        self.dataset = dataset
        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collater), device=self.device)
        return dataloader 
    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                {'params': self.model.parameters()}],
            lr=float(self.hparams.model.lr))

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

    def ministep(self, batch, t, xyz_0_prev = None):
        '''
        RF STEP FORM!!!
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


            If step with self conditioning -> noise to timestep t+1, then have x_t = denoise(x_t+1) 
        '''
        '''    
                    DATA PREPROCCESSING 
        From model_runnsrs Sampler._preprocess mostly
        '''
        # Generate diffused coordinates from 0 -> t
        poses, xyz_27 = self.diffuser.diffuse_pose(batch['xyz_27'][0][:, :14, :].to('cpu'), batch['seq'][0].to('cpu'), None)
        pose_t = poses[t].unsqueeze(0).to(batch['xyz_27'].device)
        xyz_27 = xyz_27.to(batch['xyz_27'].device)
        #import pdb; pdb.set_trace() 


        # Protein sequance stuff
        L = batch['seq'].shape[1]
        seq = torch.nn.functional.one_hot(batch['seq'], num_classes=22)

        # create t1d, zeros from idx 0 - 19, 1 in idx 20 for mask, time encoding in idx 21, and bind information from 23-27
        t1d = torch.zeros(1, L, 23).to(batch['xyz_27'].device)
        t1d[..., 20] = 1
        t1d[..., 21] = 1 - (t/self.T)
        padded_bind = torch.cat((batch['bind'], torch.zeros((1, L - batch['bind'].shape[1], batch['bind'].shape[2])).to(batch['xyz_27'].device)), dim=1)
        t1d = torch.cat((t1d, padded_bind), dim=-1)
        t2d = xyz_to_t2d(pose_t.unsqueeze(0))
        # Rotation-based protein representation
        seq_tmp = torch.full((1, L), 21).to(batch['xyz_27'].device)
        alpha, _, alpha_mask, _ = get_torsions(pose_t.reshape(-1, L, 27, 3), seq_tmp, self.torsion_indices.to(batch['xyz_27'].device), self.torsion_can_flip.to(batch['xyz_27'].device), self.torsion_ref_angles.to(batch['xyz_27'].device)) #these wierd tensors are from rfdiffusion.utils
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        # MSA information. strictly not used in my use-case but neccesary input doe to legacy from rosettafold
        print('LENGTH = ', L)
        seq_in = torch.nn.functional.one_hot(seq_tmp)
        msa_masked = torch.zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq_in[None, None]
        msa_masked[:, :, :, 22:44] = seq_in[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        msa_full = torch.zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        # idx -> order of amino acids, just ascending list
        idx_pdb = torch.tensor([i for i in range(L)])
        # mask, all set to False as no partial masking 
        mask = torch.tensor([False for i in range(L)]).to(batch['bind'].device) 
        
        # Check devices
        pose_t = pose_t.to(batch['bind'].device)
        t1d = t1d.to(batch['bind'].device)
        t2d = t2d.to(batch['bind'].device).squeeze(0)
        alpha_t = alpha_t.to(batch['bind'].device).squeeze()
        pose_t = pose_t.to(batch['bind'].device)
        seq_in = seq_in.to(batch['bind'].device).float()
        seq_tmp = seq_tmp.to(batch['bind'].device).float()
        idx_pdb = idx_pdb.to(batch['bind'].device).float().squeeze(0)
        msa_masked = msa_masked.to(batch['bind'].device)
        msa_full = msa_full.to(batch['bind'].device)
        #import pdb;pdb.set_trace()
        # Add hydrogen information clone not present from alphafold
        xyz_t = torch.clone(pose_t)
        xyz_t = xyz_t[None, None]
        xyz_t = torch.cat((xyz_t[:, :14, :].squeeze(0), torch.full((1, 1, L, 13, 3), float('nan')).to(self.device)), dim=3).squeeze(0)
        #WORKING
        print("msa_masked: ", msa_masked.shape)
        print("msa_full: ", msa_full.shape)
        print("seq_in: ", seq_in.shape)
        print('seq_tmp: ', seq_tmp.shape)
        print('xyz_t squeeze, xyz input: ', xyz_t.squeeze(dim=0).shape)
        print('idx_pdb: ', idx_pdb.shape)
        print('t1d: ',t1d.shape)
        print('t2d: ', t2d.shape)
        print('xyz_t: ', xyz_t.shape)
        print('alpha_t: ',alpha_t.shape)
        print('MASK_27 = ', mask.shape) 
        import pdb; pdb.set_trace()
        if xyz_0_prev == None:

            logits, logits_aa, logits_exp, xyz_pred, alpha_s, lddt = self.model(
                    msa_latent=msa_masked, 
                    msa_full=msa_full,
                    seq=seq_tmp,
                    xyz=xyz_t.squeeze(0),#[:, :14, :]. xyz_prev in paper
                    idx=idx_pdb,
                    t=torch.tensor(t),
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t, 
                    alpha_t=alpha_t,
                    motif_mask=mask,
                    )
        else: #for self conditioning
            logits, logits_aa, logits_exp, xyz_pred, alpha_s, lddt = self.model(
                    msa_latent=msa_masked, 
                    msa_full=msa_full,
                    seq=seq_in,
                    xyz=xyz_t.squeeze(),#[:, :14, :]. xyz_prev in paper
                    idx=idx_pdb,
                    t=torch.tensor(t),
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_0_prev, 
                    alpha_t=alpha_t,
                    motif_mask=mask,
                    )
        logits_dist, logits_omega, logits_theta, logits_phi = logits
        loss = diffusor_utils.ldiffusion(xyz_27, xyz_pred, logits_dist, logits_omega, logits_theta, logits_phi) 
        return loss, xyz_pred[-1] #check to make sure xyz_pred last structure is -1



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
        os.mkdir(f"/vast/og2114/output_home/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs")
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
