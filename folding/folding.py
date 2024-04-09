import torch
import torch.nn as nn
#import pytorch_lightning as pl
import lightning.pytorch as pl
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
import os
import json
import wandb
import csv
import random
import folding_utils
'''
TODOs (10/17/21):
* figure out reasonable train/valid set
* run a few baselines in this setup to get a handle on what performnace is like
* ESM-1b pretrained representations
* Alphafold
'''







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
        self.save_hyperparameters(cfg)
        self.batch_size = self.hparams.model.batch_size
        print("Argument hparams: ", self.hparams)
        print('batch size', self.hparams.model.batch_size)
        
        '''
        model git - https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/gvp_transformer.py
        alphabet git - https://github.com/facebookresearch/esm/blob/main/esm/data.py
        '''
        self.ifmodel, self.ifalphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        
        
        '''
        T5 model to be used as a possible decoder

        https://huggingface.co/docs/transformers/model_doc/t5
        '''
        t5_config=T5Config(
            vocab_size=len(self.ifalphabet),
            decoder_start_token_id=self.ifalphabet.get_idx('<af2>'),
            d_model=self.hparams.model.d_model,
            d_ff=self.hparams.model.d_ff,
            num_layers=self.hparams.model.layers,
            pad_token_id=self.ifalphabet.padding_idx,
            eos_token_id=self.ifalphabet.eos_idx,
        )
        self.model = T5ForConditionalGeneration(t5_config)

        '''
        loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        accuracy: https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
        '''

        self.loss = nn.CrossEntropyLoss(ignore_index=self.ifalphabet.padding_idx)
        self.accuracy = torchmetrics.Accuracy(ignore_index=self.ifalphabet.padding_idx, mdmc_average='samplewise')
        self.perplex = torch.nn.CrossEntropyLoss(reduction='none')
        
        '''
        used to record validation data for logging
        '''
        self.val_data = []
        print('initialized')
        
        

    def training_step(self, batch, batch_idx):
        '''
        Training step
        input: 
            batch: output of EncodedFastaDatasetWrapper.collate_dicts {
                'bind': torch.tensor (bind site)
                'bos_bind': torch.tensor (bos+bind site)
                'coords': torch.tensor (coords input to esm if)
                'seq': torch.tensor (protein sequence)
                'bos_seq': torch.tensor (bos+protein sequence)
                'coord_conf': torch.tensor(confidence input to esmif encoder)
                'coord_pad' torch.tensor (padding_mask input to esm if encoder)
            }

        output:
            loss for pl,
            batch sizefor pl
        '''
        '''step lr scheduler'''
        if self.global_step  != 0:
            self.lr_schedulers().step()

        start_time  = time.time()

        torch.cuda.empty_cache()
        '''
        take out attention mask - I do not think it is needed right now
        labels changed so that padding idx is -100 - needed as -100 is the default ignore index for crossEntropyLoss,and is used by T5 in this case
        pred: [
            loss,
            logits,
            ...
        ]
        '''
        
        label = batch['bind']
        label[label==self.ifalphabet.padding_idx] = -100

        try:
            pred = self.model(encoder_outputs=[batch['seq_enc']], labels=label)
        except RuntimeError:
            print(batch, batch_idx)
        
        batch['bind'][batch['bind']==-100] = self.ifalphabet.padding_idx
        #import pdb; pdb.set_trace()
        loss=self.loss(torch.transpose(pred[1],1, 2), batch['bind'])
        
        confs = self.conf(nn.functional.softmax(pred[1], dim=-1),target=batch['bind'])
        self.log('top_conf', float(confs[0]), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('low_conf', float(confs[1]), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', float(loss.item()), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc',float(self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2), batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True) #accuracy using torchmetrics accuracy
        self.log('length', int(pred[1].shape[-2]),  on_step=True,  logger=True) # length of prediction
        self.log('train_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True) # step time
       
        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):

        
        start_time = time.time()

        torch.cuda.empty_cache()

        label = batch['bind']
        label[label==self.ifalphabet.padding_idx] = -100
        try:
            pred = self.model(encoder_outputs=[batch['seq_enc']], labels=label)
        except RuntimeError:
            print(token_representations['encoder_out'], batch, batch_idx)
        batch['bind'][batch['bind']==-100] = self.ifalphabet.padding_idx
        loss=self.loss(torch.transpose(pred[1],1, 2), batch['bind'])
        confs = self.conf(nn.functional.softmax(pred[1], dim=-1),target=batch['bind'])
        self.log('val_top_conf', float(confs[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_low_conf', float(confs[1]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mean_conf', float(confs[2]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('val_loss', float(loss.item()), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc', float(self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2), batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        if str(self.cfg.model.seq_identity) == '0.9':
            print(.7)
            cs = f'{self.cfg.io.final}-9'
        elif str(self.cfg.model.seq_identity) == '0.7':
            print(.9)
            cs = f'{self.cfg.io.final}-7'
        else:
            cs = self.cfg.io.final
        if self.cfg.model.dna_clust == True:
            cs = self.cfg.io.dnafinal
        print(cs)
        dataset = folding_utils.EncodedFastaDatasetWrapper(
            folding_utils.CSVDataset(cs, 'train', clust=self.cfg.model.sample_by_cluster),

            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        

        encoder_dataset = folding_utils.EncoderDataset(dataset, batch_size=2, device=self.device, path=self.cfg.io.val_embedded), 
        dataloader = DataLoader(encoder_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=encoder_dataset.collater)
        return dataloader 
    def val_dataloader(self):
        if str(self.cfg.model.seq_identity)== '0.9':
            print(".9 seq")
            cs = f'{self.cfg.io.final}-9'
        elif str(self.cfg.model.seq_identity) == '0.7':
            print('.7 seq')
            cs = f'{self.cfg.io.final}-7'
        else:
            cs = self.cfg.io.final
        
        if self.cfg.model.dna_clust == True:
            cs = self.cfg.io.dnafinal
        print(self.cfg.model.seq_identity)
        print(cs)
        dataset = folding_utils.EncodedFastaDatasetWrapper(
            folding_utils.CSVDataset(cs, 'val', clust=self.cfg.model.sample_by_cluster),
            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        encoder_dataset = folding_utils.EncoderDataset(dataset, batch_size=self.batch_size, device=self.device, path=self.cfg.io.val_embedded), 
        dataloader = DataLoader(encoder_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=encoder_dataset.collater)
        return dataloader 

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                {'params': self.ifmodel.parameters(), 'lr': float(self.hparams.model.lr)/5},  
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
                    num_warmup_steps=4000, #was 4000
                    power=cfg.model.lrpower,
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
    
    def conf(self, tens, target):
        '''
        insight onto the top probabilities of the model. collection of data on the probabilities of each token. ex: 
        tensor([[0.1000, 0.3000, 0.6000],
            [0.1000, 0.5000, 0.4000],
            [0.1500, 0.0500, 0.8000]])
        the top would be 
            [.6, .5, .8] this list is then multiplied by the tokenwise accuracy, if thetop token is corrrect,multiply by 1, else multipy by 0.makes probability for incorrect tokens 0
        return top of these values, min of values, mean of values
        '''
        h1 = []
        h2 = []
        h3 = []
        for i in range(tens.shape[0]):
            lastidx = -1 if len((target.argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (target.argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
            # top probability for each token, ie the probability that the argmaxed token was chosen. multiplied by toknwise accuracy, so the probability if the topP token was correct else 0
            highs = (torch.amax(tens[i], -1)[:lastidx]* ((tens[i].argmax(-1)[:lastidx]==target[i][:lastidx]).int())).tolist()
            if len(highs)==0:
                highs= [0]
            h1.append(max(highs)) # maximum of top probabilities
            h2.append(min(highs))
            h3.append((sum(highs)/len(highs))) # mean of top confidencs
        return max(h1), min(h2), (sum(h3)/len(h3))


    def validation_epoch_end(self, validation_step_outputs):
        '''
            end of epoch - saves the validation data outlined in validation step to csv
        '''
        if False:
            df1 = pd.DataFrame(self.val_data)
            dictionaries=self.val_data
            print(len(dictionaries))
            import pdb; pdb.set_trace()
            keys = dictionaries[0].keys()
            a_file = open(f"/vast/og2114/rebase/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs/{self.trainer.current_epoch}-output.csv", "w")
            dict_writer = csv.DictWriter(a_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dictionaries)
            a_file.close()
            self.val_data = []
    def test_step(self, batch, batch_idx):

        
        start_time = time.time()

        torch.cuda.empty_cache()
        
        token_representations = self.ifmodel.encoder(batch['coords'], batch['coord_pad'], batch['coord_conf'])
        label = batch['bind']
        label[label==self.ifalphabet.padding_idx] = -100
        pred = self.model(encoder_outputs=[torch.transpose(token_representations['encoder_out'][0], 0, 1)], labels=label)
        batch['bind'][batch['bind']==-100] = self.ifalphabet.padding_idx
        loss=self.loss(torch.transpose(pred[1],1, 2), batch['bind'])
        '''
        record validation data into val_data
        form:  {
            seq: protein sequence
            bind: bind site ground truth
            predicted: the predicted bind site
        }
        '''
        ''' not working - to be fixed later'''
        for i in range(pred[1].shape[0]):
            try:

                lastidx = -1 if len((pred[1].argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (pred[1].argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
                self.val_data.append({
                    'seq': self.decode(batch['seq'][i].tolist()).split("<eos>")[0],
                    'bind': self.decode(batch['bind'][i].tolist()[:batch['bind'][i].tolist().index(2)]),
                    'predicted': self.decode(nn.functional.softmax(pred[1], dim=-1).argmax(-1).tolist()[:lastidx][0])
                })
                
            except IndexError:
                print('Index Error')
                import pdb; pdb.set_trace()





@hydra.main(config_path='../configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    model = RebaseT5(cfg)
    gpu = cfg.model.gpu
    cfg = model.hparams
    cfg.model.gpu = gpu
    
    wandb.init(settings=wandb.Settings(start_method='thread', code_dir="."))
    wandb.save(os.path.abspath(__file__))
    wandb_logger = WandbLogger(project="Focus",save_dir=cfg.io.wandb_dir)
    wandb_logger.watch(model)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", filename=f'{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True,save_top_k=5) 
    acc_callback = ModelCheckpoint(monitor="val_acc_epoch", filename=f'acc-{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True, save_top_k=5) 
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    BSFinder = pl.callbacks.BatchSizeFinder()
    print(model.batch_size)
    print('tune: ')
    model.batch_size = 2
    try:
        os.mkdir(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}")
    except: 
        pass
    try:
        os.mkdir(f"/vast/og2114/rebase/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs")
    except: 
        pass
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
            BSFinder,
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
