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
from lightning.pytorch.loggers import WandbLogger
from pandas import DataFrame as df
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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



class RebaseT5(pl.LightningModule):
    def __init__(self, cfg):


        super(RebaseT5, self).__init__()
        self.save_hyperparameters(cfg)
    
        try:
            self.hparams['slurm'] = str(os.environ.get('SLURM_JOB_ID'))
        except:
            pass
        
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
        
        label = batch['bind']
        label[label==self.ifalphabet.padding_idx] = -100

        # import pdb; pdb.set_trace()
        pred = self.model(encoder_outputs=[batch['seq_enc']], labels=label.long())

        
        batch['bind'][batch['bind']==-100] = self.ifalphabet.padding_idx
        #import pdb; pdb.set_trace()
        loss=self.loss(torch.transpose(pred[1],1, 2), batch['bind'].long())
        

        self.log('train_loss', float(loss.item()), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc',float(self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2), batch['bind'].long())), on_step=True, on_epoch=True, prog_bar=False, logger=True) #accuracy using torchmetrics accuracy
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
        # import pdb; pdb.set_trace()
        pred = self.model(encoder_outputs=[batch['seq_enc']], labels=label.long())
        batch['bind'][batch['bind']==-100] = self.ifalphabet.padding_idx
        loss=self.loss(torch.transpose(pred[1],1, 2), batch['bind'].long())
        
        self.log('val_loss', float(loss.item()), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc', float(self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2), batch['bind'].long())), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        if str(self.hparams.model.seq_identity) == '0.9':
            print(.7)
            cs = f'{self.hparams.io.final}-9'
        elif str(self.hparams.model.seq_identity) == '0.7':
            print(.9)
            cs = f'{self.hparams.io.final}-7'
        else:
            cs = self.hparams.io.final
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        print(cs)
        dataset = folding_utils.EncodedFastaDatasetWrapper(
            folding_utils.CSVDataset(cs, 'train', clust=False),

            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        

        encoder_dataset = folding_utils.EncoderDataset(dataset, batch_size=1, device=self.device, path=self.hparams.io.train_embedded, cluster=False)
        dataloader = DataLoader(encoder_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=encoder_dataset.collater)
        print('train dataset length:', len(encoder_dataset))        
        return dataloader 
    def val_dataloader(self):
        if str(self.hparams.model.seq_identity)== '0.9':
            print(".9 seq")
            cs = f'{self.hparams.io.final}-9'
        elif str(self.hparams.model.seq_identity) == '0.7':
            print('.7 seq')
            cs = f'{self.hparams.io.final}-7'
        else:
            cs = self.hparams.io.final
        
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        print(self.hparams.model.seq_identity)
        print(cs)
        dataset = folding_utils.EncodedFastaDatasetWrapper(
            folding_utils.CSVDataset(cs, 'val', clust=False),
            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        encoder_dataset = folding_utils.EncoderDataset(dataset, batch_size=4, device=self.device, path=self.hparams.io.val_embedded, cluster=False)
        dataloader = DataLoader(encoder_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=encoder_dataset.collater)
        print('val dataset length:', len(encoder_dataset))
        return dataloader 

    def test_dataloader(self):
        if str(self.hparams.model.seq_identity)== '0.9':
            print(".9 seq")
            cs = f'{self.hparams.io.final}-9'
        elif str(self.hparams.model.seq_identity) == '0.7':
            print('.7 seq')
            cs = f'{self.hparams.io.final}-7'
        else:
            cs = self.hparams.io.final
        
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        print(self.hparams.model.seq_identity)
        print(cs)
        dataset = folding_utils.EncodedFastaDatasetWrapper(
            folding_utils.CSVDataset(cs, 'test', clust=False),
            self.ifalphabet,
            apply_eos=True,
            apply_bos=False,
        )
        encoder_dataset = folding_utils.EncoderDataset(dataset, batch_size=4, device=self.device, path=self.hparams.io.val_embedded)
        # import pdb; pdb.set_trace()
        dataloader = DataLoader(encoder_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=encoder_dataset.collater)
        return dataloader 

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                #{'params': self.ifmodel.parameters(), 'lr': float(self.hparams.model.lr)/5},  
                {'params': self.model.decoder.parameters()}], 
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



    def test_step(self, batch, batch_idx):

        
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
    os.system('export TORCH_HOME=/vast/og2114/torch_home')
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

    print('tune: ')
    '''
    batch_trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[
            BSFinder,
            ],
        precision=cfg.model.precision,
        
        fast_dev_run=True,
        gradient_clip_val=0.3,
        )
    batch_trainer.fit(model)
    print(model.batch_size)

    '''

    trainer = pl.Trainer(
        devices=-1, 
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback, 
            lr_monitor, 
            acc_callback, 
            ],
        default_root_dir=cfg.io.checkpoints,
        accumulate_grad_batches=max(1, int(64/model.batch_size)),
        precision=cfg.model.precision,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=5,
        max_epochs=-1,
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
