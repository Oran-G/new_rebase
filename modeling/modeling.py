import torch
import lightning.pytorch as pl
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from omegaconf import DictConfig, OmegaConf
import hydra

import torchmetrics

from typing import List, Dict
from pytorch_lightning.loggers import WandbLogger


from pandas import DataFrame as df
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

import pandas as pd
import os
import modeling_utils
import wandb

'''
TODOs (10/17/21):
* figure out reasonable train/valid set
* run a few baselines in this setup to get a handle on what performnace is like
* ESM-1b pretrained representations
* Alphafold
'''


def accuracy(predict:torch.tensor, label:torch.tensor, mask:torch.tensor):
    first = (predict==label).int()
    second = first*mask
    return second.sum()/mask.sum()

class RebaseT5(pl.LightningModule):
    def __init__(self, cfg):
        super(RebaseT5, self).__init__()
        self.save_hyperparameters(cfg)
        print('batch size', self.hparams.model.batch_size)
        self.batch_size = self.hparams.model.batch_size
        

        self.dictionary = modeling_utils.InlineDictionary.from_list(
            tokenization['toks']
        )
        self.cfg = cfg

        self.perplex = torch.nn.CrossEntropyLoss(reduction='none')
        

        self.esm, self.esm_dictionary = torch.hub.load("facebookresearch/esm:main", self.hparams.esm.path)
       
        t5_config=T5Config(
            vocab_size=len(self.dictionary),
            decoder_start_token_id=self.dictionary.pad(),
            # TODO: grab these from the config
            d_model=self.hparams.model.d_model,
            d_ff=self.hparams.model.d_ff,
            d_kv=self.hparams.model.d_kv,
            num_layers=self.hparams.model.layers,
            pad_token_id=self.dictionary.pad(),
            eos_token_id=self.dictionary.eos(),
        )

        self.model = T5ForConditionalGeneration(t5_config)
        self.accuracy = torchmetrics.Accuracy(ignore_index=self.dictionary.pad())
        print('initialized')

    def perplexity(self, output, target):
        o =  output
        t = target
        return torch.mean(torch.square(self.perplex(o, t)))


    def training_step(self, batch, batch_idx):
        if self.global_step  != 0 and self.global_step % 4 == 0:
            self.lr_schedulers().step()
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        mask = (batch['embedding'][:, :, 0] != self.dictionary.pad()).int()
        output = self.model(encoder_outputs=[batch['embedding']], attention_mask=mask, labels=batch['bind'].long())

        
        
        self.log('train_loss', float(output.loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train_acc',float(accuracy(output['logits'].argmax(-1), batch['bind'].long(), (batch['bind'] != -100).int())), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
        # self.log('train_perplex',float(self.perplexity(output['logits'], batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        # import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        mask = (batch['embedding'][:, :, 0] != self.dictionary.pad()).int()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            output = self.model(encoder_outputs=[batch['embedding']], attention_mask=mask, labels=batch['bind'].long())
        self.log('val_loss', float(output.loss), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc',float(accuracy(output['logits'].argmax(-1), batch['bind'].long(), (batch['bind'] != self.dictionary.pad()).int())), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        dataset =  modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(self.cfg.io.final, 'train', self.cfg.model.name, self.cfg.io.embeddings_store_dir, clust=self.cfg.model.sample_by_cluster),
            self.dictionary,
            self.cfg.model.name,
            self.cfg.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collater)

        return dataloader
    def val_dataloader(self):
        dataset = modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(self.cfg.io.final, 'val', self.cfg.model.name, self.cfg.io.embeddings_store_dir, clust=self.cfg.model.sample_by_cluster),
            self.dictionary,
            self.cfg.model.name,
            self.cfg.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater)

        return dataloader 


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.model.lr)
        # return opt

        # figure out reaosnable number of total steps
        # 100k steps
        # 4% warmup to peak lr
        # linear decay after that

        if self.hparams.model.scheduler:
            return {
                'optimizer': opt,
                'lr_scheduler': get_linear_schedule_with_warmup(
                    optimizer=opt,
                    num_training_steps=40000,
                    num_warmup_steps=1000,
                )
            }
        else:
            return opt
            
            

    

@hydra.main(config_path='/vast/og2114/new_rebase/configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    os.system('export TORCH_HOME=/vast/og2114/torch_home')
    model = RebaseT5(cfg)
    max1 = 0
    try:
        os.mkdir(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}")
    except: 
        pass
    try:
        os.mkdir(f"/vast/og2114/rebase/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs")
    except: 
        pass
    wandb_logger = WandbLogger(project="Focus",save_dir=cfg.io.wandb_dir)
    wandb_logger.experiment.config.update(dict(cfg.model))
    wandb.save(os.path.abspath(__file__))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename=f'{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True) 
    acc_callback = ModelCheckpoint(monitor="val_acc", filename=f'acc-{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True) 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        devices=-1, 
        accelerator="gpu", 
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, acc_callback],

        default_root_dir=cfg.io.checkpoints,
        precision=cfg.model.precision,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=5,

        )
    trainer.fit(model)
    one = model.val_test()
    print(one)
    pred = pd.DataFrame(one)
    print(pred)
    pred.to_csv(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/final.csv")
    
if __name__ == '__main__':
    main()
