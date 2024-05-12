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
import time
import pickle
import torch.nn as nn

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
        if not hasattr(self.hparams.model, 'dna_clust'):
            self.hparams.model.dna_clust = False


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
        self.test_data = []
        print('initialized')
        self.test_k = 5

    def perplexity(self, output, target):
        o =  output
        t = target
        return torch.mean(torch.square(self.perplex(o, t)))


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
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
        
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        else:
            cs = self.hparams.io.final
        dataset =  modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(cs, 'train', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=self.hparams.model.sample_by_cluster),
            self.dictionary,
            self.hparams.model.name,
            self.hparams.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collater)

        return dataloader
    def val_dataloader(self):
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        else:
            cs = self.hparams.io.final
        dataset = modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(cs, 'val', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=self.hparams.model.sample_by_cluster),
            self.dictionary,
            self.hparams.model.name,
            self.hparams.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater)

        return dataloader 
    def test_dataloader(self):
        if self.hparams.model.dna_clust == True:
            cs = self.hparams.io.dnafinal
        else:
            cs = self.hparams.io.final
        dataset = modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(cs, 'test', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=False),
            self.dictionary,
            self.hparams.model.name,
            self.hparams.io.embeddings_store_dir,
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
                    num_training_steps=400000,
                    num_warmup_steps=4000,
                )
            }
        else:
            return opt

    def test_step(self, batch, batch_idx):

        class EncoderOutput():
            def __init__(self, tensor):
                self.last_hidden_state = tensor  
            def __getitem__(self, key='last_hidden_state'):
                # Check if the key is an integer and handle it accordingly
                if isinstance(key, int):
                    if key == 0:
                        return self.last_hidden_state
                    else:
                        raise IndexError("Index out of range.")  # Only one item, so index 0 is valid
                elif key == "last_hidden_state":
                        return self.last_hidden_state
                else:
                    raise KeyError(f"Key '{key}' not supported.")

            def __setitem__(self, key, value):
                # Similarly, handle integer keys for assignment
                if isinstance(key, int):
                    if key == 0:
                        self.last_hidden_state = value
                    else:
                        raise IndexError("Index out of range.")
                elif key == "last_hidden_state":
                    self.last_hidden_state = value
                else:
                    raise KeyError(f"Key '{key}' not supported.")
            def __len__(self):
                return 1
        start_time = time.time()

        torch.cuda.empty_cache()
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        # import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        mask = (batch['embedding'][:, :, 0] != self.dictionary.pad()).int()
        with torch.no_grad():
            pred = self.model(encoder_outputs=[batch['embedding']], attention_mask=mask, labels=batch['bind'].long())
            generated = self.model.generate(input_ids=None, encoder_outputs=EncoderOutput(batch['embedding']), length_penalty=-1.0)
            torch.cuda.empty_cache()
            full_generated = self.model.generate(input_ids=None, encoder_outputs=EncoderOutput(batch['embedding']), do_sample=True, num_return_sequences=self.test_k, length_penalty=-1.0)
       
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
            
            # pred_accuracy = self.accuracy(torch.transpose(nn.functional.softmax(pred[1],dim=-1), 1,2)[i], batch['bind'][i].long())
            # generated_accuracy = self.accuracy(generated[i], batch['bind'][i].long())
            lastidx = -1 if len((pred[1].argmax(-1)[i]  == self.dictionary.eos()).nonzero(as_tuple=True)[0]) == 0 else (pred[1].argmax(-1)[i]  == self.dictionary.eos()).nonzero(as_tuple=True)[0].tolist()[0]
            lastidx_generation = -1 if len((generated[i]  == self.dictionary.eos()).nonzero(as_tuple=True)[0]) == 0 else (generated[i]  == self.dictionary.eos()).nonzero(as_tuple=True)[0].tolist()[0]
            # import pdb; pdb.set_trace()
            re = {
                'id': batch['id'][i],
                'seq': self.dictionary.string(batch['seq'][i].long().tolist()).split("<eos>")[0],
                'bind': self.dictionary.string(batch['bind'][i].long().tolist()[:batch['bind'][i].tolist().index(2)]),
                'predicted': self.dictionary.string(nn.functional.softmax(pred[1][i], dim=-1).argmax(-1).tolist()[:lastidx]),
                'predicted_logits': nn.functional.softmax(pred[1][i], dim=-1)[:lastidx].to(torch.device('cpu')).tolist(),
                'generated': self.dictionary.string(generated[i][1:lastidx_generation]),
                # 'predicted_accuracy': pred_accuracy,
                # 'generated_accuracy': generated_accuracy,
            }

            for j in range(self.test_k):
                lastidx_generation = -1 if len((full_generated[(i*self.test_k) + j]  == self.dictionary.eos()).nonzero(as_tuple=True)[0]) == 0 else (full_generated[(i*self.test_k) + j] == self.dictionary.eos()).nonzero(as_tuple=True)[0].tolist()[0]
                re[f'generated_{j}'] = self.dictionary.string(full_generated[(i*self.test_k) + j][1:lastidx_generation])
            self.test_data.append(re)

    

@hydra.main(config_path='/vast/og2114/new_rebase/configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    os.system('export TORCH_HOME=/vast/og2114/torch_home')
    try:
        if cfg.model.checkpoint_path:
            print('checkpoint path:', cfg.model.checkpoint_path)
            model = RebaseT5.load_from_checkpoint(cfg.model.checkpoint_path)
        else:
            model = RebaseT5(cfg)
    except:
        model = RebaseT5(cfg)
    # print(model.hparams)
    try:
        os.mkdir(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}")
    except: 
        pass
    try:
        os.mkdir(f"/vast/og2114/rebase/runs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/training_outputs")
    except: 
        pass
    wandb_logger = WandbLogger(project="Modeling",save_dir=cfg.io.wandb_dir, name=f"{cfg.model.name}_slurm_{os.environ['SLURM_JOB_ID']}", reinit=True)
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


    # try:
    if True:
        if cfg.model.checkpoint_path and cfg.model.test_only: 
            print('test-only mode. running test')
            model = model.to(torch.device("cuda:0"))
            model.hparams.io.test_embedded = cfg.io.test_embedded
            trainer.test(model, dataloaders=model.test_dataloader())
            with open(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/{model.hparams.model.name}_test_data.pkl", "wb") as f:
                pickle.dump(model.test_data, f)
            art = wandb.Artifact("test_data", type="dataset")
            art.add_file(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/{model.hparams.model.name}_test_data.pkl", skip_cache=True)
            wandb.run.log_artifact(art)
            print(len(model.test_data))
            return
    # except:
        
    #     print('ready to train!')
    #     trainer.fit(model)
    #     model = model.to(torch.device("cuda:0"))
    #     trainer.test(model, dataloaders=model.test_dataloader())
    #     with open(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/{model.hparams.model.name}_test_data.pkl", "wb") as f:
    #         pickle.dump(model.test_data, f)
    #     art = wandb.Artifact("test_data", type="dataset")
    #     art.add_file(f"/vast/og2114/output_home/runs/slurm_{os.environ['SLURM_JOB_ID']}/{model.hparams.model.name}_test_data.pkl", skip_cache=True)
    #     wandb.run.log_artifact(art)


  
if __name__ == '__main__':
    main()
