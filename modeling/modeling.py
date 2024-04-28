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
        dataset =  modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(self.hparams.io.final, 'train', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=self.hparams.model.sample_by_cluster),
            self.dictionary,
            self.hparams.model.name,
            self.hparams.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collater)

        return dataloader
    def val_dataloader(self):
        dataset = modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(self.hparams.io.final, 'val', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=self.hparams.model.sample_by_cluster),
            self.dictionary,
            self.hparams.model.name,
            self.hparams.io.embeddings_store_dir,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater)

        return dataloader 
    def test_dataloader(self):
        dataset = modeling_utils.EmbeddedFastaDatasetWrapper(
            modeling_utils.CSVDataset(self.hparams.io.final, 'test', self.hparams.model.name, self.hparams.io.embeddings_store_dir, clust=False),
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
            def __getitem__(self, key):
                return self.last_hidden_state
            def __len__(self):
                return 1
        start_time = time.time()

        torch.cuda.empty_cache()
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        # import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        mask = (batch['embedding'][:, :, 0] != self.dictionary.pad()).int()
        
        pred = self.model(encoder_outputs=[batch['embedding']], attention_mask=mask, labels=batch['bind'].long())
        generated = self.model.generate(input_ids=None, encoder_outputs=EncoderOutput(batch['embedding']), attention_mask=mask)
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
                lastidx_generation = -1 if len((generated[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (generated[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
                # import pdb; pdb.set_trace()
                self.test_data.append({
                    'id': batch['id'][i],
                    'seq': self.decode(batch['seq'][i].long().tolist()).split("<eos>")[0],
                    'bind': self.decode(batch['bind'][i].long().tolist()[:batch['bind'][i].tolist().index(2)]),
                    'predicted': self.decode(nn.functional.softmax(pred[1], dim=-1).argmax(-1).tolist()[:lastidx][0]),
                    'predicted_logits': nn.functional.softmax(pred[1], dim=-1)[:lastidx],
                    'generated': self.decode(generated[i][:lastidx_generation]),
                    'predict_loss': loss.item(),
                })
            except:
                pass
            

    

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
    max1 = 0
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
        #add in support for test-only mode
        print(cfg.model.checkpoint_path)
        print(cfg.model.test_only)
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
