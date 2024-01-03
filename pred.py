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
from folding import RebaseT5

@hydra.main(version_base='1.2.0', config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # model = RebaseT5(cfg)
    # print('init')
    # checkpoint = torch.load('/scratch/og2114/rebase/logs/Focus/21hjudcf/checkpoints/both_dff-128_dmodel-768_lr-0.001_batch-512.ckpt')
    # print(checkpoint.keys())
    model = RebaseT5.load_from_checkpoint(checkpoint_path="/scratch/og2114/rebase/logs/Focus/vw1qopku/checkpoints/acc-small_dff-64_dmodel-768_lr-0.001_batch-32.ckpt")
    # model = RebaseT5.load_from_checkpoint(checkpoint_path='/scratch/og2114/rebase/logs/Focus/21hjudcf/checkpoints/both_dff-128_dmodel-768_lr-0.001_batch-512.ckpt')
    gpu = cfg.model.gpu
    cfg = model.hparams
    cfg.model.gpu = gpu
    wandb_logger = WandbLogger(project="Focus",save_dir=cfg.io.wandb_dir)
    wandb_logger.experiment.config.update(dict(cfg.model))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename=f'{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True) 
    acc_callback = ModelCheckpoint(monitor="val_acc", filename=f'acc-{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True) 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print(model.batch_size)
    print('tune: ')
    # trainer.tune(model)
    model.batch_size = 8
    if int(cfg.esm.layers) == 12:
        model.batch_size = 2
    if int(cfg.esm.layers) == 34:
        model.batch_size = 1
    print(model.batch_size)
    # quit()
    print(int(max(1, cfg.model.batch_size/model.batch_size)))
    # trainer.__init__(
    trainer = pl.Trainer(
        gpus=int(cfg.model.gpu), 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback, lr_monitor, acc_callback],
        # check_val_every_n_epoch=1000,
        # max_epochs=cfg.model.max_epochs,
        default_root_dir=cfg.io.checkpoints,
        accumulate_grad_batches=int(max(1, cfg.model.batch_size/model.batch_size/int(cfg.model.gpu))),
        precision=cfg.model.precision,
        accelerator='ddp',
        log_every_n_steps=5,

        )
    #import pdb; pdb.set_trace()
    model.to( torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    trainer.test(model, dataloaders=model.val_dataloader())
    print(model.test_data)
    import csv
    dictionaries=model.test_data
    keys = dictionaries[0].keys()
    a_file = open("output.csv", "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(dictionaries)
    a_file.close()




if __name__ == '__main__':
    main()

