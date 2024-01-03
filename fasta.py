from Bio import SeqIO
import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    '''
    In the meantime, you can get started on some data analysis. Make plots of the following:
        - number of sequences in REBASE
        - fraction with DNA labels, fraction that don’t have 
        - what the DNA labels look like (are they all pretty similar?)
        - See if similarity between length of protein and length of bind exists?

        A  adenosine          C  cytidine             G  guanine
		T  thymidine          N  A/G/C/T (any)        U  uridine 
		K  G/T (keto)         S  G/C (strong)         Y  T/C (pyrimidine) 
		M  A/C (amino)        W  A/T (weak)           R  G/A (purine)        
		B  G/T/C              D  G/A/T                H  A/C/T      
		V  G/C/A 
        ^^In constants.py^^
    '''
   
    df = pd.read_csv(cfg.io.output)
    print('columns:', df.columns)
    print('Total:', df.shape[0])
    print(df.dropna())
    # df = df.dropna()
    records = []
    def mapper(row):
        # print(str(row[1].seq))
        # print(str(row[1][1]))
        # quit()
        records.append(
            SeqRecord(
                Seq(row[1][1]),
                id=str(row[0]),
                name=str(row[1][0]),
                description=str(row[1][2]),
        ))

    for row in df.iterrows():
        mapper(row)
    SeqIO.write(records, cfg.io.fasta, "fasta")

if __name__ == "__main__":
    main()