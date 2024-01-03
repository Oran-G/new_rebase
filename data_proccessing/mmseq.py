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
import os

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import csv
# import fasta 

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    '''
    https://github.com/soedinglab/MMseqs2
    ^^ git with documentation^^
    '''
    # must run fasta.py first


    # os.system('brew install mmseqs2')
    print('brew install mmseqs2')
    os.system(f'mmseqs createdb {cfg.io.fasta} DB')
    print(f'mmseqs createdb {cfg.io.fasta} DB')
    os.system(f'mmseqs cluster DB DB_clu {cfg.io.tmp} --min-seq-id 0.3')
    print(f'mmseqs cluster DB DB_clu {cfg.io.tmp} --min-seq-id 0.3')
    os.system(f'mmseqs createtsv DB DB DB_clu data.tsv')
    print(f'mmseqs createtsv DB DB DB_clu data.tsv')
    
    df = pd.read_csv(cfg.io.output)

    with open('data.tsv', 'r') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        dicts = {row[1]:row[0] for row in read_tsv}
        print(len(dicts.keys()))
        newcol = []
        # if 2 in 
        # df = df.dropna()
        for row in df.iterrows():
            # print(row[0])
            # print(row[1].id)
            newcol.append(dicts[str(row[0])])
            # quit()
            
        df['cluster'] = newcol 
    print(df)
    table = df.to_csv(pd.DataFrame(cfg.io.cluster))
    # pq.write_table(table, cfg.io.cluster)
    os.system(f'rm data.tsv')
    print(f'rm data.tsv')

if __name__ == "__main__":
    main()