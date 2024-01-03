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
import csv
@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    df = pd.read_csv(cfg.io.cluster_data, engine='pyarrow')
    
    # df.dropna()
    print(df)


    # with open(cfg.io.cluster, 'r') as f:
    #     read_tsv = csv.reader(f, delimiter="\t")
    #     dicts = {row[1]:row[0] for row in read_tsv}
    #     print(len(dicts.keys()))
    #     newcol = []
    #     # if 2 in 
    #     df = df.dropna()
    #     for row in df.iterrows():
    #         # print(row[0])
    #         # print(row[1].id)
    #         newcol.append(dicts[str(row[0])])
    #         # quit()
            
    #     df['cluster'] = newcol 
    # print(df)
    # table = pa.Table.from_pandas(df)
    # pq.write_table(table, cfg.io.cluster_data)


if __name__ == "__main__":
    main()