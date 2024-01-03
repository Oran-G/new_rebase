import matplotlib.pyplot as plt
import csv
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
    df = pd.read_parquet(cfg.io.cluster, engine='pyarrow')
    # read_tsv = csv.reader(f, delimiter="\t")
    #     dicts = {row[0]:0 for row in read_tsv}
    df = df.dropna()
    df = df.drop_duplicates(subset=['seq'])

    dicts = {}
    for row in df.iterrows():
        if row[1][3] in list(dicts.keys()):
            dicts[row[1][3]]+=1
        else:
            dicts[row[1][3]]=1
    # # print(dicts[list(dicts.keys())[5]])
    # # ase = []
    # # for key in dicts.keys():
    # #     for i in range(dicts[key]):
    # #         ase.append(dicts[key])
        
    # # df = pd.DataFrame.from_dict({'data':[dicts[key] for key in list(dicts.keys())]})
    sizes = [[], [], [], []]
    for key in dicts.keys():
        if int(dicts[key])< 10:
            sizes[0].append(key)
            
        elif dicts[key]>10 and dicts[key]<= 100:
    #         print(50)
            sizes[1].append(key)
        elif dicts[key]> 100 and dicts[key] < 1000:
            sizes[2].append(key)
        else:
            sizes[3].append(key)
    # print(sizes[0][1])
    # print(sizes[1][0])
    # print(sizes[2][0])
    # print(sizes[3][6])
    # print(dicts['57682'])
    print(len(sizes[0]))
    
    print(len(sizes[1]))
    
    print(len(sizes[2]))
    print(sizes[2])
    print(len(sizes[3]))
    print(sizes[3])
    quit()
    validation = []
    for i in range(4):
        # print(5^i)
        # print(5**i)
        # print(sizes[3-i][j] for j in range(5**i))
        for j in range(5**i):
            validation.append(sizes[3-i][j] )

    print(len(validation))

    test = []
    for i in range(4):
        # print(5^i)
        # print(5**i)
        # print(sizes[3-i][j] for j in range(5**i))
        for j in range(5**i):
            test.append(sizes[3-i][j+(5**i)] )
    print(len(test))

    val_data = df[df['cluster'].isin(validation)]
    test_data = df[df['cluster'].isin(test)]
    df.drop(val_data.index, inplace = True)
    df.drop(test_data.index, inplace = True)
    print(len(df)+len(test_data)+len(val_data))
    print(len(df))

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
    SeqIO.write(records, f'{cfg.io.train}.fasta', "fasta")


    records = []
    for row in val_data.iterrows():
        mapper(row)
    SeqIO.write(records, f'{cfg.io.val}.fasta', "fasta")

    records = []
    for row in test_data.iterrows():
        mapper(row)
    SeqIO.write(records, f'{cfg.io.test}.fasta', "fasta")

if __name__ == "__main__":
    main()

