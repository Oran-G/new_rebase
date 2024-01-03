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
import csv
@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print('brew install mmseqs2')
    records = list(SeqIO.parse(cfg.io.finput, "fasta"))
    new_records = []
    print(len(records))
        
    for row in records:
        if row.description.split(' ')[0] != '':
            new_records.append(
                SeqRecord(
                    row.seq,
                    id=row.id,
                    name=row.name,
                    description=row.description,
            ))
        
            
    print(len(new_records))
    
    
    SeqIO.write(new_records, cfg.io.finput, "fasta")
    os.system(f'mmseqs createdb {cfg.io.finput} DB')
    print(f'mmseqs createdb {cfg.io.finput} DB')
    os.system(f'mmseqs cluster  DB DB_clu {cfg.io.tmp} --min-seq-id 0.7')
    print(f'mmseqs cluster DB DB_clu {cfg.io.tmp} --min-seq-id 0.7')
    os.system(f'mmseqs createtsv DB DB DB_clu {cfg.io.temp}')
    print(f'mmseqs createtsv DB DB DB_clu {cfg.io.temp}')


    to_df = {
        'id':[],
        'seq':[],
        'bind':[],
        'name':[],
        'cluster':[],
        'split':[],
        
    }

    records = list(SeqIO.parse(cfg.io.finput, "fasta"))

    read_tsv = list(csv.reader(open(cfg.io.temp, 'r'), delimiter="\t"))
    print(read_tsv[0])
    clust = {row[1]:row[0] for row in read_tsv}
    print(len(clust))
    dicts = {}
    for row in read_tsv:
        if row[0] in list(dicts.keys()):
            dicts[row[0]]+=1
        else:
            dicts[row[0]]=1
    sizes = [[], [], [], []]
    for key in dicts.keys():
        if int(dicts[key])< 10:
            sizes[0].append(key)
            
        elif dicts[key]>10 and dicts[key]<= 100:
            sizes[1].append(key)

        elif dicts[key]> 100 and dicts[key] < 1000:
            sizes[2].append(key)

        else:
            sizes[3].append(key)
    print('.7 sizes', len(sizes[0]), len(sizes[1]), len(sizes[2]), len(sizes[3]))

    validation = []
    for i in range(4):
        print(6**i)
        for j in range(6**i):
            validation.append(sizes[3-i][j] )
    print(len(validation))

    test = []
    for i in range(4):
        print(6**i)
        for j in range(6**i):
            test.append(sizes[3-i][j+(6**i)] )
    print(len(test))
    train = []
    for i in range(4):
        print(6**i)
        for j in range(len(sizes[i]) - (2*(6**i))):
            train.append(sizes[i][(2*(6**i)) + j])
    print(len(train))
    print(len(train)+len(test)+len(validation))


    splits = [train, validation, test]
    for record in records:
        if record.description.split(' ')[3] != '':
            name = record.name
            ids = record.id
            seq = str(record.seq)
            cluster = clust[record.id]
            bind = record.description.split(' ')[3]
            s = 0
            for i in range(len(splits)):
                if clust[record.id] in splits[i] :
                    s = i
            to_df['name'].append(name)
            to_df['id'].append(ids)
            to_df['seq'].append(seq)
            to_df['cluster'].append(cluster)
            to_df['bind'].append(bind)
            to_df['split'].append(s)
        # except:
        #     print(record)
    
    df = pd.DataFrame.from_dict(to_df)
    df.to_csv(f"{cfg.io.final}-7")
    print(cfg.io.final)
    # print(records[53072])
    # print(records[53073])


    # print(records[5].description.split(' '))
    
    # for record in records:
        
    print(OmegaConf.to_yaml(cfg))
    print('brew install mmseqs2')
    records = list(SeqIO.parse(cfg.io.finput, "fasta"))
    new_records = []
    print(len(records))
        
    for row in records:
        if row.description.split(' ')[0] != '':
            new_records.append(
                SeqRecord(
                    row.seq,
                    id=row.id,
                    name=row.name,
                    description=row.description,
            ))
        
            
    print(len(new_records))
    
    
    SeqIO.write(new_records, cfg.io.finput, "fasta")
    os.system(f'mmseqs createdb {cfg.io.finput} DB')
    print(f'mmseqs createdb {cfg.io.finput} DB')
    os.system(f'mmseqs cluster  DB DB_clu {cfg.io.tmp} --min-seq-id 0.9')
    print(f'mmseqs cluster DB DB_clu {cfg.io.tmp} --min-seq-id 0.9')
    os.system(f'mmseqs createtsv DB DB DB_clu {cfg.io.temp}')
    print(f'mmseqs createtsv DB DB DB_clu {cfg.io.temp}')


    to_df = {
        'id':[],
        'seq':[],
        'bind':[],
        'name':[],
        'cluster':[],
        'split':[],
        
    }

    records = list(SeqIO.parse(cfg.io.finput, "fasta"))

    read_tsv = list(csv.reader(open(cfg.io.temp, 'r'), delimiter="\t"))
    print(read_tsv[0])
    clust = {row[1]:row[0] for row in read_tsv}
    print(len(clust))
    dicts = {}
    for row in read_tsv:
        if row[0] in list(dicts.keys()):
            dicts[row[0]]+=1
        else:
            dicts[row[0]]=1
    sizes = [[], [], [], []]
    for key in dicts.keys():
        if int(dicts[key])< 10:
            sizes[0].append(key)
            
        elif dicts[key]>10 and dicts[key]<= 100:
            sizes[1].append(key)

        elif dicts[key]> 100 and dicts[key] < 1000:
            sizes[2].append(key)

        else:
            sizes[3].append(key)
    print('.9 sizes', len(sizes[0]), len(sizes[1]), len(sizes[2]), len(sizes[3]))

    validation = []
    for i in range(4):
        print(6**i)
        for j in range(6**i):
            validation.append(sizes[3-i][j] )
    print(len(validation))

    test = []
    for i in range(4):
        print(6**i)
        for j in range(6**i):
            test.append(sizes[3-i][j+(6**i)] )
    print(len(test))
    train = []
    for i in range(4):
        print(6**i)
        for j in range(len(sizes[i]) - (2*(6**i))):
            train.append(sizes[i][(2*(6**i)) + j])
    print(len(train))
    print(len(train)+len(test)+len(validation))


    splits = [train, validation, test]
    for record in records:
        if record.description.split(' ')[3] != '':
            name = record.name
            ids = record.id
            seq = str(record.seq)
            cluster = clust[record.id]
            bind = record.description.split(' ')[3]
            s = 0
            for i in range(len(splits)):
                if clust[record.id] in splits[i] :
                    s = i
            to_df['name'].append(name)
            to_df['id'].append(ids)
            to_df['seq'].append(seq)
            to_df['cluster'].append(cluster)
            to_df['bind'].append(bind)
            to_df['split'].append(s)
        # except:
        #     print(record)
    
    df = pd.DataFrame.from_dict(to_df)
    df.to_csv(f"{cfg.io.final}-9")
    print(cfg.io.final)
    # print(records[53072])
    # print(records[53073])


    # print(records[5].description.split(' '))
    
    # for record in records:
        


if __name__ == "__main__":
    main()
