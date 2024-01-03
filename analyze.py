import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from constants import neucleotides


def bind_analysis(df):
    neucs = [{'A':0, 'G':0, 'C':0, 'T':0} for i in range(15)]
    lengths = [0 for _ in range(16)]
    neuc_note = neucleotides
    def add_neucs(seq):

        for i in range(len(seq.replace(',','').upper())):
            for j in neuc_note[seq.replace(',','').upper()[i]]:
                neucs[i][j]+=1/len(neuc_note[seq.replace(',','').upper()[i]])
        lengths[len(seq.replace(',','').upper())]+=1


    
    df.bind.map(add_neucs)
    for i in range(len(neucs)):
        for key in neucs[i].keys():
            neucs[i][key] = int(neucs[i][key])
    return neucs, lengths

def length_analysis(df):
    lens = []
    plen = []
    nnlen = []
    for index, row in df.iterrows():
        lens.append(len(row['seq'])/len(row['bind']))
        plen.append(len(row['seq']))
        nnlen.append(len(row['bind']))
    from scipy.stats import pearsonr
    return sum(lens)/len(lens), pearsonr(plen, nnlen), lens

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
   
    df = pd.read_parquet(cfg.io.output, engine='pyarrow')
    print('columns:', df.columns)
    print('Total:', df.shape[0])
    print(df.dropna())
    df = df.dropna()
    print('Contains DNA bind site:', df.shape[0])
    print('Longest protein seq:', df.seq.map(len).max())
    print('Longest bind site seq:', df.bind.map(len).max())
    neucs, lengths = bind_analysis(df)
    print('Neucleotides per position: ', neucs)
    print('Number of bind sites of length [idx]:', lengths)
    mean,pearson, lens = length_analysis(df)
    print('Mean Protein seq length / Bind seq length:', mean)
    print('Pearsonr Protein seq length vs Bind seq length:', pearson)



if __name__ == "__main__":
    main()