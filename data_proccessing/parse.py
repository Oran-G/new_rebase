import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    '''
    In the meantime, you can get started on some data analysis. Make plots of the following:
        - number of sequences in REBASE
        - fraction with DNA labels, fraction that don’t have 
        - what the DNA labels look like (are they all pretty similar?)
    '''
    with open(cfg.io.input, 'r') as f:
        coke = f.readlines()
        df = pd.DataFrame(columns=['name', 'seq', 'bind'])
        start = False
        seq, name, bind = '', '', ''
        total, binds = 0, 0
        
        for line in tqdm(coke):
            if line == '\n':
                if start ==  True:
                    # print(seq, name, bind)
                    # if bind !='':
                    if True:
                        print('t')
                        df.append({'name':name, 'seq':seq, 'bind': bind.replace(',','') if bind != '' else None}, ignore_index=True)
                    
                    seq = ''
                    name = ''
                    bind = ''
                    start = False
                    
            elif line[0] == '>':
                l = line.split(' ')
                name = l[0].replace('>','')
                bind = l[3]
                b=False
                start = True
                total+=1
                if bind !='':
                    binds+=1
                    
                # print(line.split(' '))
                # if bind >5:
                #     quit()
                # bind += 1
            else:
                seq+=line.replace(' ','').replace('\n','')
                
        # table = pa.Table.from_pandas(df)
        print('Total:', total)
        print('Contains DNA bind site:', binds)
        # df.dropna()
        # pq.write_table(table, 'args.output')
        print(df)
        df.to_csv(cfg.io.output)

        f.close() 


if __name__ == "__main__":
    main()
