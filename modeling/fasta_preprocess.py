import os
import argparse
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
def modify_fasta_labels(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        f_out.write('')  # Completely overwrite output_file before adding in the data
        for line in f_in:
            if line.startswith('>'):
                label = line.split()[0]
                f_out.write(f'{label}\n')
            else:
                f_out.write(line)


def create_embeddings(input_file: str, output_types: List[List[str]]):
    for model_name, path in output_types:
        print(model_name, path)
        print(f'python3 /vast/og2114/new_rebase/modeling/extract.py {model_name} {input_file} {path} --include per_tok')
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if it doesn't exist
        os.system(f'python3 /vast/og2114/new_rebase/modeling/extract.py {model_name} {input_file} {path} --include per_tok')
    
# Usage example
@hydra.main(config_path='../configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    

    os.system('export TORCH_HOME=/vast/og2114/torch_home')
    parser = argparse.ArgumentParser(
                    prog='ESM_embedder',
                    description='Embeds protein sequences using ESM')
    parser.add_argument('--model_name', nargs='*', 
        help='The name of the model to use for embedding', 
        default=['esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']
    )
    parser.add_argument('--input_path', type=str, 
        help='The name of the original fasta file', 
        default=f'{cfg.io.finput}'
    )
    parser.add_argument('--output_path', type=str, 
        help='The name of the fasta file to be created for embedding', 
        default=f'{cfg.io.fasta_for_esm_embedder}'
    )
    args = parser.parse_args()
    input_file = args.input_path
    output_file = args.output_path
    modify_fasta_labels(input_file, output_file)

    outputs = [[model_name, f'{cfg.io.embeddings_store_dir}/{model_name}'] for model_name in args.model_name]
    create_embeddings(output_file, outputs)
    
if __name__ == '__main__':
    main()
