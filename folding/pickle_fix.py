import pickle as pkl
import torch
import argparse
import tqdm
parser = argparse.ArgumentParser(
    prog='ESM_embedder',
    description='Embeds protein sequences using ESM')
parser.add_argument('--data_path', nargs='*', 
    help='path to the pickled data file', 
    default='/vast/og2114/output_home/runs/slurm_45989785/esm2_t30_150M_UR50D_test_data.pkl'
)
'/vast/og2114/output_home/runs/slurm_45989785/foldbig_test_data.pkl'
'/vast/og2114/output_home/runs/slurm_45989785/esm2_t33_650M_UR50D_test_data.pkl'  
'/vast/og2114/output_home/runs/slurm_45989785/fold_test_data.pkl'
args = parser.parse_args()
data = pkl.load(open(args.data_path, 'rb'))
new_data = []
for row in  enumerate(tqdm.tqdm(data)):
    new_data.append({
        'id': row['id'],
        'seq': row['seq'],
        'bind': row['bind'],
        'predicted': row['predicted'],
        'predicted_logits': row['predicted_logits'].to(torch.device('cpu')).tolist(),
        'generated': row['generated'],
    })
pkl.dump(new_data, open(args.data_path, 'wb'))