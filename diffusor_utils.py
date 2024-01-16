import torch
import math
from rfdiffusion.coord6d import get_coords6d
import rfdiffusion.kinematics as kinematics
PARAMS = { #see page 33/table 6
    **kinematics.PARAMS
    "T":200,
    "wtrans": 0.5, 
    "wrot":1.0, 
    "dclamp":10, 
    "gamma":.99, 
    "w2d":1.0,
    "lr":,
    "lr_decay":,
    "Bt0": .01, #formula is Bt0 + (t/t)(BtT - Bt0)
    "BtT": .07,
    "Br0": 1.06, #formula is (t*Br0) + .5*(t/T)**2(BrT - Br0)
    "BrT": 1.77,
    "crd_scale":0.25,

}
def gramschmidt(z: torch.tensor):
    #get r from z using gramschmidt method described pg. 3 supplemental methods
    if len(z.shape) == 4:
        batch = []
        for b in range(z.shape[0]):
            batch.append(gramschmidt(z[b]))
        return torch.stack(batch)
    else:
        residues = []
        for l in range(z.shape[0]):
            r1 = (z[l, 1] - z[l, 0])/torch.norm(z[l, 0] - z[l,  1])
            tmpr2 =  (z[l, 2] - z[l, 0]) - (torch.dot((z[l, 2] - z[l, 0]),  r1)* r1)
            r2 = (tmpr2)/torch.norm(tmpr2)
            r3 = torch.linalg.cross(r1, r2)
            residues.append(torch.stack([r1, r2, r3]))
        return torch.stack(residues)


def xyztox(xyz_27: torch.tensor):
    z = xyz_27[:, :3, :]
    y = gramschmidt(x)
    return (y, z)
def dframe(x, xpred, wtrans, wrot, dclamp):
    if len(x[0].shape) == 4:
        batch = []
        for b in range(x[0].shape[0]):
            batch.append(dframe((x[0, b], x[1, b]), (x_pred[0, b], x_pred[1, b]), wtrans, wrot, dclamp))
        return torch.stack(batch)
    else:
        total = 0
        for l in range(x[0].shape[0]):
            total += wtrans*(min(torch.linalg.norm((x[1] - x_pred[1]), ord=2).pow(2), dclamp)**2) + wrot*((torch.linalg.norm((torch.eye(3) - (xpred[0].t() @ x[0]))).pow(2))**2)
        return math.sqrt(total / x[0].shape[0])
def lframe(xyz_27, xyz_27_preds, wtrans, wrot, dclamp, gamma): #loss described on page 28
    if len(x[0].shape) == 5:
        batch = []
        for b in range(xyz_27.shape[0]):
            batch.append(lframe(xyz_27[b], xyz_27_preds[b], wtrans, wrot, dclamp, gamma))
        return torch.stack(batch)
    else:
        full_total = 0
        divisor = 0
        for i in range(xyz_27_preds.shape[0]): 
            # look more into gamma and structure of the xyz_27_preds, and see if time goes up or down with index. 
            #Gamma term should get larger loser to t=0
            gamma_term = (gamma**(xyz_27_preds.shape[0] - i))
            full_total += gamma_term*(dframe(xyztox(xyz_27), xyztox(xyz_27_preds[i]), wtrans, wrot, dclamp)**2)
            divisor += gamma_term
        return full_total / gamma_term
def l2d(logits_dist, logits_omega, logits_theta, logits_phi, xyz_27): #loss as described on page 30
    
    #c6d : pytorch tensor of shape [batch,nres,nres,4]
    #      stores stacked dist,omega,theta,phi 2D maps 
    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = kinematics.c6d_to_bins(kinematics.xyz_to_c6d(xyz_27)[0])
    dist, omega, theta, phi = c6d[..., 0], c6d[..., 1], c6d[..., 2] ,c6d[..., 3]
    return torch.nn.functional.cross_entropy(logits_dist, dist) + 
        torch.nn.functional.cross_entropy(logits_omega, omega) + 
        torch.nn.functional.cross_entropy(logits_theta, theta) +
        torch.nn.functional.cross_entropy(logits_phi,  phi)
def ldiffusion(xyz_27, xyz_27_preds, logits_dist, logits_omega, logits_theta, logits_phi, wtrans, wrot, dclamp, gamma, w2d):
    return lframe(xyz_27, xyz_27_preds, wtrans, wrot, dclamp, gamma) + 
        (w2d*l2d(logits_dist, logits_omega, logits_theta, logits_phi, xyz_27))


class RFdict():
    def __init__(self):
        self.one_letter = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "?", "-"] #20 AA, unk, mask. this is taken from rfidffusion.chemical, and will also be used for Neuc
        self.letter_idx = {self.one_letter[i]:i for i in range(len(self.one_letter))}
        self.pad_idx = 22
        self.eos_idx = 23
        self.padding_idx = 22
    def encode(self, line):
        idx = []
        for char in line:
            if char in self.one_letter:
                idx.append(self.letter_idx[char])
            else:
                idx.append(20)
        return idx
    def pad(self, line, length):
        idx = line
        for i in range(length - len(idx)):
            idx.append(self.pad_idx)
        return idx
    def eos(self, line):
        return line.append(eos_idx)

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

class CSVDataset(Dataset):
    def __init__(self, csv_path, split, split_seed=42, supervised=True, plddt=85, clust=False):
        super().__init__()
        """
        args:
            csv_path: path to data
            split: one of "train" "val" "test"
            split_seed: used for future work, not yet
            supervised: if True drop all samples without a bind site
            plddt: plddt cutoff for alphafold confidence
        """
        print('start of data')
        self.df = pd.read_csv(csv_path)
        if supervised:
            self.df = self.df.dropna()

        print("pre filter",len(self.df))

        def alpha(ids):
            return os.path.isfile(f'/vast/og2114/rebase/20220519/output/{ids}/ranked_0.pdb') and (max(json.load(open(f'/vast/og2114/rebase/20220519/output/{ids}/ranking_debug.json'))['plddts'].values()) >= plddt)
        #self.df  = self.df[self.df['id'].apply(alpha) ==True ]
        self.df = self.df[self.df['id'] != 'Csp7507ORF4224P']
        print("post filter",len(self.df))

        spl = self.split(split)
        self.data = spl[['seq','bind', 'id']].to_dict('records')
        print(len(self.data))
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.clustered_data = {}
        tmp_clust = self.df.cluster.unique()
        self.cluster_idxs =[]
        for cluster in tmp_clust:
            t = spl[spl['cluster'] == cluster][['seq','bind', 'id']].to_dict('records')
            if len(t) != 0:
                self.clustered_data[cluster]= spl[spl['cluster'] == cluster][['seq','bind', 'id']].to_dict('records')
                self.cluster_idxs.append(cluster)
        self.use_cluster=clust
        
        num = 0
        for data in self.data:
            
            if len(data['seq']) > 376:
                num += 1
        print("SEQUENCES OVER LENGTH 376: ", num)
        
        print('initialized', self.__len__())
    def __getitem__(self, idx):
        if self.use_cluster == False:
            return self.data[idx]
        else:
            return self.clustered_data[self.cluster_idxs[idx]][random.randint(0, (len(self.clustered_data[self.cluster_idxs[idx]])-1))]
    
    def __len__(self):
        if self.use_cluster== False:
            return len(self.data)
        else:
            return len(self.cluster_idxs)

    def split(self, split):
        '''
        splits data on train/val/test

        args:
            split: One of "train" "val" "test"

        returns:
            subsection of data included in the train/val/test split
        '''
        if split.lower() == 'train':
            tmp = self.df[self.df['split'] != 1]
            return tmp[tmp['split'] != 2]

        elif split.lower() == 'val':
            return self.df[self.df['split'] == 1]

        elif split.lower() == 'test':
            return self.df[self.df['split'] == 2]


class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens.   will usually have eos already applied,
        but won't have bos. Hence the defaults here.

        args:
            dataset: CSVDataset of data
            dictionary: esmif1 dictionary used in code
        '''

        super().__init__(dataset)
        self.dictionary = dictionary
        self.apply_bos = apply_bos
        self.apply_eos = apply_eos
        '''
        batchConverter git line 217 - https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py
        '''
        self.batch_converter_coords = esm.inverse_folding.util.CoordBatchConverter(self.dictionary)

    def __getitem__(self, idx):
        '''
        Get item from dataset:
        returns:
        {
            'bind': torch.tensor (bind site)
            'xyz_27': rf.inference.utils.process_target coords output
            'mask_27': rf.inference.utils.process_target mask output
            'seq': torch.tensor sequence
        } to be post-proccessed in self.collate_dicts()

        https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/inference/utils.py#L613
        '''

    
        proccessed = process_target(f"/vast/og2114/rebase/20220519/output/{self.dataset[idx]['id']}/ranked_0.pdb")
        print(proccessed.keys())
        MAXLEN = 271
        if torch.tensor(proccessed['xyz_27']).shape[0] >= MAXLEN:
            start = random.randint(0, torch.tensor(proccessed['xyz_27']).shape[0] - (MAXLEN+1))
            end = start + MAXLEN
        else:
            start = 0
            end = -1
        #import pdb; pdb.set_trace()
        return {
            'bind':torch.tensor( self.dictionary.encode(self.dataset[idx]['bind'])),
            'xyz_27': torch.tensor(proccessed['xyz_27'])[start: end], #tensor of atomic coords
            'mask_27': torch.tensor(proccessed['mask_27'][start: end]), #tensor of true/false for if the atoms exist
            'seq': torch.tensor(proccessed['seq'][start:end]), #tensor of idx
            'idx_pdb': proccessed['pdb_idx'][start:end],

        }

    def __len__(self):
        return len(self.dataset)

    def collate_tensors(self, batch: List[torch.tensor], bos=False, eos=False):
        '''
        utility for collating tensors together, applying eos and bos if needed,
        padding samples with self.dictionary.padding_idx as neccesary for length

        input:
            batch: [
                torch.tensor shape[l1],
                torch.tensor shape[l2],
                ...
            ]
            bos: bool, apply bos (defaults to class init settings) - !!!BOS is practically <af2>, idx 34!!!
            eos: bool, apply eos (defaults to class init settings)
        output:
            torch.tensor shape[len(input), max(l1, l2, ...)+bos+eos]
        '''
        
        if eos == None:
            eos = self.dictionary.eos()

        batch_size = len(batch)
        max_len = max(el.size(0) for el in batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(bos) + int(eos) # eos and bos
            ),
            dtype=torch.int64,
        ).fill_(self.dictionary.padding_idx)

        if bos:
            tokens[:, 0] = self.dictionary.get_idx('<af2>')

        for idx, el in enumerate(batch):
            tokens[idx, int(bos):(el.size(0) + int(bos))] = el

            # import pdb; pdb.set_trace()
            if eos:
                tokens[idx, el.size(0) + int(bos)] = self.dictionary.eos_idx

        return tokens



    def collater(self, batch):
        if isinstance(batch, list) and torch.is_tensor(batch[0]):
            return self.collate_tensors(batch)
        else:
            return self.collate_dicts(batch)
    def collate_dicts(self, batch: List[Dict[str, torch.tensor]]):
        '''
        combine sequences of the form
        [
            {
                'bind': torch.tensor (bind site)
                'xyz_27': rf.inference.utils.process_target coords output
                'mask_27': rf.inference.utils.process_target mask output
                'seq': torch.tensor sequence
                'idx_pdb': proccessed['idx_pdb'],
            },
            {
                'bind': torch.tensor (bind site)
                'xyz_27': rf.inference.utils.process_target coords output
                'mask_27': rf.inference.utils.process_target mask output
                'seq': torch.tensor sequence
                'idx_pdb': proccessed['idx_pdb'],
            },
        ]
        into a collated form:
        {
            'bind': torch.tensor (bind site)
            'bos_bind': torch.tensor (bos+bind site)
            
            'seq': torch.tensor (protein sequence)
            'xyz_27': torch.tensor (coords input to rf)
            'mask_27' torch.tensor (mask input to rf)
            'idx_pdb': list of idx_pdb in form [(chain, i) for i in len]
        }
        
        applying the padding correctly to capture different lengths
        '''

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]



        post_proccessed = {
            'bind': self.collate_tensors(select_by_key(batch, 'bind'), eos=False),
            
            'seq': self.collate_tensors(select_by_key(batch, 'seq')),
            'xyz_27': torch.stack(select_by_key(batch, 'xyz_27'), dim=0),
            'mask_27': torch.stack(select_by_key(batch, 'mask_27'), dim=0),
            'idx_pdb': select_by_key(batch, 'idx_pdb'),
            
        }
        return post_proccessed