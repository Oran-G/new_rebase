import pandas as pd
import pickle
from typing import List
import psa
import seaborn as sns 
import editdistance
#pip install pairwise-sequence-alignment
def analyze_data(path):
    # Read the CSV file
    data = pd.read_csv(path)
    
    # Split the data into labeled and unlabeled proteins
    labeled_data = data[data['bind'].notnull()]
    unlabeled_data = data[data['bind'].isnull()]
    
    # Get the clusters of proteins in the test set
    test_clusters = set(data[data['split'] == 2]['cluster'])
    
    # Create a set of unlabeled proteins with the same cluster as a protein in the test set
    unlabeled_proteins_with_test_cluster = unlabeled_data[unlabeled_data['cluster'].isin(test_clusters)]
    
    return labeled_data, unlabeled_proteins_with_test_cluster
class DataAnalyzer:
    def __init__(self, full, unlabeled, combined_path=None):
        if combined_path:
            with open(combined_path, 'rb') as f:
                self.data = pickle.load(f)
                self.labeled = self.data['labeled']
                self.unlabeled = self.data['unlabeled']
                self.combined = self.data['combined']
        else:
            self.labeled = pd.read_csv(full)
            self.unlabeled = pd.read_csv(unlabeled)
            self.combined = self.labeled.copy()
            self.combined['labeled'] = self.labeled['bind'].notnull()
            # for every unlabeled protein, replace the bind site of that row with the seq_greedy from the corresponding row in the unlabeled dataset (using the id column)
            for idx, row in self.unlabeled.iterrows():
                self.combined.loc[self.combined['id'] == row['id'], 'bind'] = row['seq_greedy']
            with open(combined_path, 'wb') as f:
                pickle.dump({'labeled': self.labeled, 'unlabeled': self.unlabeled, 'combined': self.combined}, f)
        self.unlabeled_labeled_similarity = None
    def homology_search(self, id):
        #return all proteins in self.data with 'test' in the split, and with the same cluster as the protein with the given id. also add a row to the returned dataframe that mentions if the protein has the same bind site as the protein with the given id
        cluster = self.combined[self.combined['id'] == id]['cluster'].values[0]
        test_proteins = self.combined[(self.combined['split'] == 2) & (self.combined['cluster'] == cluster)]
        test_proteins['same_bind'] = test_proteins['bind'] == self.combined[self.combined['id'] == id]['bind'].values[0]
        return test_proteins
    def bind_groups(self):
        #this function should return a dictionary where the keys are the bind sites and the values are a list of protein dicts of type {'id': id, 'seq': seq, 'cluster': cluster 'labeled': labeled} for every protein in the dataset with that bind site
        bind_groups = {}
        for idx, row in self.combined.iterrows():
            if row['bind'] not in bind_groups:
                bind_groups[row['bind']] = []
            bind_groups[row['bind']].append({'id': row['id'], 'seq': row['seq'], 'cluster':row['cluster'], 'labeled': row['labeled']})
        return bind_groups
    def multiclass_search(self):
        #return bind_groups thatbhave multiple clusters present
        #this sees if there are examples of proteins with the same bind site but different clusters
        #if the proteins from different clustered are labeled, it suggests convergent evolution. if they are unlabeled, it suggests that possiblythe model is learning the active site, and how that is the only important part of the protein.
        bind_groups = self.bind_groups()

        wierd_groups = {}  
        for bind, proteins in bind_groups.items():
            clusters = set()
            for protein in proteins:
                clusters.add(protein['cluster'])
            if len(clusters) > 1:
                wierd_groups[bind] = proteins
        return wierd_groups

    def calculate_percentage(self):
        # Get the clusters of unlabeled proteins
        labeled_clusters = set(self.labeled['cluster'])
        
        # Count the number of unlabeled proteins with at least one labeled protein in their cluster
        count = 0
        for row in self.unlabeled.iterrows():
            if row['cluster'] in labeled_clusters:
                count += 1
        
        # Calculate the percentage
        percentage = (count / len(self.unlabeled)) * 100
        
        return percentage
    def topk_similarity(self, k: int):
        # Calculate the sequence similarity between test proteins and unlabeled proteins
        if self.unlabeled_labeled_similarity is None:
            similarity_scores = []
            for unlabeled_id, unlabeled_row in self.unlabeled.iterrows():
            
                test_cluster = test_row['cluster']
                test_bind = test_row['bind']
                
                # Call homology_search for each unlabeled protein in the same cluster
                test_proteins = self.homology_search(test_id)
                
                
                for test_id, test_row in test_proteins.iterrows():
                    similarity = calculate_similarity(test_row['seq'], unlabeled_row['seq'])
                    similarity_scores.append({'test_id': test_row['id'], 'unlabeled_id': unlabeled_row['id'], 'similarity': similarity})
            
            # Sort the similarity scores in descending order
            similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return the top k similarity scores
            self.unlabeled_labeled_similarity = similarity_scores
            return similarity_scores[:k]
        else:
            return self.unlabeled_labeled_similarity[:k]
    def calculate_similarity(self, seq1: str, seq2: str): -> float:
        # Calculate the similarity between two sequences
        return psa.needle(moltype='prot', qseq=seq1, sseqs=seq2).score / 100
    def recall(self, targets: List[str], predictions: List[str]):
        # Calculate the recall of the predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                correct += 1
        
        return correct / len(targets)
    def precision(self, targets: List[str], predictions: List[str]):
        # Calculate the neucleotide-wise precision of the predictions. 
        #output the mean accuracy, standard deviation, and list of accuracies fro each target 
        accuracies = []
        for target, prediction in zip(targets, predictions):
            correct = 0
            for t, p in zip(target, prediction):
                if t == p:
                    correct += 1
            accuracies.append(correct / len(target))
        return np.mean(accuracies), np.std(accuracies), accuracies
class TestAnalyzer():
    def __init__(self, data):
        self.data = pickle.load(open(data, 'rb'))
        new_data = []
        for row in self.data:
            row = self.accuracy(row)
            row = self.edit_distance(row)
            new_data.append(row)
        self.data = new_data

        self.df = pd.DataFrame.from_records(self.data)

    def accuracy(self, row):
        #return the mean accuracy of the model on the test set
        #store the accuracy between the predicted sequence and the actual sequence, stored in row['predicted'] and row['bind']
        correct = 0 
        for t, p in zip(row['bind'], row['predicted']):
            if t == p:
                correct += 1
        predicted_accuracy = correct / len(row['bind'])
        #store the accuracy between the generated sequence and the actual sequence, stored in row['predicted'] and row['bind']. these sequences might be different lengths
        row['predicted_accuracy'] = predicted_accuracy
        return row
    def edit_distance(self, row):
        row['generated_accuracy'] = editdistance.eval(row['bind'], row['generated'])
        return row
        
    def plot(self, mode='predicted'):
        if mode not in ['predicted', 'generated']:
            raise ValueError('mode must be either "predicted" or "generated"')
        #seaborn line plot of the accuracies, sorted by the accuracy

        sns.lineplot(x=range(len(self.df)), y=self.df[f'{mode}_accuracy'].sort_values(), title='Accuracy of the model on the test set', xlabel='Test protein', ylabel='Accuracy')
    def recall(self, mode='predicted'):
        #return the mean recall of the model on the test set
        if mode not in ['predicted', 'generated']:
            raise ValueError('mode must be either "predicted" or "generated"')
        correct = self.df[self.df[f'{mode}_accuracy'] == 1]

