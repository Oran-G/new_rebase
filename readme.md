  Restriction endonucleases (REs) are highly impactful tools in genetic engineering, but their inability to be programmed has hindered their utility compared to new techniques like CRISPR. Unlike CRISPR, every RE is specific to a DNA cutting site. For many sites in the genome, no RE can be found. The ability to discover or generate new REs on-demand that specifically target a desired DNA cut site could fundamentally change the tool of choice for biochemical research, industrial biotechnology, therapeutic applications, and more. In this project, I developed AI-GENRE, a deep learning-based tool for the design of novel REs with desired DNA cut sites. To build this model, I first hypothesized that a coherent structural understanding of REs would be critical to addressing this task. Unfortunately, there is a paucity of open-source RE crystal structures in the protein data bank (PDB), so I therefore created the Restriction Enzyme Structure Atlas (RESA), which is the first comprehensive dataset of predicted restriction enzyme protein structures. Starting with a set of 61,180 candidate REs and their DNA cut sites, I developed a scalable AlphaFold2 inference pipeline to predict the structures of all candidate REs. After filtering for high confidence structures, I recovered 52,958 RE structures, which were predicted by AlphaFold2 to have near experimental accuracy (mean pLDDT = 90.8). These predicted structures were paired with known DNA cut sites and served as the dataset for AI-GENRE. To further improve performance of AI-GENRE, I next hypothesized that the predictions could be improved by pre-training AI-GENRE on a large compendium of protein structures. To that end, I used a neural network (ESM-IF) that was trained on an inverse protein foldingŽ task on a massive compendium of 10M+ structures. I posited that a neural network trained for protein sequence design (protein structure -> protein sequence) could be fine-tuned to predict DNA cut sites of REs (protein structure -> DNA sequence). Indeed, fine-tuning ESM-IF on RESA resulted in state-of-the-art performance on predicting the cut sites of held-out REs. After training dozens of AI-GENRE configurations and saving the highest performing model, I selected 2000 unique proteins from RESA where the bind sites are significantly different from proteins in the training set to test AI-GENREs application on unknown enzymes. I predicted their binding sites from scratch, finding that the model achieved an 86% top-p accuracy on the generated bind sites. Notably, AI-GENRE recovered the bind site perfectly in 63% of cases compared to 20% on previous baseline, highlighting the utility of the model to determine the binding site of new restriction enzymes with unprecedented accuracy. Furthermore, I calculated the edit distances on the same held-out group, finding that on average, predictions were off by less than 1 edit, compared to 4.3 edits on a sequence-based neural network. Finally, I show that AI-GENRE can be used to design REs for new DNA cut sites, where no REs have previously been found. The ability to accurately predict DNA cut sites in silico provides a scalable method to generate a wide array of new REs for a variety of applications. On-demand generation of new REs with AI-GENRE will greatly improve the scalability of restriction enzymes as a gene editing tool by providing scientists with the ability to target any binding site with ease.


# CODEBASE STRUCTURE
## Modeling: Code to predict bind site based on sequence alone
## Folding: Code to predict bind site based on structure
## Diffusion: WOP, used to desing novel enzymes based on bind site










# REBASE data download and parse
### Download data from FTP
In Data repo:
> brew install inetutils
>ftp ftp.neb.com     (username: anonymous, password: your email address)
>cd pub/rebase       (to enter the correct directory)
>dir                 (to list the names of available files)
>get Readme |more    (to view the file called Readme, spaces matter!)
>get Readme          (to copy the file called Readme to your machine)
>quit                (returns you back to your local system).

On redhat:
```
module load ncftp/3.2.6
ncftp ftp.neb.com
cd pub/rebase
dir
get type2.110
get type2ref.110
get type2ref.txt
get type2.txt
get Type_II_methyltransferase_genes_DNA.txt
get Type_II_methyltransferase_genes_Protein.txt
get All_Type_II_restriction_enzyme_genes_Protein.txt
get All_Type_II_restriction_enzyme_genes_DNA.txt
exit
```

Update io yaml to have 
finput:  (fasta, input file from rebase)
final: (csv, path to final file)
temp: (tsv, path of temporary clustering data that can be later removed)
then run 
> python3 to_csv.py



To run training experiments 
add checkpoints to yaml file (path to model storage)
change model configs to have the number of GPUs availible
install requirements

python3 modeling.py model=lightning,deep,wide,both --multirun
OR for single experiment
python3 modeling.py 
