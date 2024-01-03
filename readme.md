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
