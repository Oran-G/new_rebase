import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
path = '/vast/og2114/rebase/20220428/output'
l = list()
nines=0
total = 0
for prot in os.listdir(path):
    if prot != 'AbaC258ORFBAP':
        pl = json.load(open(f'{path}/{prot}/ranking_debug.json'))['plddts'].values()
        l.append(int(max(pl)))
        if max(pl) >= 85:
            nines+=1
        total+=1

print("Above 85: "+ str(nines/total))
print(type(l[0]))
# print('hi!')
print(list(zip(l))[0:20])
print((pd.DataFrame(l, columns=['plddts'])))
# quit()
print(type(list(zip(l))))
sns.kdeplot(pd.Series(l))
# plt.hist(l, alpha=0.1)
plt.savefig(fname='/scratch/og2114/rebase/plddt.png')
plt.show()

