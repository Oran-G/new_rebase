from typing import Dict,List
import math
import random
'''
We build custom vector databases at Exa because they are so critical to our search. 
Your mission is to create a vector database from scratch.

The database should:
Allow user to add (url, embedding) pairs
Allow user to search topK urls given an embedding input

Notes:
Don’t use any external library like numpy
Try and write this as quickly as you can. We can think of how to optimize it later
'''
class Database():
    def __init__(self):
        self.data: List[Tuple[str, List[float]]] = []
    def add(self, url: str, embed: List[float]) -> None:
        self.data.append((url, embed))
    def search(self, embedding:List[float], K:int) -> List[str]:
        values: Tuple[List[int], List[str]] = []
        for d in range(len(self.data)):
            dist = 0
            for k in range(len(embedding)):
                dist = dist + abs(embedding[k] - self.data[d][1][k])
            values[0].append(dist)
            values[1].append(self.data[d][0])
    
        top_k = sorted(values[0], reverse=True)[:k]
        r: List[str] = []
        r2: List[int] = []
        for i in range(len(values[0])):
            if values[0][i] in top_k:
                r.append(values[1][i])
                r2.append(values[0][i])
        return r, r2
        
if __name__ == '__main__':
    database = Database()
    embed_dim = 10
    test_cases = 100
    k = 5
    for i in range(test_cases):
        database.add(f'link_{i}', [random.randint(0, 100)/100 for i in range(embed_dim)])
    searched = [random.randint(0, 100)/100 for i in range(embed_dim)]
    print(f'searching agaist embedding: {searched}')
    print(database.search(searched, k))
