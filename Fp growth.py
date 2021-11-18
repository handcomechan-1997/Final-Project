import fp_growth as fpg
import numpy as np

rating_matrix = np.load('rating_matrix.npy')
row = len(rating_matrix)
col = len(rating_matrix[0])

transaction = []

for i in range(row):
    tmp = []
    for j in range(col):
        if rating_matrix[i][j]>=8:
            tmp.append(j)
    transaction.append(tmp)

frequent_itemset = fpg.find_frequent_itemsets(transaction,10,include_support=True)
print(frequent_itemset)


