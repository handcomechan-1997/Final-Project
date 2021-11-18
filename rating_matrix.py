import pandas as pd
import numpy as np

file = pd.read_csv("data/rating.csv")
print(file.head(5))

user_dict = {}
anime_dict = {}

user_id_new = 0
anime_id_new = 0
for index,row in file.iterrows():
    print(type(row['user_id']))
    if row['user_id'] not in user_dict:
        user_dict[row['user_id']] = user_id_new
        user_id_new+=1
    if row['anime_id'] not in anime_dict:
        anime_dict[row['anime_id']] = anime_id_new
        anime_id_new +=1

m,n = user_id_new,anime_id_new

rating_matrix = [[0 for _ in range(n)] for _ in range(m)]
for index,row in file.iterrows():
    rating_matrix[user_dict[row['user_id']]][anime_dict[row['anime_id']]] = row['rating']

rating_matrix = np.array(rating_matrix)
print(rating_matrix)


