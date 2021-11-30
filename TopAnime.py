import pandas as pd
import numpy as np

TOP_NUMBER = 10
file = pd.read_csv("data/rating.csv")
file = file[file['rating']!=-1]
file = file.head(10000)

# map original id to new id
anime_dict = {}
# map original id to rating_number
anime_number = {}

new_id = 0

for index,row in file.iterrows():
    #print(row['user_id'])
    if row['anime_id'] not in anime_dict:
        anime_dict[row['anime_id']] = new_id
        new_id+=1
        anime_number[row['anime_id']] = 1
    else:
        anime_number[row['anime_id']] +=1


a = sorted(anime_number.items(),key=lambda x:x[1],reverse=True)
print(a)

Top_anime_list = []
count = 0
for key in a:
    Top_anime_list.append(key[0])
    count+=1
    if count == TOP_NUMBER:
        break

print(Top_anime_list)
reduce_user_dict = {}
reduce_user_id = 0
reduce_anime_dict = {}
reduce_anime_id = 0

# focus on the top anime, and count the number of user
for index,row in file.iterrows():
    if row['anime_id'] in Top_anime_list:
        print(anime_dict[row['anime_id']])
        if row['anime_id'] not in reduce_anime_dict:
            reduce_anime_dict[row['anime_id']] = reduce_anime_id
            reduce_anime_id+=1
        if row['user_id'] not in reduce_user_dict:
            reduce_user_dict[row['user_id']] = reduce_user_id
            reduce_user_id+=1

## number of user, number of anime
m,n = reduce_user_id,reduce_anime_id

rating_matrix = [[0 for _ in range(n)]for _ in range(m)]

print(m,n)

for index,row in file.iterrows():
    if row['anime_id'] in Top_anime_list:
        rating_matrix[reduce_user_dict[row['user_id']]][reduce_anime_dict[row['anime_id']]] = row['rating']

print(rating_matrix)

rating_matrix = np.array(rating_matrix)
print(rating_matrix.shape)
np.save("rating_matrix_reduce_anime"+str(TOP_NUMBER)+".npy",rating_matrix)











# rating_matrix = np.array(rating_matrix)
# print(rating_matrix.shape)
#mnp.save("rating_matrix1000.npy",rating_matrix)

