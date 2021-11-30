import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
file = pd.read_csv('data/anime.csv')
# for test
file = file.head(10000)
# count genre_number
genre_type = {}
genre_number = 0

type_dic = {}
type_number = 0

new_anime_id_dic = {}
anime_id = 0

real_index = 0
for index,row in file.iterrows():
    if not type(row['genre']) == str:
        continue
    new_anime_id_dic[row['anime_id']] = anime_id
    anime_id+=1
    a = list(map(str,row['genre'].replace(" ","").split(",")))
    b = row['type']
    if b not in type_dic:
        type_dic[b] = type_number
        type_number+=1
    for genre in a:
        if genre not in genre_type:
            genre_type[genre] = genre_number
            genre_number+=1
print(genre_number,type_number)

#anime_vector = [0 for _ in range(genre_number+type_number)]
anime_attributes = [[] for _ in range(anime_id)]


for index,row in file.iterrows():
    if not type(row['genre']) == str:
        continue
    a = list(map(str, row['genre'].replace(" ", "").split(",")))
    anime_vector = [0 for _ in range(genre_number + type_number)]
    for genre in a:
        anime_vector[genre_type[genre]] = 1
    anime_vector[genre_number+type_dic[row['type']]] = 1
    anime_attributes[real_index] = anime_vector
    real_index+=1

print(anime_attributes)
anime_attributes = np.array(anime_attributes)


# test
user_id =  123

## test
def get_user_like(user_id):
    ##hard code to 13
    return 13

cosine_sim = cosine_similarity(anime_attributes)
like_anime_index = get_user_like(user_id)
similar_anime = list(enumerate(cosine_sim[like_anime_index]))
recommendation = sorted(similar_anime,key=lambda x:x[1],reverse=True)
i = 0
for anime in recommendation:
    print(anime[0])
    i+=1
    if i==5:
        break








