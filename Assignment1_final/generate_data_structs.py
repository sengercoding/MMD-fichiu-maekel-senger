from data_util import load_movielens_tf

from config import ConfigLf
import shelve
from scipy.sparse import coo_array
import numpy as np

ratings_tf, user_ids_voc, movie_ids_voc = load_movielens_tf(ConfigLf)
rated_by = shelve.open("rated_by")
user_col = shelve.open("user_col")

#test = ratings_tf.take(1)
i=0
max_id = 0 
for example in ratings_tf:
    i+=1
    if i % 1000 == 0:
        print(str(i)+"k")
    if str(example["movie_id"].numpy()) in rated_by:

        temp = rated_by[str(example["movie_id"].numpy())]
        temp.append(example["user_id"].numpy())
    else:
        temp = [example["user_id"].numpy()]
    rated_by[str(example["movie_id"].numpy())] = temp
    
    if example["movie_id"].numpy() > max_id:
        max_id = int(example["movie_id"].numpy())
    
    if str(example["user_id"].numpy()) in user_col:
        temp2 = user_col[str(example["user_id"].numpy())]
        temp2.append( (example["movie_id"].numpy(), example["user_rating"].numpy()))
    else:    
        temp2 = [(example["movie_id"].numpy(), example["user_rating"].numpy())]
    user_col[str(example["user_id"].numpy())] = temp2

max_id += 1
for key in user_col:
    temp3 = user_col[key]
    temp4 =   list(zip(*temp3))
    idx = np.array(temp4[0])
    data = np.array(temp4[1])
    
    user_col[key] = coo_array((data, (np.zeros(len(idx)),idx)), shape=(1, max_id) )
                              
rated_by.close()
user_col.close()