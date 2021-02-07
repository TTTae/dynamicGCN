import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import pickle


def reinitialize_tbatches():
    global current_tbatches_user, current_tbatches_item, current_tbatches_timestamp
    global tbatchid_user, tbatchid_item

    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    #current_tbatches_ns_label = defaultdict(list)
    #current_tbatches_interaction_dim =defaultdict(list)

    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)



filename ='epinion_with_rating_timestamp_txt/ts_interaction_dict.pkl'
infile = open(filename, 'rb')
ts_interaction_dict = pickle.load(infile)
infile.close()
print(len(ts_interaction_dict))

ts_list = list(ts_interaction_dict.keys())

cached_tbatches_user = {}
cached_tbatches_item = {}
reinitialize_tbatches()
for i in ts_list:
    interaction_data = ts_interaction_dict[i]
    user_sequence_id = interaction_data[:, 0]
    item_sequence_id = interaction_data[:, 1]

    for j in range(len(user_sequence_id)):
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]

        # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
        tbatch_to_insert = max(tbatchid_user[userid], tbatchid_item[itemid]) + 1
        tbatchid_user[userid] = tbatch_to_insert
        tbatchid_item[itemid] = tbatch_to_insert

        current_tbatches_user[tbatch_to_insert].append(userid)
        current_tbatches_item[tbatch_to_insert].append(itemid)

    print("Adding one timestamp finished")

    cached_tbatches_user[i] = current_tbatches_user
    cached_tbatches_item[i] = current_tbatches_item

    reinitialize_tbatches()
    tbatch_to_insert = -1

print(len(cached_tbatches_user))
print(len(cached_tbatches_item))




