import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle


# def init_feature(num_user, num_item, feature_size):
#     initializer = nn.init.xavier_uniform_
#     user_emb = nn.Embedding(num_user, feature_size)
#     user_emb.weight = nn.Parameter(initializer(torch.empty(num_user, feature_size)), requires_grad=False)
#     item_emb = nn.Embedding(num_item, feature_size)
#     item_emb.weight = nn.Parameter(initializer(torch.empty(num_item, feature_size)), requires_grad=False)
#     return user_emb, item_emb
#
#
# initializer = nn.init.xavier_uniform_
# # user_emb = nn.Parameter(initializer(torch.empty(1000, 64)), requires_grad=False)
# # item_emb = nn.Parameter(initializer(torch.empty(2000, 64)), requires_grad=False)
#
# user_emb, item_emb = init_feature(10, 20, 16)
#
# unique_node_list = [0, 1, 2, 3]
# emb_matrix = user_emb(torch.LongTensor(unique_node_list))


# adding dim information
#
# filename ='epinion_with_rating_timestamp_txt/ts_interaction_dict.pkl'
# infile = open(filename, 'rb')
# ts_interaction_dict = pickle.load(infile)
# infile.close()
# print(len(ts_interaction_dict))
#
# filename1 = 'epinion_with_rating_timestamp_txt/dim1_dict_v2.pkl'
# infile = open(filename1, 'rb')
# dim1_dict = pickle.load(infile)
# infile.close()
# print(len(dim1_dict))
#
#
# filename2 = 'epinion_with_rating_timestamp_txt/dim2_dict'
# infile = open(filename2, 'rb')
# dim2_dict = pickle.load(infile)
# infile.close()
# print(len(dim2_dict))
#
# final_ts_dim_dict = {}
#
# for k, v in ts_interaction_dict.items():
#     # dim 1
#     v_info = v[:, :2]
#     print("v info: ", v_info.shape)
#     dim1_info = dim1_dict[k]
#     print("dim 1 info: ", dim1_info.shape)
#     dim1_info = np.vstack((v_info, dim1_info))
#     print("all info: ", dim1_info.shape)
#     dim1_list = [1]*len(dim1_info)
#     dim1_list = np.array(dim1_list)
#     dim1_list = dim1_list.reshape(-1, 1)
#     dim1_info = np.concatenate((dim1_info, dim1_list), axis=1)
#     print("Final size: ", dim1_info.shape)
#     print(dim1_info)
#
#     # dim 2
#     v_copy_dim2 = v[:, :2]
#     print("v info2: ", v_copy_dim2.shape)
#     dim2_info = dim2_dict[k]
#     print("dim 2 info: ", dim2_info.shape)
#     dim2_info = np.vstack((v_copy_dim2, dim2_info))
#     print("all dim 2 info: ", dim2_info.shape)
#     dim2_list = [2] * len(dim2_info)
#     dim2_list = np.array(dim2_list)
#     dim2_list = dim2_list.reshape(-1, 1)
#     dim2_info = np.concatenate((dim2_info, dim2_list), axis=1)
#     print("Final size: ", dim2_info.shape)
#     print(dim2_info)
#
#     final_dim_info = np.vstack((dim1_info, dim2_info))
#     print(final_dim_info.shape)
#     final_ts_dim_dict[k] = final_dim_info
#
#
# print(len(final_ts_dim_dict))
# filename = 'final_ts_dim_dict_v2.pkl'
# outfile = open(filename,'wb')
# pickle.dump(final_ts_dim_dict,outfile)
# outfile.close()






# adding dim information on test set

# filename ='epinion_with_rating_timestamp_txt/ts_interaction_dict_test.pkl'
# infile = open(filename, 'rb')
# ts_interaction_dict_test = pickle.load(infile)
# infile.close()
# print(len(ts_interaction_dict_test))
#
# filename1 = 'epinion_with_rating_timestamp_txt/dim1_dict_v2_test.pkl'
# infile = open(filename1, 'rb')
# dim1_dict = pickle.load(infile)
# infile.close()
# print(len(dim1_dict))
#
#
# filename2 = 'epinion_with_rating_timestamp_txt/dim2_dict'
# infile = open(filename2, 'rb')
# dim2_dict = pickle.load(infile)
# infile.close()
# print(len(dim2_dict))
#
# final_ts_dim_dict_test = {}
#
# for k, v in ts_interaction_dict_test.items():
#     # dim 1
#     v_info = v[:, :2]
#     print("v info: ", v_info.shape)
#     dim1_info = dim1_dict[k]
#     print("dim 1 info: ", dim1_info.shape)
#     dim1_info = np.vstack((v_info, dim1_info))
#     print("all info: ", dim1_info.shape)
#     dim1_list = [1]*len(dim1_info)
#     dim1_list = np.array(dim1_list)
#     dim1_list = dim1_list.reshape(-1, 1)
#     dim1_info = np.concatenate((dim1_info, dim1_list), axis=1)
#     print("Final size: ", dim1_info.shape)
#     print(dim1_info)
#
#     # dim 2
#     v_copy_dim2 = v[:, :2]
#     print("v info2: ", v_copy_dim2.shape)
#     dim2_info = dim2_dict[k]
#     print("dim 2 info: ", dim2_info.shape)
#     dim2_info = np.vstack((v_copy_dim2, dim2_info))
#     print("all dim 2 info: ", dim2_info.shape)
#     dim2_list = [2] * len(dim2_info)
#     dim2_list = np.array(dim2_list)
#     dim2_list = dim2_list.reshape(-1, 1)
#     dim2_info = np.concatenate((dim2_info, dim2_list), axis=1)
#     print("Final size: ", dim2_info.shape)
#     print(dim2_info)
#
#     final_dim_info = np.vstack((dim1_info, dim2_info))
#     print(final_dim_info.shape)
#     final_ts_dim_dict_test[k] = final_dim_info
#
#
# print(len(final_ts_dim_dict_test))
# filename = 'final_ts_dim_dict_test_v2.pkl'
# outfile = open(filename, 'wb')
# pickle.dump(final_ts_dim_dict_test, outfile)
# outfile.close()

#just check
filename = 'final_ts_dim_dict_test_v2.pkl'
infile = open(filename, 'rb')
test_dict = pickle.load(infile)
infile.close()
print(len(test_dict))


# try to_neigh_dim

# for dim in range(1,3):
#     to_neigh_dims = dict()
#     to_neigh_dims[dim] = dict()
#     to_neigh_dims[dim][1] = set([1])
#     to_neigh_dims[dim][1].add(2)
#     to_neigh_dims[dim][1].add(2)
#     print(to_neigh_dims)
#     break



########### just check

# my_data = np.loadtxt("epinion_with_rating_timestamp_txt/my_data.txt")
# source = my_data[:, 1]
# unique, cnt = np.unique(my_data, return_counts=True)
# print(unique)
# print(cnt)
#
#
# filename1 = 'epinion_with_rating_timestamp_txt/dim2_dict'
# infile = open(filename1, 'rb')
# dim1_dict = pickle.load(infile)
# infile.close()
# key_list = list(dim1_dict.keys())
# for i in range(len(key_list)):
#     data = dim1_dict[key_list[i]]
#     source = data[:, 0]
#     unique, cnt = np.unique(source, return_counts=True)
#     print(unique[0])
#     print(cnt[0])




