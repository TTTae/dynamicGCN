import numpy as np
import pandas as pd
import pickle


# # adding dim information
#
# filename ='epinion_with_rating_timestamp_txt/ts_interaction_dict_v2.pkl'
# infile = open(filename, 'rb')
# ts_interaction_dict = pickle.load(infile)
# infile.close()
# print(len(ts_interaction_dict))
#
# filename1 = 'epinion_with_rating_timestamp_txt/dim1_dict_v3.pkl'
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
# filename = 'final_ts_dim_dict_v3.pkl'
# outfile = open(filename,'wb')
# pickle.dump(final_ts_dim_dict,outfile)
# outfile.close()




filename ='epinion_with_rating_timestamp_txt/ts_interaction_dict_test_v2.pkl'
infile = open(filename, 'rb')
ts_interaction_dict_test = pickle.load(infile)
infile.close()
print(len(ts_interaction_dict_test))

filename1 = 'epinion_with_rating_timestamp_txt/dim1_dict_v3_test.pkl'
infile = open(filename1, 'rb')
dim1_dict = pickle.load(infile)
infile.close()
print(len(dim1_dict))


filename2 = 'epinion_with_rating_timestamp_txt/dim2_dict'
infile = open(filename2, 'rb')
dim2_dict = pickle.load(infile)
infile.close()
print(len(dim2_dict))

final_ts_dim_dict_test = {}

for k, v in ts_interaction_dict_test.items():
    # dim 1
    v_info = v[:, :2]
    print("v info: ", v_info.shape)
    dim1_info = dim1_dict[k]
    print("dim 1 info: ", dim1_info.shape)
    dim1_info = np.vstack((v_info, dim1_info))
    print("all info: ", dim1_info.shape)
    dim1_list = [1]*len(dim1_info)
    dim1_list = np.array(dim1_list)
    dim1_list = dim1_list.reshape(-1, 1)
    dim1_info = np.concatenate((dim1_info, dim1_list), axis=1)
    print("Final size: ", dim1_info.shape)
    print(dim1_info)

    # dim 2
    v_copy_dim2 = v[:, :2]
    print("v info2: ", v_copy_dim2.shape)
    dim2_info = dim2_dict[k]
    print("dim 2 info: ", dim2_info.shape)
    dim2_info = np.vstack((v_copy_dim2, dim2_info))
    print("all dim 2 info: ", dim2_info.shape)
    dim2_list = [2] * len(dim2_info)
    dim2_list = np.array(dim2_list)
    dim2_list = dim2_list.reshape(-1, 1)
    dim2_info = np.concatenate((dim2_info, dim2_list), axis=1)
    print("Final size: ", dim2_info.shape)
    print(dim2_info)

    final_dim_info = np.vstack((dim1_info, dim2_info))
    print(final_dim_info.shape)
    final_ts_dim_dict_test[k] = final_dim_info


print(len(final_ts_dim_dict_test))
# filename = 'final_ts_dim_dict_test_v3.pkl'
# outfile = open(filename, 'wb')
# pickle.dump(final_ts_dim_dict_test, outfile)
# outfile.close()


