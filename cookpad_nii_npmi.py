import re
import pandas as pd
import codecs
import pickle
import numpy as np
import MeCab
import sys
import collections
import math
import copy 

file_name = input('file name ?')
#with open('./dataset/count_{}.pickle'.format(file_name), 'rb') as f:
#    count = pickle.load(f)
ingred_data = pd.read_csv('./dataset/ingred_cooccur_{}.csv'.format(file_name),  encoding='ms932',index_col=0, sep=',',skiprows=0) 
#ingred_data = ingred_data.rename(columns={'total': 'チーズオール'})
#ingred_data = ingred_data.iloc[:50,:]
ingred_col_df = pd.read_csv('./dataset/ingred_columns_words_count_{}.csv'.format(file_name),  encoding='ms932',index_col=0, sep=',',skiprows=0) 
ingred_row_df = pd.read_csv('./dataset/ingred_row_words_count_{}.csv'.format(file_name),  encoding='ms932',index_col=0, sep=',',skiprows=0) 
ingred_columns_words_prob={}


no_of_total_recipes = 17247

for ingred_name, prob in ingred_col_df.iterrows():
    print(ingred_name)
    print(prob.values[0])
    ingred_columns_words_prob[ingred_name]=prob.values[0]/no_of_total_recipes
ingred_row_words_prob={}
for ingred_name, prob in ingred_row_df.iterrows():
    # if ingred_name =='しょうゆ' or ingred_name=='にんにく':
    #     print(ingred_name,prob.values) 
    ingred_row_words_prob[ingred_name]=prob.values[0]/no_of_total_recipes


'''
with open('./dataset/ingred_columns_words_prob.pickle', 'rb') as f:
    ingred_columns_words_prob = pickle.load(f)
with open('./dataset/ingred_row_words_prob.pickle', 'rb') as f:
    ingred_row_words_prob = pickle.load(f)
'''

npmi_data =[]
index_name =[]
count_data =[]
for ingred_name,freq_vec in ingred_data.iterrows():        
    #print(freq_vec)
    if ingred_name not in ingred_row_words_prob:
        continue
    joint_probs = [ p/no_of_total_recipes for k,p in freq_vec.items()]
    ingred_columns_prob = [ingred_columns_words_prob[c_name] for c_name, f in freq_vec.items()]
    #print(ingred_columns_prob)
    
        
    ingred_row_prob = ingred_row_words_prob[ingred_name]
    npmi_vec = np.zeros(len(joint_probs))
    for i in range(len(joint_probs)):
        #print(ingred_columns_prob[i])
        #print(ingred_row_prob)
        if -np.log(joint_probs[i])==0 or ingred_row_prob*ingred_columns_prob[i]==0 or joint_probs[i]==0:
            npmi_vec[i] = -0.99
        else :
            npmi_vec[i] = np.log(joint_probs[i]/(ingred_columns_prob[i]*ingred_row_prob))/(-np.log(joint_probs[i]))  
    #npmi_vec = np.append(npmi_vec) 
    npmi_data.append(npmi_vec)
    index_name.append(ingred_name)
    count_data.append(freq_vec.values)
    
npmi_data =np.array(npmi_data)
count_data = np.array(count_data)
#print(count_data)
var_list = []
df_list = []
for i in range(npmi_data.shape[1]):
    var_list.append('variable_name_{}'.format(str(i)))
    df_list.append('df_{}'.format(str(i)))

ingred_columns =  ingred_data.columns.tolist()
ingred_names = np.array(index_name).reshape(-1,1)
#print(ingred_names)
concat_list=[]
for i in range(npmi_data.shape[1]):

    variable_name =  var_list[i]
    
    df_name = df_list[i]
    count_index = np.where(count_data<10)
    npmi_data[count_index] = 0
    con=np.concatenate([ingred_names,npmi_data[:,i].reshape(-1,1),count_data[:,i].reshape(-1,1)], axis=1)
    
    #df = pd.DataFrame(con, columns = ['ingred_name',ingred_columns[i],'count'])
    df = pd.DataFrame(con, columns = [ingred_columns[i],'NPMI','count'])
    df = df.reset_index(drop=True)
    # 共起素材名，NPMI, 共起頻度を列とするdataframeをさくせいし(df)、df_1, df_2 のようにそれぞれ変数名を変えて放り込み、concat_listにappendする
    exec("df_name = df")
    exec("concat_list.append(df_name)")
    
    
# print(npmi_data)
# print(npmi_data.shape[1])
for i in range(npmi_data.shape[1]):
    # concat_list には　見出し素材名毎に　共起素材名　NPMI countを要素とするdataframeが入っている。
    concat_list[i] = concat_list[i].astype({'NPMI':'float64'})#
    concat_list[i]=concat_list[i].sort_values('NPMI', ascending=False)
    concat_list[i] = concat_list[i].dropna()
    concat_list[i] = concat_list[i].reset_index(drop=True)
    #print(concat_list[i])
# 列見出し素材毎のdataframe concat_list[i]をすべて列方向に連結する
npmi_df = pd.concat(concat_list,axis =1)

    

'''
columns.append('total count')
print(columns)
npmi_df = pd.DataFrame(npmi_data ,columns= columns, index=index_name)
npmi_df = npmi_df.sort_values('チーズオール', ascending=False)
'''
with codecs.open("./dataset/ingred_cooccur_npmi_{}.csv".format(file_name), "w", "ms932", "ignore") as f: 
    npmi_df.to_csv(f, index=False, encoding="ms932", mode='a', header=True)

