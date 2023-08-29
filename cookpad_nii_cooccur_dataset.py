import json
import pandas as pd
import codecs
import pickle
import numpy as np
import json


file_name=input('input dish name')

#dataset = {'ingred_dataset':ingred_dataset, 'titles':titles,'summeries':summeries}

with open('./dataset/dataset.pickle', 'rb') as f:
    cookpad_dataset = pickle.load(f)
ingred_dataset=cookpad_dataset['ingred_dataset']
titles = cookpad_dataset['titles']  
summeries = cookpad_dataset['summeries']  
tsukurepos = cookpad_dataset['tsukurepos']
# ingred_dataset2=cookpad_dataset['ingred_dataset2']

# 以下のcookpad_nii_ingred_count{}.csvが、共起、npmiなど後続すべてのデータセット用の辞書のもとになるので、ノイズになる語彙はこの
# csvを開けて手作業で削除するとノイズを除去できる

f_name = './dataset/cookpad_nii_ingred_count{}.csv'.format(file_name) #ファイル名ごとに毎回変更
ingred_count = pd.read_csv(f_name,  encoding='ms932',index_col=0, sep=',',skiprows=0)
unique_dict = pd.read_csv('./dataset/unique_dict.csv',  encoding='ms932', sep=',',skiprows=0)
unique_dict = {u['var']:u['uniq'] for k,u in unique_dict.iterrows()}

count_ingreds={}
for ing, count in ingred_count.iterrows():
    # if ing in filter_words:
    #     continue
    if ing not in unique_dict:    
        count_ingreds[ing]= count.values[0]
    else:
        u_ing=unique_dict[ing]
        if u_ing in count_ingreds:            
            count_ingreds[u_ing]+= count.values[0]
        else:
            count_ingreds[u_ing]= count.values[0]
 
key_word = ["all","バター"]
 
 
ing_id=0
threshold = int(10)#今回の研究では10を入れる input('input freq threshold >:')
cookpad_nii_dic={}
for ing, count in count_ingreds.items():
    #print(count)
    if count > threshold:
        cookpad_nii_dic[ing] = ing_id
        ing_id+=1
cookpad_nii_dic = dict(sorted(cookpad_nii_dic.items(), key=lambda x:x[1]))
ingreds_dic_w_id = {i:k for k,i in cookpad_nii_dic.items()}


dic = [[w,k] for w,k in cookpad_nii_dic.items()]
dic_df = pd.DataFrame(dic, columns=['ingredient','id'])
with open('./dataset/ingred_dic.pickle'.format(file_name), 'wb') as f:
    pickle.dump(cookpad_nii_dic, f)
with open('./dataset/ingreds_dic_w_id_{}.pickle'.format(file_name), 'wb') as f:
    pickle.dump(ingreds_dic_w_id, f)
with codecs.open('./dataset/ingred_dic.csv', "w", "ms932", "ignore") as f:
    dic_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True) 



ingred_list = [ i for i,v in cookpad_nii_dic.items()]
bag_of_ingred={}
for c in key_word:    
    bag_of_ingred[c]={ing:0 for ing,ing_id in cookpad_nii_dic.items()}

def create_dataset_LDA(dataset_LDA,recipe_ingreds,cook_style):    
       
    dataset = dataset_LDA[cook_style] 
    dataset.append(recipe_ingreds)
    dataset_LDA[cook_style] = dataset

def update_bag_of_ingred(bag_of_ingred,ingred_list,cook_style):    

    ingred_dic = bag_of_ingred[cook_style] # 共起を調べたい素材の共起素材bow
    for ing_name in ingred_list:# レシピ毎の共起素材名リスト
        if ing_name != cook_style:
            ingred_dic[ing_name] +=1 # 共起頻度を更新 抽出対象のc_styleA,Bがあったとする。Aの共起素材頻度については、A=0, B=共起頻度　Bの共起素材頻度については、A=共起頻度　B=0のように書き出す
    bag_of_ingred[cook_style]=ingred_dic 



def update_ingred_count(data,ingred_count_dic):
    for d in data:
        ingred_count_dic[d]+=1

def create_dataset_filter(title,summery,data,cook_style):
    dataset = dataset_filter[cook_style]
    dataset.append([title,summery,data])
    dataset_filter[cook_style]=dataset
    
    
tsukurepo_no = int(input('threshold tsukurepo_no :'))#0を入れる
ingred_dataset_filter=[]
summeries_filer=[]
titles_filter=[]
tsukurepo_filter=[]
ingred_count_dic = {w:0 for w,i in cookpad_nii_dic.items()}
dataset_LDA = {c:[] for c in key_word}
dataset_filter = {c:[] for c in key_word}
kw_count_vec = np.zeros(len(key_word))
for ingred_data, title, summery, t in zip(ingred_dataset,titles,summeries, tsukurepos):
    print(t)
    if int(t) >= tsukurepo_no:
    
        ingred_data = [unique_dict[w] if w in unique_dict else w for w in ingred_data] 
        data = [d for d in ingred_data if d in  cookpad_nii_dic]  # ここで、後続の全てのデータセットについて単語をcookpad_nii_dicにあるものだけにフィルタしている。
        # この辞書はcookpad_nii_ingred_count豚20221205.csvの単語の頻度リストから作成している。このファイルは、語彙の揺れをユニーク化して、ユニーク化した単語に頻度を集計したもの
        # 一方、ingred_datasetは揺れを吸収していないので、このフィルタリングで、大きな値をもつ単語がフィルタされている。そのため、フィルタされた単語の頻度が、後続のデータセットに反映されないという問題がある
        # 影響があるのは、./dataset/ingred_row_words_count_{}　./dataset/dataset_LDA_{}　./dataset/ingred_cooccur_{}　./dataset/dataset_byCookstyle.pickle'
        # /dataset/cookpad_dataset_filter.pickle'
        # なお、もともとのcookpad_nii_ingred_count豚20221205.csvの単語頻度は、頻度閾値で辞書を作るためだけに使われており、後続のデータセットには継承してないので影響はない。
        # レシピタイトル毎の素材リストにもとづき、w2v similarを計算する際に、揺れのある表記がw2vの学習対象に入ってないので、かなり絞られる。
        ingred_dataset_filter.append(data) # もとのcookpadデータの行数を圧縮しないよう対応している
        update_ingred_count(data,ingred_count_dic)
        title_summery = str(title) + str(summery)
        tsukurepo_filter.append(t)
        titles_filter.append(title)
        summeries_filer.append(summery)
        
        indices = [i for i,c in enumerate(key_word) if c in ingred_data]#  title_summery
        
        if len(indices)!=0:
            kw_count_vec[indices[0]]+=1
            create_dataset_LDA(dataset_LDA,data,key_word[indices[0]])
            update_bag_of_ingred(bag_of_ingred,data,key_word[indices[0]])
            create_dataset_filter(title,summery,data,key_word[indices[0]])
        kw_count_vec[0]+=1 
        create_dataset_LDA(dataset_LDA,data,key_word[0])
        update_bag_of_ingred(bag_of_ingred,data,key_word[0])
        create_dataset_filter(title,summery,data,key_word[0])

dataset_filter_dic={}
for k,d in dataset_filter.items():            
    dataset_filter_df = pd.DataFrame(d,columns=['title','summery','ingred_list'])
    dataset_filter_dic[k]=dataset_filter_df
with open('./dataset/dataset_byCookstyle.pickle', 'wb') as f:
    pickle.dump(dataset_filter_dic, f)  
   
    
col = [k for k,v in cookpad_nii_dic.items()]
ing_count_vec = [v for k,v in ingred_count_dic.items()]
ingred_count_df = pd.DataFrame([ ing_count_vec], columns=col, index = ['count'])       
ingred_count_df = ingred_count_df.transpose()
kw_count_df = pd.DataFrame([kw_count_vec], columns=key_word, index=['count'])
kw_count_df =  kw_count_df.transpose()

cookpad_dataset_filter={'ingred_dataset_filter':ingred_dataset_filter,'titles_filter':titles_filter,'summeries_filer':summeries_filer,'tsukurepo_filter':tsukurepo_filter}
with open('./dataset/cookpad_dataset_filter.pickle', 'wb') as f:
    pickle.dump(cookpad_dataset_filter, f)


def create_bow(dataset):
    
    bow=[]
    for data in dataset:
        word_vec = np.zeros(len(cookpad_nii_dic))
        for word in data:        
            w_id = cookpad_nii_dic[word]
            word_vec[w_id] +=1
        bow.append(word_vec)
    return bow


bow_LDA={}       
header = [k for k,v in cookpad_nii_dic.items()]
for kw, dataset in dataset_LDA.items():
    
    bow = create_bow(dataset)
    bow_df = pd.DataFrame(bow, columns = header)
    bow_LDA[kw] = bow_df
    


ingred_freq=[]
index = []
for k, bag_of_ing in bag_of_ingred.items() :

    ing_vec = np.zeros(len(cookpad_nii_dic))
    for ing_name, freq in bag_of_ing.items():
        sequence = cookpad_nii_dic[ing_name]
        ing_vec[sequence] = freq
    ingred_freq.append(ing_vec)
    index.append(k)
    
    #bag_of_ing_sort = dict(sorted(bag_of_ing.items(), key=lambda x:x[1],reverse=True))
    #bag_of_ing_top20={k:v for i,(k,v) in enumerate(bag_of_ing_sort.items()) if i <20} 
    #print(k,bag_of_ing_top20 ) 

ingred_columns = [n for n,v in cookpad_nii_dic.items()]
bag_of_ingred_df = pd.DataFrame(ingred_freq, columns =ingred_columns, index =index)
bag_of_ingred_df =  bag_of_ingred_df.transpose()

assert len(ingred_dataset_filter) == len(titles_filter) and len(titles_filter) == len(summeries_filer) and len(ingred_dataset_filter) == len(tsukurepo_filter), 'unmatch len dataset'



with codecs.open("./dataset/ingred_cooccur_{}.csv".format(file_name), "w", "ms932", "ignore") as cookpad_file: 
    #header=Trueで、見出しを書き出す
    bag_of_ingred_df.to_csv(cookpad_file, index=True, encoding="ms932", mode='w', header=True)
   

with open('./dataset/ingred_row_words_count_{}.pickle'.format(file_name), 'wb') as f:
    pickle.dump(ingred_dataset_filter, f)
f_name = './dataset/ingred_columns_words_count_{}.csv'.format(file_name)
with codecs.open(f_name, "w", "ms932", "ignore") as f: 
    kw_count_df.to_csv(f, index=True, encoding="ms932", mode='w', header=True)  
f_name = './dataset/ingred_row_words_count_{}.csv'.format(file_name)
with codecs.open(f_name, "w", "ms932", "ignore") as f: 
    ingred_count_df.to_csv(f, index=True, encoding="ms932", mode='w', header=True) 

   
   
with open('./dataset/dataset_LDA_{}.pickle'.format(file_name), 'wb') as f:
    pickle.dump(dataset_LDA, f)
with open('./dataset/bow_LDA_{}.pickle'.format(file_name), 'wb') as f:
    pickle.dump(bow_LDA, f)
