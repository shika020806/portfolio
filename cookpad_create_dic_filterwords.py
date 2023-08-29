import json
import pandas as pd
import codecs
import pickle
import numpy as np
import json
import ingred_tokenizer

file_name=input('dish name?')
json_open = open('./dataset/{}.json'.format(file_name), 'r')
json_load = json.load(json_open)
unique_dict = pd.read_csv('./dataset/unique_dict.csv',  encoding='ms932', sep=',',skiprows=0)
unique_dict = {u['var']:u['uniq'] for k,u in unique_dict.iterrows()}
#cook_style = ['煮込み','シチュー','ロースト','ソテー','グリル','角煮','焼き']


def create_dict(recipe_ingred):# 新しい素材名についてingred_name_dicに　素材名:id で登録。また共起頻度bag_of_ingredに　新しい共起素材名：０　を追加
        
       
    ingred_list=[ingred_tokenizer.tokenize(ing) for ing in recipe_ingred ]
    ingred_list=[ing for ing in ingred_list if ing !='']# and dish_name not in ing] 
       
    if len(ingred_list)!=0: # スパイス名が出現しないレシピは読み飛ばす
        
        for ingred_name in ingred_list:
            
            if ingred_name not in count_ingreds: # 素材固有表現辞書は、kw(料理空間もしくは代替スパイス）によらず共通{素材名:id}                     
                                    
                count_ingreds[ingred_name] = 1                
            else:
                count_ingreds[ingred_name] +=1   
    else:
        ingred_list=[]
    
    return ingred_list
    
count_ingreds={}
ingred_dataset=[]
titles = []
summeries = []
tsukurepos = []
############################
filter_ingred =["トマト"]#使用にしたい単語を入れる
filter_ingred2 =["パスタ"]
############################
for i,recipe in enumerate(json_load): 
    #print(recipe)
    if i % 100 ==0:
        print(i)
    
    ingredient_list = create_dict(recipe['ingredients'])

    ingredient_list_after = []

    for j in ingredient_list:#揺れの吸収の処理が行われていないので処理の追加
        if j not in unique_dict:    
            ingredient_list_after.append(j)
        else:
            ingredient_list_after.append(unique_dict[j])

    filter_count = [i for i in filter_ingred if i in ingredient_list_after]

    if len(filter_count) == 0:
        continue

    filter_count2 = [i for i in filter_ingred2 if i in ingredient_list_after]

    if len(filter_count2) == 0:
        continue 

    ingred_dataset.append(ingredient_list_after)
    titles.append(recipe['title'])
    summeries.append(recipe['summery'])
    tsukurepos.append(recipe['tsukurepo_no'])
    #if i > 1000:
    #    break
    
count_ingreds_array = [[ing, count] for ing, count in count_ingreds.items()]

count_ingreds_df = pd.DataFrame(count_ingreds_array, columns = ['ingredients','count'])

f_name = './dataset/cookpad_nii_ingred_count{}.csv'.format(file_name)
with codecs.open(f_name, "w", "ms932", "ignore") as f: 
    count_ingreds_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)

dataset = {'ingred_dataset':ingred_dataset, 'titles':titles,'summeries':summeries,'tsukurepos':tsukurepos}
with open('./dataset/dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

# df_data = pd.DataFrame(ingred_dataset)
# with codecs.open(f_name, "w", "ms932", "ignore") as f: 
#     df_data.to_csv(f, index=True, encoding="ms932", mode='w', header=True)  



