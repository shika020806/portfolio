import re
import json
import codecs
import json
import pickle
import os
import pymongo
import pandas as pd
import numpy as np

with open(r'material_string_list.pickle', 'rb') as read:
  material_string = pickle.load(read)
cookpad_df = pd.read_csv(r'C:/Users/211K10152/OneDrive - 学校法人立正大学学園/ドキュメント/クックパッド研究/クックパッド関連/第一回課題/utf-8_wafu-pasta20230116.csv')
#json_open = open('./recipe1m/layer1.json', 'r')
#json_load = json.load(json_open)
dish_list=[]
while True:
    
    dish_name = input('input dish name :')
    if dish_name!='q':
        dish_list.append(dish_name)
    else:
        break
        
        
#client = pymongo.MongoClient(host='localhost', port=27017)
#cookpad_db = client['cookpad_nii_database']
#flavor_collection = cookpad_db['recipe_collection']


#key_word = ['bake','roast','grill','steam','boil','marinate','stew']
dummy_list = ["dummy"]
recipe_new = []
counter = 0
n = 0
#for i,recipe in enumerate(json_load): 
for i,recipe in  cookpad_df.iterrows():  #flavor_collection.find(no_cursor_timeout=True)
    recipe_row ={}
    #'title','title_id','summery','tsukurepo_no','ingredients'
    ingreds = [k for k in material_string[i] if k != 'none'  ]#recipe['ingredient'].items() 
    ingreds_str = ','.join(ingreds)
    #print(ingreds_str)
    title = recipe["タイトル"] #recipe['title']
    title_ingreds = ingreds_str+','+title
    match_list =[d for d in dish_list if d in title_ingreds]
    
    
    if dish_list[0]=='all':
        recipe_row['tsukurepo_no'] = 0 #dummy_list[0] つけたし
        recipe_row['summery'] = recipe["概要"] #recipe['summery']　つけたし
        recipe_row['title'] = recipe["タイトル"] #recipe['title']        
        recipe_row['ingredients'] = material_string[n] #もとはingred
        inst = "dummy" #[v for v in dummy_list] #recipe['process'].items() if k !='none'
        recipe_row['instructions'] = inst
        recipe_new.append(recipe_row)
        counter+=1
        
    elif len(match_list)!=0:
        print(counter)
        recipe_row['title'] = recipe["タイトル"] #recipe['title']
        recipe_row['summery'] = recipe["概要"] #recipe['summery']
        recipe_row['tsukurepo_no'] = 0 #dummy_list[0]
 #recipe['tsukurepo_no']
        #print(recipe['tsukurepo_no'])
        #recipe_row['url'] = recipe['url']
        recipe_row['ingredients'] = material_string[n] #もとはingred
        #inst = [i['text'] for i in recipe['instructions']  ]all
        #recipe_row['instructions'] = inst
        recipe_new.append(recipe_row)
        counter +=1
    #if counter > 10000:
    #    break
    n += 1
        
with open('./dataset/{}.json'.format(dish_list[0]), 'w') as f:
    json.dump(recipe_new, f)
with open('./dataset/count_{}.pickle'.format(dish_list[0]), 'wb') as f:
    pickle.dump(counter, f)