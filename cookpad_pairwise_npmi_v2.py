import re
import json
import pandas as pd
import codecs
import pickle
import numpy as np
import japanize_matplotlib
import json
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graphviz import Graph
import pydotplus as pdp
from PIL import Image
from graphviz import Digraph
import networkx as nx
import collections

#https://magazine.techacademy.jp/magazine/20695
#https://qiita.com/miyase256/items/c3954bb51aa23a3ad13a
#https://programgenjin.hatenablog.com/entry/2019/02/26/075121
#https://stackoverflow.com/questions/23018684/how-to-add-an-annotation-outside-of-a-node-in-graphviz-dot
#https://simply-k.hatenadiary.org/entry/20100727/1280224098
#https://qiita.com/tomati2021/items/426ae2cc89099bf7ecc3
#http://int-info.com/PyLearn/PyLearnTIPS01.html
file_name = input('file name ?')
key_word = input('key word ?')#主成分分析上で主だった素材を入れる

with open(f'./dataset/bow_LDA_{file_name}.pickle', 'rb') as f4:
    voc_vecs = pickle.load(f4)
ingred_row_df = pd.read_csv(f'./dataset/ingred_row_words_count_{file_name}.csv',  encoding='ms932',index_col=0, sep=',',skiprows=0) 
npmi_byStyle = pd.read_csv(f'./dataset/ingred_cooccur_npmi_{file_name}.csv',  encoding='cp932', sep=',',skiprows=0)
cookstyle_count = pd.read_csv(f'./dataset/ingred_columns_words_count_{file_name}.csv', index_col=0, encoding='cp932', sep=',',skiprows=0)

# flavor_topic = pd.read_csv('./dataset/document_vectors202212.csv',  encoding='ms932', sep=',',skiprows=0) 
# ingred_topic = {ing:topic for ing,topic in zip(flavor_topic['ingredient'],flavor_topic['max_topic'])}



header=npmi_byStyle.columns.tolist()
ingred_row_words_prob={}
no_of_total_recipes = 17247#149342# #cookstyle_count.loc[cooking_style,:].values[0] #1715595 #
ingred_prob={}
for ingred_name, prob in ingred_row_df.iterrows():
    ingred_prob[ingred_name]=prob.values[0]/no_of_total_recipes
    
    
    

idx = header.index(key_word)
npmi_byStyle = npmi_byStyle.iloc[:,idx:idx+2]
ingred_names= [ing for ing,npmi in zip(npmi_byStyle.iloc[:,0],npmi_byStyle.iloc[:,1]) if npmi >= 0.46]#if npmi >= 閾値　でラインを決められる
if len(ingred_names) <= 5:
    print("---"*10)
    print("npmiの閾値が高すぎるかもしれません\nline54の閾値を変えてください")
    print("---"*10)
else:
    print("適切です")


def npmi_compute(x,dataset):
    cooccour =0
    if x[0] in ingred_prob and x[1] in ingred_prob:
        for x1, x2 in zip(dataset[x[0]],dataset[x[1]]):
        
            if x1 >0 and x2 >0:
                cooccour+=1
        #print(x[0],x[1])    
        joint_prob = cooccour/no_of_total_recipes
        if -np.log(joint_prob)==0 or ingred_prob[x[0]]*ingred_prob[x[1]]==0 or joint_prob==0:
            npmi = 0
        else :
            print(f"x0:{ingred_prob[x[0]]}  x1:{ingred_prob[x[1]]}")
            print(f"{x[0]}と{x[1]}の共起回数は{cooccour}です")
            npmi = np.log(joint_prob/(ingred_prob[x[0]]*ingred_prob[x[1]]))/(-np.log(joint_prob))
    else:
        npmi = 0
    return npmi


dataset = voc_vecs[key_word]
#ingred_names = dataset.columns.tolist()
#ingred_names = [i for i in ingred_names if i in ingred_npmiDic]
npmi_vecs = {ing:np.ones(len(ingred_names)) for ing in ingred_names}
index=[]
for k,x in enumerate(itertools.permutations(ingred_names, 2)):
    print(x,str(k)+'/'+str(len(ingred_names)**2))
    
    npmi_val = npmi_compute(x,dataset)
    print(npmi_val)
    vec = npmi_vecs[x[0]]
    idx = ingred_names.index(x[1])
    vec[idx] = npmi_val
    npmi_vecs[x[0]] = vec

index = []
npmi_array =[]
for ingred, npmi_vec in npmi_vecs.items():

    index.append(ingred)
    npmi_array.append(npmi_vec)
    
    
pairwise_npmi = pd.DataFrame(npmi_array, index = index, columns = ingred_names)    
with codecs.open('./dataset/pairwise_npmi.csv', "w", "ms932", "ignore") as f: 
    pairwise_npmi.to_csv(f, index=True, encoding="ms932", mode='w', header=True)
with open('./dataset/pairwise_npmi.pickle', 'wb') as f:
    pickle.dump(pairwise_npmi, f)

#sns.set(font=["IPAexGothic"], font_scale=10/10)
#fig, ax1 = plt.subplots(1, 1)
#sns.heatmap(pairwise_npmi, annot=True,square=True,cbar=True, ax=ax1)
#plt.show()

# graphbizのカラー　https://programgenjin.hatenablog.com/entry/2019/02/26/075121
#topic_sematic = {0:'スパイシー',1:'シトラス',2:'焦臭',3:'発酵熟成',4:'熟成',5:'ベリー',6:'硫黄・刺激臭',7:'ウッディ',8:'グリーン',9:'エキゾチック'}
colors=['chartreuse','bisque','aquamarine','gold','darkolivegreen1','coral','cadetblue','cyan','crimson','deeppink']
def draw_graph_gviz(cm, threshold):

    
    edge_indices = np.where(cm > threshold)
    
    edges = [[cm.index[i], cm.index[j]] for i, j in zip(edge_indices[0], edge_indices[1]) if i > j ]
    # print(edges)
    edge_labels = [cm.loc[i,j] for i, j in zip(edges[0], edges[1]) if i > j ]#if i > j]
    node_filter=[]
    for e in edges:
        node_filter+=e
    node_filter=list(set(node_filter))           
    
    #print(edges)
    g = Graph(format='png')    
    g.attr('node', shape='box', fontname='MS Gothic') # 
    
    
    for i,k in enumerate(range(cm.shape[0])):
        if cm.index[k] in node_filter:
            # topic = [ topic for ing,topic in ingred_topic.items() if ing in cm.index[k]]
            
            # if len(topic)!=0:
            #     color=colors[topic[0]]
            # else:
            #     color = 'grey'
            g.node(cm.index[k],style='filled', fillcolor="grey")
            
            #g.node(cm.index[k])
        #print(cm.index[k])

    for e,(i, j) in enumerate(edges):
        g.edge(j, i, label=str(round(cm.loc[i,j],2)))
    
    
    g.attr('node', shape='circle',fontname='MS Gothic') #    
    # for topic,label in topic_sematic.items():
    #     color = colors[topic]        
    #     g.node(label,style='filled', fillcolor=color)
        
    return g


def create_graph_nx(cm, threshold):

    edge_indices = np.where(cm > threshold)#dataframeに対して、np.whereで条件を与えると、それを満たす要素の行番号配列、列番号配列の２つの1次元配列が取り出される
    edges = [[cm.index[i], cm.index[j]] for i, j in zip(edge_indices[0], edge_indices[1]) if i > j]
    G = nx.Graph() 
    '''
    edges=[]
    for i, j in zip(edge_indices[0],edge_indices[1]):
        if i > j:
           G.add_edge(cm.index[i],cm.index[j], weight=round(cm.iloc[i,j],3)) 
           edges.append([cm.index[i], cm.index[j]])
    '''
    edge_array=[]
    for e in edges:
        edge_array+=e
    node_filter=list(set(edge_array))
    node_degree = dict(collections.Counter(edge_array))
    node_degree = dict(sorted(node_degree.items(), key=lambda x:x[1],reverse=True))
    #print(node_filter)
    for k in range(cm.shape[0]):
        if cm.index[k] in node_filter:
            G.add_node(cm.index[k])       
    #edges_label = [(i,j,cm.loc[i,j]) for i, j in zip(edges[0], edges[1])]
    #print(edges_label)
    for i, j in edges:
        G.add_edge(i,j, weight=round(cm.loc[i,j],3)) 
    
    
    return G, node_degree
    
####################
threshold = 0.46 #閾値
####################

print(pairwise_npmi)
g=draw_graph_gviz(pairwise_npmi, threshold)
g.view()
g.render(filename='./dataset/pairwise_npmi_graph', format='png', cleanup=True, directory=None)
g.render(filename=f'./ネットワーク図/main_{file_name}_key_{key_word}_threshold_{threshold}_overview', format='png', cleanup=True, directory=None)
with open('./dataset/G_viz.pickle', 'wb') as f:
    pickle.dump(g, f)
    
    
create_graph_nx(pairwise_npmi, threshold)

'''
# ----------  まずthresholdは0以上固定として、ネットワークオブジェクトGを書き出す。これは、後続プログラムw2vでのsimilarityを計算する際にできるだけボキャブラリーを確保したいため
G,node_degree = create_graph_nx(pairwise_npmi, 0)
with open('./dataset/G.pickle', 'wb') as f:
    pickle.dump(G, f)
'''
# -----------  threshold以上のネットワークG2を書き出す。これはノードの次数を計算して、中心を検出するため     
G2, node_degree=create_graph_nx(pairwise_npmi, threshold)
with open('./dataset/G2.pickle', 'wb') as f:
    pickle.dump(G2, f)
with open('./dataset/node_degrees.pickle', 'wb') as f:
    pickle.dump(node_degree, f)
