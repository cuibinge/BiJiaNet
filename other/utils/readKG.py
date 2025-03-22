# -*- codeing = utf-8 -*-
# @Time: 2023/7/25 18:30
# @Author:SunHuan

import pandas as pd


class_name = ['Seagrass bed',
              'Spartina alterniflora',
              'Reed',
              'Tamarix',
              'Tidal flat',
              'Sparse vegetation',
              'Sea',
              'Yellow River',
              'Pond',
              'Cloud']

ch2en = { '海草床':'Seagrass bed',
          '互花米草':'Spartina alterniflora',
          '芦苇':'Reed',
          '柽柳混生区':'Tamarix',
          '裸潮滩':'Tidal flat',
          '潮滩稀疏植被':'Sparse vegetation',
          '海水':'Sea',
          '黄河':'Yellow River',
          '坑塘':'Pond',
          '云':'Cloud'}

relations = ['attribute']

entities_dict = {name:i for i,name in enumerate(class_name)}
relation_dict = {rel:i for i,rel in enumerate(relations)}



def getAttribTriples():

    #kg = pd.read_csv(r'./YanchengModel/attr_yancheng.csv',encoding='gbk')
    kg = pd.read_csv(r'F:/NET/对比方法/huan/MyModel/attr.csv',encoding='gbk')

    knowledge_graph = []
    entity_names =[]
    relation_names = []

    #严格按照对应顺序
    for i,row in kg.iterrows():

        head = row.iloc[0]
        rel = row.iloc[1]
        tail = row.iloc[2]

        if head not in entity_names:
            entity_names.append(head)
        if rel not in relation_names:
            relation_names.append(rel)
        if tail not in entity_names:
            entity_names.append(tail)

        knowledge_graph.append((head,rel,tail))

        # print(i,row.iloc[0],row.iloc[1],row.iloc[2])

    # num_entities = len(entity_names)
    # num_rels = len(relation_names)
    # print(num_rels,num_entities)

    entity2id = {name: i for i, name in enumerate(entity_names)}
    relation2id = {name: i for i, name in enumerate(relation_names)}

    knowledge_graph = dataset(knowledge_graph,entity2id,relation2id)

    return knowledge_graph


def dataset(knowledge_graph,entity2id,relation2id):
    kgs = []
    for sample in knowledge_graph:
        kgs.append([entity2id[sample[0]],relation2id[sample[1]],entity2id[sample[2]]])
    return kgs

# kg = getAttribTriples()
# print(kg)