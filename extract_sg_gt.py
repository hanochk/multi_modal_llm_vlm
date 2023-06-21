import os
from database.arangodb import NEBULA_DB
from typing import NamedTuple
import pandas as pd
import numpy as np
import json 
import tqdm
from nebula3_experiments.ipc_utils import *
from nebula3_experiments.vg_eval import Sg_handler, spice_get_triplets, get_sc_graph, tuples_from_sg

os.environ["TRANSFORMERS_CACHE"] = "/storage/hft_cache"
os.environ["TORCH_HOME"] = "/storage/torch_cache"
os.environ["CONFIG_NAME"] = "giltest"

os.environ["ARANGO_DB"]="ipc_200"
nebula_db = NEBULA_DB()


GRAPH_COLLECTION = "Graphs_Jun_10_1"
doc = 'Graphs_Jun_10_1'

vgenome_metadata = "/storage/vg_data/"

with open(os.path.join('/storage/ipc_data/paragraphs_v1.json'), "r") as f:
    images_data = json.load(f)
sample_ids = np.loadtxt(os.path.join(vgenome_metadata, "sample_ids_ipc_vgenome_ids.txt"))

ipc_data = json.load(open('/storage/ipc_data/paragraphs_v1.json','r'))
image_ids_related_to_ipc = [images_data[int(ix)]['image_id'] for ix in sample_ids]
# vg_ind_related_to_ipc = [ix for ix , x in enumerate(ipc_data) if x['image_id']==image_id][0]

result_path = '/notebooks/multi_modal/sg_results'
collection = nebula_db.get_doc_by_key2({},GRAPH_COLLECTION)
for doc in tqdm.tqdm(collection):

    # print(doc)
    image_id = doc['url'].split('/')[-1].split('.jpg')[0]
    
    ipc = get_visual_genome_record_by_ipc_id(int(image_id))
    sg = get_sc_graph(ipc['image_id'])
    gt_obj_attr = dict()
    for i, (visual_objects, attrib) in enumerate(zip(sg.objects, sg.attributes)):
        # print(visual_objects.names, attrib)
        gt_obj_attr.update({i: [visual_objects.names, attrib]})

    for rel in sg.relationships:
        i += 1
        gt_obj_attr.update({i: rel})

    df_gt = pd.DataFrame(gt_obj_attr)
    df_gt = df_gt.transpose()
    df_gt.to_csv(os.path.join(result_path, str(image_id) + '_gt.csv'), index=False)

    obj_attr = [x['label'] for x in doc['objects']]

    if 0:
        objects = dict()
        objects.update({"caption": doc['caption']})
        objects.update({"objects": doc['objects']})
        objects.update({"relations": doc['relations']})
        

    else:
        objects = dict()
        objects.update({"caption": doc['caption']})
        inx = 0
        for item_i in doc['objects']:
            if item_i['type'] == 'object' or 1 :
                inx += 1
                objects.update({inx: item_i})
        
        # # attrib = dict()
        # for item_i in doc['objects']:
        #     if item_i['type'] == 'has_attribute':
        #         inx += 1
        #         # item_i['type']
        #         objects.update({inx: item_i['label']})
        # attrib = dict()
        for item_i in doc['relations']:
            inx += 1
            # item_i['type']
            objects.update({inx: item_i})

        # rel = [x['label'] for x in doc['relations']]
    
    csv_file_name = str(image_id) + '_ipc200_sg_ft_vs_dima_sg'  + '.csv'
    df = pd.DataFrame(objects)
    # df = pd.DataFrame.from_dict(list(objects.items()))
    df = df.transpose()
    df.to_csv(os.path.join(result_path, csv_file_name), index=False)

    
print(0)