# export PYTHONPATH=/notebooks/pip_install/
"""
preliminaries:
# cp -p /datasets/vg_data/image_data.json to /inputs/vg-data-checkpoint
install packages;
nebula3-vg-driver 
nebula3-experiments
X-VLM issue with GDRIVE : need permission
gdown.download( "https://drive.google.com/u/0/uc?id=1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN", "retrieval_mscoco_checkpoint_9.pth")
"""
import numpy as np
import nltk
import tqdm 
import torch 
from PIL import Image
import sys
import os
import pickle
import pandas as pd
import requests
import torch
from time import time
# nltk.download('punkt')
# from nltk import tokenize
# 16-A100(40G) machine, our largest model with ViT-G and FlanT5-XXL 

os.environ["TRANSFORMERS_CACHE"] = "/storage/hft_cache"
os.environ["TORCH_HOME"] = "/storage/torch_cache"
os.environ["CONFIG_NAME"] = "giltest"  # implicit overwrite the default by  : "vg_data_basedir": "/storage/vg_data"



# sys.path.insert(0,"/notebooks/nebula3_experiments")
from nebula3_experiments.ipc_utils import *
from nebula3_experiments.vg_eval import Sg_handler, spice_get_triplets, get_sc_graph, tuples_from_sg, VGEvaluation
sys.path.insert(0,"/notebooks")
sys.path.insert(0,"/notebooks/visual_clues/")

from visual_clues.ontology_implementation import SingleOntologyImplementation
from visual_clues.ontology_factory import *
from visual_clues.vlm_factory import VlmFactory
from visual_clues.vlm_interface import VlmInterface

sys.path.insert(0,"/notebooks/local_vision_processors/")
sys.path.insert(0,"/notebooks/local_vision_processors/configs/")
sys.path.insert(0,"/notebooks/local_vision_processors/base_models/xvlm/")
from local_vision_processors.vision_models import XVLMModel# pip install tritonclient\[all\] --target /notebooks/pip_install/
# from visual_clues.vlm_implementation import VlmChunker
# Since ive removed visual_clues package for develop than need to : pip install -U transformers
#TODO: @@HK X-VLM make sure what is the best prompt way of asking VLM
# Reolsve issue woth path of model     'x-vlm' : '/notebooks/nebula3_playground/x-vlm/checkpoint_9.pth',
# From viper: xvlm.py  :     configs/base_config.yaml thresh_xvlm: 0.6
# https://github.com/cvlab-columbia/viper/blob/9700f4a1104f98373424aea1c944ef19505b58fa/vision_models.py#L1189

result_path = "/notebooks/nebula3_playground"
use_remote = False
from os import getenv
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

class VideoProcessingConf:
    def __init__(self) -> None:

        self.MOVIES_PATH = getenv('MOVIES_PATH', '/datasets/media/movies')
        self.LOCAL_MOVIES_PATH = getenv('LOCAL_MOVIES_PATH', '/tmp')
        self.FRAMES_PATH = getenv('FRAMES_PATH', '/datasets/media/frames')
        self.LOCAL_FRAMES_PATH = getenv('LOCAL_FRAMES_PATH', '/tmp/frames')
        self.LOCAL_FRAMES_PATH_RESULTS_TO_UPLOAD = getenv('LOCAL_FRAMES_PATH', '/tmp/frames/reid')
        self.WEB_PREFIX = getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
        self.WEB_HOST = getenv('WEB_HOST', '74.82.29.209')
        self.WEB_USERPASS = getenv('WEB_USERPASS', 'paperspace:Nebula@12345')
    def get_movies_path(self):
        return (self.MOVIES_PATH)
    def get_local_movies_path(self):
        return (self.LOCAL_MOVIES_PATH)
    def get_frames_path(self):
        return (self.FRAMES_PATH)
    def get_local_frames_path(self):
        return (self.LOCAL_FRAMES_PATH)
    def get_web_prefix(self):
        return (self.WEB_PREFIX)
    def get_web_host(self):
        return (self.WEB_HOST)
    def get_web_userpass(self):
        return (self.WEB_USERPASS)


class RemoteStorage():
    def __init__(self):
        # super().__init__()
        self.vp_config = VideoProcessingConf()
        # make dirs
        if not os.path.isdir(self.vp_config.get_local_frames_path()):
            os.mkdir(self.vp_config.get_local_frames_path())
        # init ssh
        userpass = self.vp_config.get_web_userpass().split(":")
        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())
        # self.ssh.load_system_host_keys()
        self.ssh.connect(self.vp_config.get_web_host(),
                         username=userpass[0], password=userpass[1])
        self.scp = SCPClient(self.ssh.get_transport())
        # after init all

    def upload_files_to_web(self, uploads):
        try:
            for local_path, remote_path in uploads.items():
                self.scp.put(local_path, recursive=True, remote_path=remote_path) # if foler doent exist due to random key prefix for re_id files hence it will be written as is w/o the /re_id path
        except Exception as exp:
            print(f'An exception occurred: {exp}')
            return False
        return True

def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh))) 

evaluator = VGEvaluation()
remote_storage = RemoteStorage()


min_h=60
min_w=60
top_k = 20
top_k_recall_calc = 10

prefix_prompt = "A photo of a"
use_prefix_str = ''

predefined_ipc = False
predefined_ipc_str = ''
if predefined_ipc:
    image_ids_related_to_ipc = [2395479, 2323701,  2410301, 2340766]#[2328018, 2336194, 2337655, 2320416]#2336194: 2*elephabnt 2337655 lady behind surfboard #[2324582, 2331094, 2323701, 2340749, 2344612]
    predefined_ipc_str = '_predefined_ipc'

# vlm_type = 'xvlm' #'blip_itc', 'clip'

for vlm_type in ['blip2']: #['blip2', 'blip_itc', 'xvlm', 'clip']:
    if vlm_type == 'xvlm':
        ontology_factory = OntologyFactory()
        xvlm_model = XVLMModel()
    elif vlm_type == 'blip2':
        ontology_factory = OntologyFactory()
        sys.path.insert(0,"/notebooks/nebula3_experiments/")
        from notebooks.nebula3_experiments.blip2 import *
        blip2 = BLIP2(model="blip2") #file, caption, file_or_url='file'

    for ontology in ['ovad_attributes', 'vg_attributes']:
    # ontology = 'ovad_attributes'#'vg_attributes' # ovad_attributes
        if  vlm_type == 'clip' or vlm_type == 'blip_itc':
            ontology_imp = SingleOntologyImplementation(ontology, vlm_type)
        elif vlm_type == 'xvlm' or vlm_type == 'blip2':
            ontology_list = ontology_factory.get_ontology(ontology)

        csv_file_name = 'results_ipc_attribute_vlm_' + str(vlm_type) + '_ontology_' + str(ontology) + '_topk_' + str(top_k_recall_calc) + '.csv'


        results = list()
        for inx, ipc_id in enumerate(tqdm.tqdm(image_ids_related_to_ipc)):
            ipc = get_visual_genome_record_by_ipc_id(ipc_id)
            # vg_ind_related_to_ipc = [ix for ix , x in enumerate(ipc_data) if x['image_id']==ipc_id][0]
            # ipc = ipc_data[vg_ind_related_to_ipc]
            image_id = os.path.basename(ipc['url'])
            image_path = os.path.join('/datasets/visualgenome/VG', image_id)
            image = Image.open(image_path).convert('RGB')   

            sg = get_sc_graph(ipc['image_id'])
            for i, (visual_objects, attrib) in enumerate(zip(sg.objects, sg.attributes)): #sg.objects[0].height
                h = visual_objects.height
                w = visual_objects.width
                y = visual_objects.y
                x = visual_objects.x
                xmin, ymin, xmax, ymax = bbox_xywh_to_xyxy((x,y,w,h))
                
                crop_image = image.crop((xmin, ymin, xmax, ymax))
                if (h * w) > (min_w * min_h) and any(attrib.attribute):    # skip mirror is [] by any(attrib.attribute)
                # for ib, rel in enumerate(sg.attributes):
                    start = time()
                    if vlm_type == 'clip' or vlm_type == 'blip_itc':
                        vlm_sim = ontology_imp.compute_scores(crop_image)
                    elif vlm_type == 'blip2':

                        filename = 'crop2.jpg'
                        if use_remote:
                            local_folder = 'tmp'
                            re_id_result_path = os.path.join('/notebooks/nebula3_playground', local_folder)
                            crop_image.save(os.path.join(re_id_result_path, filename))
                            # re_id_mdfs_web_dir = 'paperspace@74.82.29.209:/datasets/'
                            re_id_mdfs_web_dir = '/datasets/media/services'
                            uploads = {re_id_result_path: re_id_mdfs_web_dir}
                            web_dir = remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services'
                            remote_storage.upload_files_to_web(uploads)
                        else:
                            local_folder = 'tmp'
                            re_id_result_path = os.path.join('/notebooks/nebula3_playground', local_folder)
                            crop_image.save(os.path.join(re_id_result_path, filename))

                        vlm_sim = list()
                        for attrib_ont in ontology_list:
                            itc_score = blip2.process_image_and_captions(file=os.path.join(re_id_result_path, filename), 
                                            caption=attrib_ont, match_head = "itc")
                            itc_gt = itc_score.detach().cpu().numpy()[0].item()
                            vlm_sim.append(itc_gt)
                        vlm_sim = [(ont,float(sim)) for ont,sim in zip(ontology_list, vlm_sim)]

                        
                    elif vlm_type == 'xvlm':
                        vlm_sim = xvlm_model.forward(np.array(crop_image), ontology_list)
                        vlm_sim = [(ont,float(sim)) for ont,sim in zip(ontology_list, vlm_sim[0])]
                    else:
                        raise


                    roi_per_image_vlm_score_attrib = torch.from_numpy(np.array(vlm_sim)[:,1].astype('float32'))
                    v_top_k_attrib, i_topk_attrib = torch.topk(roi_per_image_vlm_score_attrib, k=top_k, dim=0) # worst cast if only one ROI/BB than it has to prsent the same top-k
                    pred_vlm_attrib = [np.array(np.array(vlm_sim)[:,0])[j] for j in np.array(i_topk_attrib)]

                    print('VLM {} time: {:6f} seconds'.format(vlm_type, time() - start))

                    obj_dict = {'image_id': image_id, 'vlm':vlm_type, 'ontology':ontology, 'bbox': (x,y,w,h), 'gt_obj': visual_objects, 
                    'gt_attrib' : attrib.attribute}
                    [obj_dict.update({ix: [batch_ontology, float(vlm_sim.__format__('.3f'))]}) for ix, (batch_ontology, vlm_sim) in enumerate(zip(pred_vlm_attrib, v_top_k_attrib.numpy()))]
                    # if len(attrib.attribute) >1:
                    #     print('ka')
                    # recall_triplets

                    top_out_of_bb_list = pred_vlm_attrib[:top_k_recall_calc]
                    top_out_of_bb_list = [tuple([x]) for x in top_out_of_bb_list]
                    gt = [tuple([x]) for x in attrib.attribute]
                    recall = evaluator.recall_triplets_mean(src=gt, dst=top_out_of_bb_list)

                    obj_dict.update({'recall': recall})
                    
                    results.append(obj_dict)
            # Intermediate save 
            if (inx % 10) == 0:
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(result_path, csv_file_name), index=False)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_path, csv_file_name), index=False)
        print(np.mean(df.recall.to_numpy()))
    # self.vlm = VlmChunker(VlmFactory().get_vlm(self.settings["vlm_fusion"]), chunk_size=50)
    # vlm1 = VlmFactory().get_vlm("clip")
    # print(vlm1)
    # [obj_dict.update({batch_ontology: float(vlm_sim)}) for batch_ontology, vlm_sim in zip(np.array(vlm_sim)[:,0], np.array(vlm_sim)[:,1])]


# const.py, ontology


"""
    ontology_implementation = SingleOntologyImplementation('objects', 'clip')

    image = ontology_implementation.vlm.load_image_url("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg")
    ontology_implementation.compute_scores(image)

start with this:
https://github.com/NEBULA3PR0JECT/visual_clues/blob/main/visual_clues/ontology_factory.py
https://github.com/NEBULA3PR0JECT/visual_clues/tree/main/visual_clues/visual_token_ontology/vg
https://github.com/NEBULA3PR0JECT/visual_clues/blob/main/visual_clues/ontology_implementation.py
"""
