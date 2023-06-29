# export PYTHONPATH=/notebooks/pip_install/
# pip install -U transformers

# export GEVENT_SUPPORT=True for debug in flask mode 
import os
from database.arangodb import NEBULA_DB, DBBase
from typing import NamedTuple
from nebula3_experiments.prompts_utils import *
import time
from nebula3_experiments.vg_eval import VGEvaluation  # in case error of "Failed to import transformers, cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_"  : then pip install requests==2.29.0
from typing import Union
from sklearn.cluster import KMeans
import tqdm
os.environ["TRANSFORMERS_CACHE"] = "/storage/hft_cache"
os.environ["TORCH_HOME"] = "/storage/torch_cache"
os.environ["CONFIG_NAME"] = "giltest"

os.environ["ARANGO_DB"]="ipc_200"
nebula_db = NEBULA_DB()

VISUAL_CLUES_COLLECTION = 's4_visual_clues'
REID_CLUES_COLLECTION = 's4_re_id'
MOVIES_COLLECTION = "Movies"
FUSION_COLLECTION = "s4_fusion"
LLM_OUTPUT_COLLECTION = "s4_llm_output"
KEY_COLLECTION = "llm_config"
FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
# FS_SAMPLES = 5                   # Samples for few-shot gpt

from abc import ABC, abstractmethod
import openai
from itertools import compress
from huggingface_hub.inference_api import InferenceApi
import requests

import importlib
print(importlib.metadata.version('openai'))

def flatten(lst): return [x for l in lst for x in l]

# Hosted Inference API :HF HTTP request
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Model Hub is where the members of the Hugging Face community can host all of their model checkpoints 
# hf = HuggingFaceHub(repo_id="google/flan-t5-xl")
# if 0:
#     hf_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", device_map="auto", torch_dtype=torch.bfloat16)
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
#     model = HuggingFaceLLM(hf_model, tokenizer)

api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_input_type_from_db(pipeline_id, collection):
    nre = DBBase()
    pipeline_data = nre.get_doc_by_key({'_key': pipeline_id}, collection)
    if pipeline_data:
        if "dataset" in pipeline_data["inputs"]["videoprocessing"]:
            input_type = pipeline_data["inputs"]["videoprocessing"]["dataset"]["type"]
        else:
            input_type = pipeline_data["inputs"]["videoprocessing"]["movies"][0]["type"]
    return input_type

class SummarizeScene():
    def __init__(self, prompting_type: str='few_shot', gpt_type: str='gpt-3.5-turbo-16k',semantic_token_deduplication: bool=True,
                    min_shots_for_semantic_similar_dedup: int=40, write_res_to_db: bool=True, caption_callback=None, 
                    verbose: bool=False):
        
        self.write_res_to_db = write_res_to_db
        self.gpt_type = gpt_type
        self.prompting_type = prompting_type
        self.semantic_token_deduplication = semantic_token_deduplication
        self.verbose = verbose
        self.caption_callback = caption_callback
        self.one_shot_context_ex_prefix_summary = '''Video Summary: 3 persons, in a room, Susan, Michael, & Tom. They look strange, Tom with a giant head, michael with a mask, one of them is giant. The three people appear very tense as they look around frantically. '''
        self.one_shot_context_ex_prefix_caption = '''Caption 1: Susan standing in front of a tv screen with a camera. 
                            Caption 2: Michael with a body in the middle of the screen. 
                            Caption 3: Tom standing next to a giant human in the shape of a human. 
                            Caption 4: Susan standing in front of a camera and looking at the camera. 
                            Caption 5: Susan in a blue dress and a man in a suit. 
                            Caption 6: Susan in a room with a bike in the background. 
                            Caption 7: Tom sitting on a chair in the middle of a room. 
                            Caption 8: Michael with a mask on his face and Michael with a mask on his. 
                            Caption 9: Tom in a suit and tie with a giant head. 
                            Caption 10: Michael in a suit with the capt that says, i'm. 
                            Caption 11: Tom in a spider suit sitting on a train car. 
                            Caption 12: Tom in a blue shirt and Tom in a black shirt. 
                            Caption 13: Susan standing in front of a bookcase with a bookcase in the background. {}.'''.format(self.one_shot_context_ex_prefix_summary)

        self.one_shot_context_ex_prefix_caption = self.one_shot_context_ex_prefix_caption.replace('\n ',' ').replace("  ", "")

        self.one_shot_context_ex_prefix_then = '''Susan standing in front of a tv screen with a camera and then Michael with a body in the middle of the screen and then Tom standing next to a giant human in the shape of a human and then Susan standing in front of a camera and looking at the camera and then Susan in a blue dress and a man in a suit and then Susan in a room with a bike in the background and then Tom sitting on a chair in the middle of a room and then Michael with a mask on his face and Michael with a mask on his and then Tom in a suit and tie with a giant head and then Michael in a suit with the capt that says,'i'm ' and then Tom in a spider suit sitting on a train car and then Tom in a blue shirt and Tom in a black shirt and then Susan standing in front of a bookcase with a bookcase in the background. {}.''' \
        .format(self.one_shot_context_ex_prefix_summary)

        # self.places = 'indoor'
        self.top_k_per_mdf = 1
        cluster_based_place = True
        if self.verbose:
            print("promting_type", self.prompting_type)


        if self.gpt_type == 'HF_':
            InferenceApi(repo_id="gpt-j-6b-shakespeare", token=api_token)
        elif self.gpt_type == 'chat_gpt_3.5' or self.gpt_type == 'gpt-4' or self.gpt_type == 'gpt-3.5-turbo-16k':
            self.chatgpt = ChatGptLLM()
        # elif self.gpt_type == 'text-davinci-003':
        #     context_win = 4096
        if self.semantic_token_deduplication:
            self.evaluator = VGEvaluation()
        else:
            evaluator = None

        self.min_shots_for_semantic_similar_dedup = min_shots_for_semantic_similar_dedup
        self.collection_name = 's4_scene_summary'

        return

# Semantic de-dupllication
    def _places_semantic_dedup(self, mdf_no: list[int], movie_id: str):
        # Place voting
        all_scene = list()
        place_per_scene_elements = dict()
        rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION) # + scene_elements
        scene_elements = rc_movie_id['scene_elements']

        for ix, frame_num in enumerate(mdf_no):
            # TODO per SE clustering 
            mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
            obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
            scene = [obj['global_scenes']['blip'][x][0] for x in range(len(obj['global_scenes']['blip']))][:self.top_k_per_mdf]
            # print([obj['global_scenes']['blip'][x] for x in range(len(obj['global_scenes']['blip']))][:1])
            all_scene.append(scene)
            scene_boundary = [x for x in scene_elements if (frame_num >= x[0] and frame_num < x[1])][0]
            if str(scene_boundary) in place_per_scene_elements:
                place_per_scene_elements[str(scene_boundary)].extend(scene)
            else:
                place_per_scene_elements[str(scene_boundary)] = scene

        all_scene = flatten(all_scene)
        uniq_places, cnt = np.unique(all_scene, return_counts=True)
        # n_scenes_by_length = max(1+int(len(mdf_no)/50), uniq_places.shape[0]) #actually it doesn;t do a thing
        # scene_top_k_frequent = uniq_places[np.argmax(cnt)] # take most frequent place
        scene_top_k_frequent = uniq_places[np.argsort(cnt)[::-1]]#[:n_scenes_by_length] 
        if 0:
            semantic_similar_places_max_set = self._semantic_similar_places_max_set_cover(tokens=all_scene)
        
        if self.semantic_token_deduplication and len(place_per_scene_elements) > self.min_shots_for_semantic_similar_dedup:
            scene_top_k_frequent = self._merge_semantic_similar_tokens(tokens=all_scene)

        if self.verbose and (self.semantic_token_deduplication and len(place_per_scene_elements) < self.min_shots_for_semantic_similar_dedup):
            print('Too short clip/scene to filter places by semantic simillarity')


        if 0: # unittest
            for i in np.arange(9,11,1):
                locals()['all_centroids_places_' + str(i)], locals()['sum_square_within_dist_' + str(i)], _ = self._cluster_based_place_inference(kmeans_n_cluster=i)
                
        return scene_top_k_frequent

# pre_defined_mdf_in_frame_no : given specific frames to process over that matches MDF file name 
    def summarize_scene_forward(self, movie_id: str, frame_boundary: list[list]= [], caption_type='vlm'):

        print("summarize_scene_forward : movie_id {} frame_boundary {} caption_type {}".format(movie_id, frame_boundary, caption_type))

        if caption_type != 'vlm' and caption_type != 'dense_caption':
            print("Unknown caption type option given : {} but should be (vlm/dense_caption)".format(caption_type))
            return

        if frame_boundary != []:
            if not (any(isinstance(el, list) for el in frame_boundary)):
                print("Frame boundary structure error : should be [[fstart1, fstop1][fstart2, fstop2]]")
            all_summ = list()
            all_mdf_no = list()
            for scn_frame in range(len(frame_boundary)):
                if len(frame_boundary[scn_frame]) == 2:
                    summ, mdf_no = self._summarize_scene_forward_scene(movie_id, frame_boundary[scn_frame], caption_type=caption_type)
                    all_summ.append(summ)
                    all_mdf_no.append(mdf_no)
                else:
                    print("Warning frame start/stop is missing need to supply 2 elements", frame_boundary[scn_frame])
            if self.write_res_to_db:
                self._insert_json_to_db(movie_id, all_summ, all_mdf_no)
            
            return all_summ
        else:
            summ, mdf_no = self._summarize_scene_forward_scene(movie_id, caption_type=caption_type)
            if self.write_res_to_db:
                self._insert_json_to_db(movie_id, summ, mdf_no)
            
            return summ
        

    def _summarize_scene_forward_scene(self, movie_id: str, frame_boundary: list[int]= [], caption_type:str= 'vlm'):

        all_caption = list()
        all_reid_caption = list()
        all_global_tokens = list()
        all_obj_LLM_OUTPUT_COLLECTION_cand = list()
        all_obj_LLM_OUTPUT_COLLECTION_cand_re_id = list()
        try:
            rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION) # + scene_elements
            input_type = get_input_type_from_db(rc_movie_id['pipeline_id'], "pipelines")
            if input_type == 'image':
                print("Movie_id {} of image is not supported but only videos".format(movie_id))
                return -1, -1
            # scene_elements = rc_movie_id['scene_elements']
            movie_name = os.path.basename(rc_movie_id['url_path'])
            self.movie_name = movie_name
            rc_reid = nebula_db.get_doc_by_key({'movie_id': movie_id}, REID_CLUES_COLLECTION)
            rc_reid_fusion = nebula_db.get_doc_by_key2({'movie_id': movie_id}, FUSION_COLLECTION)
        except Exception as e:
            print(e)        
            return -1, -1
# Actors name 
        man_names = list(np.unique(['James', 'Allan', 'Ron', 'George' ,'Nicolas', 'John', 'daniel', 'Henry', 'Jack', 'Leo', 'Oliver']))
        woman_names = list(np.unique(['Jane', 'Jennifer', 'Eileen', 'Sandra', 'Emma', 'Charlotte', 'Mia']))
        celeb_id_name_dict = dict()
        if rc_reid_fusion:
            print("Found actors names in DB")            
            celeb_id_name  = [{int(rec['rois'][0]['face_id']): rec['rois'][0]['reid_name']} for rec in rc_reid_fusion]
            for f in celeb_id_name:
                celeb_id_name_dict.update(f)   # Uniqeness actor name dict         

        print("Celeb list", celeb_id_name_dict)
        all_ids = list()
        # all_scene = list()
        id_prior_knowledge_among_many = dict()
        mdf_no = sorted(flatten(rc_movie_id['mdfs']))

        if frame_boundary != []:
            mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]

        semantic_similar_places = self._places_semantic_dedup(mdf_no, movie_id=movie_id)
# is indoor 
        is_indoor = any([True if x in  semantic_similar_places else False for x in ['lab', 'room', 'store', 'indoor', 'office', 'motel', 'home', 'house', 'bar', 'kitchen']])    #https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
        if is_indoor: # @@HK TODO TOP-gun has faces w/o outdoor hence MDF based on faces only is not a good option hence any() =>all()
            reid = True
        if isinstance(semantic_similar_places, (np.ndarray, np.generic)):
            scene_top_k_frequent = ' and or '.join(list(semantic_similar_places))
        else:
            scene_top_k_frequent = ' and or '.join(semantic_similar_places)


        for ix, frame_num in enumerate(tqdm.tqdm(mdf_no)):
                
            mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
            obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
            
            if caption_type == 'vlm':            
                if self.caption_callback:
                    caption = self.caption_callback(obj['url'])
                else:
                    caption = obj['global_caption']['blip']
            elif caption_type == 'dense_caption':
                caption = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)['candidate']
                all_obj_LLM_OUTPUT_COLLECTION_cand.append(caption)
            else:
                raise 
                
            scene = obj['global_scenes']['blip'][0][0]
            # all_scene.append(scene)
            all_global_tokens.extend([x[0] for x in obj['global_objects']['blip']])
            # mdf_re_id_dict = rc_reid['frames'][ix]
            mdf_re_id_dict = [x  for x in rc_reid['frames'] if x['frame_num']==frame_num]
            if mdf_re_id_dict: #and is_indoor: #places == 'indoor':  # conditioned on man in the scene if places==indoor
                reid = True
                assert(mdf_re_id_dict[0]['frame_num'] == frame_num)
                for id_rec in mdf_re_id_dict: # match many2many girl lady, woman to IDs at first
                    if 'face_no_id' in id_rec:
                        pass # TBD
                        
                    if 're-id' in id_rec:
                        ids_n = id_rec['re-id']
                        if ids_n: # in case face but no Re_id, skip
        #TODO @@HK a woaman in 1st scene goes to Id where same ID can appears later under" persons" 
        # Movies/-6576299517238034659 'a man in a car looking at Susan in the back seat" However there only 2 IDs "a man in a car looking at a woman in the back seat" no woman!! ''two men in a red car, one is driving and the other is driving'' but only 1 ID is recognized so ? 
                            all_ids.extend([ids['id'] for ids in ids_n])
                            # Gender exclusive
                            male_str = ['man', 'person', 'boy', 'human', 'someone']  #TODO add 'someone' to man/woman if they have celeb name remove the "a boy"
                            female_str = ['woman', 'lady' , 'girl']
                            many_person_str = ['men', 'women', 'person', 'people']

                            is_male = list(compress(male_str, [caption.find(x)>0 for x in male_str]))
                            is_female = list(compress(female_str, [caption.find(x)>0 for x in female_str]))

                            # if len(ids_n) > 1 or 1:
                            if 'men' in caption:
                                ids_phrase = ', ' + ' and '.join([celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]) + ', '
                                caption_re_id = caption.replace('men', 'men' + ids_phrase) 
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('men', 'men' + ids_phrase)
                            elif 'women' in caption:
                                ids_phrase = ', ' + ' and '.join([celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]) for ids in ids_n]) + ', '
                                caption_re_id = caption.replace('women', 'women' + ids_phrase) 
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('women', 'women' + ids_phrase)
                            elif 'person' in caption:
                                ids_phrase = ', ' + ' and '.join([celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]) + ', '
                                caption_re_id = caption.replace('person', 'person' + ids_phrase)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', 'person'  +ids_phrase)
                            elif 'people' in caption:  # @@HK to test effect  on Top gun re-id_frame0032 (a group of people sitting in an airplane with a man in the middle of the : Id within the people and one is with man need to refine)
                                ids_phrase = ', ' + ' and '.join([celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]) + ', '
                                caption_re_id = caption.replace('people', 'people with' + ids_phrase)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('people', 'people'  +ids_phrase)
# @@TODO : 1st singular IDs man/woman placing than plural and should be mutual exclusive , hence manage a list when you can take out items following placing singular IDS
# The people will by Id[0] + people and Id[1] instead of man =>gender classification needed, or ID with celeb name can cover up the whole issue
                            # ids = id_rec['re-id'][0]  # TODO take the relavant Gender based ID out of the IDs in the MDF
                            for ids in id_rec['re-id']:
                        # elif len(ids_n) == 1:
                            # ids = id_rec['re-id'][0]
                                if 'woman' in caption:
                                    if 'a woman' in caption :                    
                                        caption_re_id = caption.lower().replace('a woman', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a woman', woman_names[ids['id']])
                                    else:
                                        caption_re_id = caption.lower().replace('woman', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('woman', woman_names[ids['id']])
                                elif 'lady' in caption:
                                    if 'a lady' in caption:
                                        caption_re_id = caption.lower().replace('a lady', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a lady', woman_names[ids['id']])
                                    else:
                                        caption_re_id = caption.lower().replace('lady', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('lady', woman_names[ids['id']])
                                elif 'girl' in caption:
                                    if 'a girl' in caption:
                                        caption_re_id = caption.lower().replace('a girl', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a girl', woman_names[ids['id']])
                                    else:
                                        caption_re_id = caption.replace('girl', celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('girl', woman_names[ids['id']])
                                elif 'man' in caption:
                                    if 'a man' in caption:
                                        caption_re_id = caption.lower().replace('a man', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', man_names[ids['id']], 1)# TODO the obj_LLM_OUTPUT_COLLECTION_cand can chnage the a man to the man 
                                    else:
                                        caption_re_id = caption.replace('man', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
                                elif 'boy' in caption:
                                    if 'a boy' in caption:
                                        caption_re_id = caption.lower().replace('a boy', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a boy', man_names[ids['id']])
                                    else:
                                        caption_re_id = caption.replace('boy', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                elif 'person' in caption:
                                    if 'a person' in caption:
                                        caption_re_id = caption.lower().replace('a person', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', man_names[ids['id']])
                                    else:
                                        caption_re_id = caption.replace('person', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', man_names[ids['id']])
                                elif 'human' in caption:
                                    if 'a human' in caption:
                                        caption_re_id = caption.lower().replace('a human', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a human', man_names[ids['id']])
                                    else:
                                        caption_re_id = caption.lower().replace('human', celeb_id_name_dict.get(ids['id'], man_names[ids['id']]), 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('human', man_names[ids['id']])
                                # else: # could be found under people/plural list
                                #     print('Warning Id was found but was not associated n IDS: {} !!!! Caption: {} movie name: {}'.format(len(ids_n), caption, movie_name))
                                
                        
                all_reid_caption.append(caption_re_id)
            else:
                all_reid_caption.append(caption)

        # if all_reid_caption:
        seq_caption = ' and then '.join(all_reid_caption)
        seq_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x  for ix, x in enumerate(all_reid_caption)])
        n_uniq_ids = np.unique(all_ids).shape[0]
            # seq_dense_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x for ix, x in enumerate(all_obj_LLM_OUTPUT_COLLECTION_cand_re_id)])
        # else:
        #     seq_caption = ' and then '.join(all_caption)     # for ZS       
        #     seq_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x  for ix, x in enumerate(all_caption)])
        #     n_uniq_ids = 0
            


        if self.prompting_type == 'zeroshot':
            prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : {} Summary :'''.format(scene_top_k_frequent, n_uniq_ids, seq_caption_w_caption)
            # prompt = "Give a concise summary of the following video scene captions separated by the word 'then':{} Summary :".format(seq_caption)
        elif self.prompting_type == 'few_shot':
            prompt_prefix_caption = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption_w_caption, scene_top_k_frequent, n_uniq_ids, 
                                                    in_context_examples=self.one_shot_context_ex_prefix_caption, few_shot_seperator = '''###''',
                                                    prolog_refine=', by 2-3 sentences, ', uniq_id_prior_put_in_caption_end=True)
            prompt_prefix_then = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption, scene_top_k_frequent, n_uniq_ids, 
                                                        in_context_examples=self.one_shot_context_ex_prefix_then, few_shot_seperator = '''###''',
                                                        prolog_refine=', by 2-3 sentences, ', uniq_id_prior_put_in_caption_end=True)
            
            self.prompt_prefix_caption = prompt_prefix_caption
            self.prompt_prefix_then = prompt_prefix_then
            # https://github.com/NEBULA3PR0JECT/nebula3_llm_task/blob/8254fb4bb1f81ae87ece51f91cf76d5a778ed6f1/llm_orchestration.py#LL545C31-L548C34
        else:
            raise
        # concise 
        if self.gpt_type == 'HF_':
            hf_uservice = False
            model_id = "google/flan-ul2"#"google/flan-t5" #"distilbert-base-uncased"
            if hf_uservice:
                model_id = "google/flan-ul2"#"google/flan-t5" #"distilbert-base-uncased"
                data = query("The goal of life is [MASK].", model_id, api_token)
                while 'error' in data.keys():
                    print(data)
            else: #Inference API
                # inference = InferenceApi(repo_id="bert-base-uncased", token=api_token)
                InferenceApi(repo_id="gpt-j-6b-shakespeare", token=api_token)
                res = inference(inputs="The goal of life is [MASK].")

        elif self.gpt_type == 'text-davinci-003':
            if len(prompt_prefix_caption) >4096-120: # MosaicML MPT-7B-Instruct 2K (https://huggingface.co/mosaicml/mpt-7b-instruct, https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)
                print('Context window is too long', len(prompt_prefix_caption))
            opportunities = 10
            while (opportunities):
                rc = gpt_execute(prompt_prefix_then, model='text-davinci-003', n=1, max_tokens=256)
                if rc == []:
                    time.sleep(1)
                    opportunities -= 1
                    continue
                else:
                    break

        elif self.gpt_type == 'chat_gpt_3.5' or self.gpt_type == 'gpt-4' or self.gpt_type == 'gpt-3.5-turbo-16k':
            if len(prompt_prefix_then) > 4096-256 and self.gpt_type == 'chat_gpt_3.5':
                print('Context window is too long', len(prompt_prefix_then))
            if len(prompt_prefix_then) > 4*4096-256 and self.gpt_type == 'gpt-3.5-turbo-16k':
                print('Context window is too long', len(prompt_prefix_then))
            if len(prompt_prefix_then) > 32*1024-256 and self.gpt_type == 'gpt-4':
                print('Context window is too long', len(prompt_prefix_then))
            
            opportunities = 10
            while (opportunities):
                rc = self.chatgpt.completion(prompt_prefix_caption, n=1, max_tokens=256, model=self.gpt_type) #TODO add ChatGPT 16K
                # rc = self.chatgpt.completion(prompt_prefix_then, n=1, max_tokens=256, model=self.gpt_type) #TODO add ChatGPT 16K
                if rc == []:
                    time.sleep(1)
                    opportunities -= 1
                    continue
                else:
                    break
        
        if n_uniq_ids >0:
            rc[0]  = '''The video shows {} main character. {}'''.format(n_uniq_ids, rc[0]) 
        
        return rc[0], mdf_no

    def _insert_json_to_db(self, movie_id:str, scene_summ: str, mdf_no: list):

        combined_json = {'movie_id': movie_id, 'SM_MDF': mdf_no, 'scene_summary': scene_summ}
        res = nebula_db.write_doc_by_key(combined_json, self.collection_name, overwrite=True, key_list=['movie_id'])
        print("Successfully inserted to database. Collection name: {}".format(self.collection_name))

        return

    def _cluster_based_place_inference(self, kmeans_n_cluster: int =None, top_k_by_cluster: int=5):
        
        df = pd.read_csv(os.path.join("/notebooks/multi_modal", "ontology_blip2_itc_per_mdf_top_gun.csv"), index_col=False)       
        # eval(df['frame3907.jpg'].dropna().values[0])
        ontology_list_len = [len(eval(df[x].dropna().values[0])) for x in df.keys()][0]
        n_mdf = len(df)
        
        if kmeans_n_cluster is None:
            kmeans_n_cluster = 1+int(n_mdf/30)

        mdf_places_retrival_score = [eval(df[x].dropna().values[0]) for x in df.keys()]
        mdf_no = [x for x in df.keys()]
        vlm_score_embed_per_mdf = np.array([y[1] for x in mdf_places_retrival_score for y in x]).reshape((n_mdf , -1))  #[x for l in lst for x in l]
        ontology_by_csv = np.array([y[0] for x in mdf_places_retrival_score for y in x]).reshape((n_mdf , -1))[0, :]
        

        # Sanity
        if 0:
            mdf_k = 'frame0014.jpg' # GT is 
            score_14 = eval(df[mdf_k].dropna().values[0])
            blip2_itc_mdf = np.array([x[1] for x in score_14]).reshape((ontology_list_len , -1))
            blip2_itc_text = np.array([x[0] for x in score_14]).reshape((ontology_list_len , -1))
            top_k_ind_per_mdf = np.argsort(blip2_itc_mdf.reshape(-1))[::-1][:top_k_by_cluster]
            # ontology_by_csv[top_k_ind_per_mdf]
            # all(ontology_by_csv[top_k_ind_per_mdf] == ['lecture room', 'conference room', 'television room', 'auditorium', 'classroom'])
            assert(all(ontology_by_csv[top_k_ind_per_mdf] == ['lecture room', 'conference room', 'television room', 'auditorium', 'classroom']))
            print([x for x in eval(df[mdf_k].dropna().values[0]) if x[0]=="lecture room"])
        # import sklearn 
        # print('The scikit-learn version is {}.'.format(sklearn.__version__)) 
        kmeans = KMeans(n_clusters=kmeans_n_cluster, random_state=0, n_init="auto").fit(vlm_score_embed_per_mdf)
        sum_square_within_dist = -kmeans.score(vlm_score_embed_per_mdf)
        assert(kmeans.cluster_centers_.shape[1]==ontology_list_len)
    # Per cluster members in terms of MDf No.
        classify_mdf = kmeans.predict(vlm_score_embed_per_mdf)
        cluster_mdfs = [list(compress(mdf_no, (classify_mdf == x))) for x in np.unique(classify_mdf)]

        all_centroids_places = list()
        for clust in np.arange(kmeans_n_cluster):
            top_k_ind_per_cluster = np.argsort(kmeans.cluster_centers_[clust, :])[::-1][:top_k_by_cluster]
            if self.verbose:
                print(kmeans.cluster_centers_[clust, :][top_k_ind_per_cluster])
                print(ontology_by_csv[top_k_ind_per_cluster])

            all_centroids_places.append(ontology_by_csv[top_k_ind_per_cluster])

        return all_centroids_places, sum_square_within_dist, cluster_mdfs


    def _semantic_similar_places_max_set_cover(self, tokens: list, topk: int=10, greedy: bool=True) -> str:

        uniq_places, cnt = np.unique(tokens, return_counts=True)
        frequent_uniq_places = uniq_places[np.argsort(cnt)[::-1]] # sort according to frequency
        # SentenceBERT score
        similarity_places = self.evaluator.compute_triplet_scores(src=[tuple([x]) for x in frequent_uniq_places], dst = [tuple([x]) for x in frequent_uniq_places])
        dist_places = 1- similarity_places
        max_set_entity = list()
        max_set_id = list()
        if greedy:
            # np.fill_diagonal(similarity_places, 0)

            max_set_entity.append(frequent_uniq_places[0])
            max_set_id.append(0)
            while len(max_set_entity) <topk:
                # np.take(dist_places, max_set_id, axis=1)
                greedy_id = np.argmax(np.take(dist_places, max_set_id, axis=1).sum(axis=1))
                max_set_id.append(greedy_id)
                max_set_entity.append(frequent_uniq_places[greedy_id])
                
        else:
            raise
        return max_set_entity

#topk == -1 then no additional top k 

    def _merge_semantic_similar_tokens(self, tokens: list, topk:int=10, sim_th: int=0.7, verbose:bool=False) -> str:

        uniq_places, cnt = np.unique(tokens, return_counts=True)
        frequent_uniq_places = uniq_places[np.argsort(cnt)[::-1]] # sort according to frequency
        # SentenceBERT score
        similarity_places = self.evaluator.compute_triplet_scores(src=[tuple([x]) for x in frequent_uniq_places], dst = [tuple([x]) for x in frequent_uniq_places])
        
        
        np.fill_diagonal(similarity_places, 0)
        simillar_places = list()
        for ix, ele in enumerate(similarity_places):
            if any(similarity_places[ix:, ix]>sim_th): # lower diagonal simillar ones will be removed 
                # print(frequent_uniq_places, similarity_places[:ix, ix])
                removed_places = frequent_uniq_places[np.where(similarity_places[ix:, ix]>sim_th)[0]+ix]
                # print(removed_places, ix)
                if verbose:
                    print("{} Like {} ".format(removed_places, frequent_uniq_places[ix]))
                simillar_places.extend(removed_places)
        
        simillar_places = [x.strip() for x in simillar_places]
        simillar_places = np.unique(simillar_places)
        g = list(frequent_uniq_places)
        [g.remove(x) for x in simillar_places]
        
        if topk != -1:
            top_k_uniq_not_sim = g[:topk]
        else: 
            g = top_k_uniq_not_sim
        
        return top_k_uniq_not_sim # todo elevator/door exhibit as door

class LLMBase(ABC):
    @abstractmethod
    def completion(prompt_template: str, *args, n=1, **kwargs):
        pass
# pip install openai==0.27.0 --target /notebooks/pip_install/

# Movies/7417592353856606351
#subsequent captions of key-frames 1 ### 2 

def get_few_shot_prompt_paragraph_based_to_tuple_4K(query_paragraph: str, scene: str, n_uniq_ids: int, 
                                                    in_context_examples: str, **kwargs):
    few_shot_seperator = kwargs.pop('few_shot_seperator', None)
    prolog_refine = kwargs.pop('prolog_refine', '')
    uniq_id_prior_put_in_caption_end = kwargs.pop('uniq_id_prior_put_in_caption_end', None)
    
    # prolog = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place. Example: Video captions '''

# Alternatives TODO HK@@ : Provide a summary for the following article  ; move the "that were taken place at {}" to the end of prompt ask for action
    if uniq_id_prior_put_in_caption_end:
        prolog = '''Summarize the video {}given the captions that were taken place at {}. Example of video captions and summary: '''
        prolog = prolog.format(prolog_refine, scene)
    else:
        if n_uniq_ids > 0:
            prolog = '''Summarize the video {}given the captions that were taken place at {} with {} persons. Tell what they are doing. Example of video captions and summary: '''
            prolog = prolog.format(prolog_refine, scene, n_uniq_ids)
        else:
            prolog = '''Summarize the video {}given the captions that were taken place at {}. Tell what they are doing. Example of video captions and summary: '''
            prolog = prolog.format(prolog_refine, scene)

    if uniq_id_prior_put_in_caption_end:
        epilog = '''Video captions: {}{}. Video Summary: '''
        suffix_prior = '''. The captions are noisy and sometimes include people who are not there. We know for certain there are exactly {} main characters in the scene. Tell what they are doing and thier names'''.format(n_uniq_ids)
        epilog = epilog.format(query_paragraph, suffix_prior)
    else:
        epilog = '''Video captions: {}. Video Summary: '''
        epilog = epilog.format(query_paragraph, suffix_prior, n_uniq_ids)

    if few_shot_seperator:
        in_context_examples = in_context_examples  + '\n{}\n' .format(few_shot_seperator)   #
    prompt = '{}{}{}'.format(prolog, in_context_examples, epilog).strip()

    # print(prompt)
    return prompt
"""
        prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : 
            Example:'' {} '' {} Video summary :'''.format(scene, n_uniq_ids, shot_example, seq_caption_w_caption)

"""

# pip install openai
# pip install --upgrade openai
# pip show openai
class ChatGptLLM(LLMBase):
    def completion(self, prompt_template: str, *args, n=1, model=CHAT_GPT_MODEL, **kwargs):
        prompt = prompt_template.format(*args) 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]  
        max_tokens = kwargs.pop('max_tokens', 256)
        response = openai.ChatCompletion.create(messages=messages, max_tokens=max_tokens, n=n, model=model, **kwargs)
        return [x['message']['content'].strip() for x in response['choices']]


class MovieImageId(NamedTuple):
    movie_id: str
    frame_num: int

def process_benchmark(benchmark_name, **kwargs):
    results = []
    if not nebula_db.db.has_collection(EVAL_COLLECTION_NAME):
        nebula_db.db.create_collection(EVAL_COLLECTION_NAME)
    benchmark = list(nebula_db.db.collection('Movies').find({'misc.benchmark_name': benchmark_name}))
    print("Processing {} items".format(len(benchmark)))
    for mobj in benchmark:
        assert(mobj['mdfs'] == [[0]])
        mid = MovieImageId(mobj['_id'],0)
        curr_key = {'movie_id': mobj['_id'], 'benchmark_name': mobj['misc']['benchmark_name'], 'benchmark_tag': mobj['misc']['benchmark_tag']}
        curr_doc = nebula_db.get_doc_by_key2(curr_key, EVAL_COLLECTION_NAME)
        if curr_doc:
            print("Found existing eval result, moving on: ")
            # print(curr_doc.pop())
            continue
        try:
            rc = process_recall(mid, **kwargs)
        except:
            print("Failed to evaluate mid: {}".format(mid[0]))
            continue
        rc['movie_id']=mid[0]
        rc['benchmark_name']=mobj['misc']['benchmark_name']
        rc['benchmark_tag']=mobj['misc']['benchmark_tag']
        print(rc)
        results.append(rc)
        rc1 = nebula_db.write_doc_by_key(rc,EVAL_COLLECTION_NAME,key_list=['image_id', 'movie_id', 'benchmark_name','benchmark_tag'])
        print("Result from writing:")
        print(rc1)
    return results


def main():

    summarize_scene = SummarizeScene()

    result_path = "/notebooks/nebula3_playground"
    unique_run_name = str(int(time.time()))
    # http://209.51.170.37:8087/docs
    add_action = True

    results = list()
    all_movie_id = list()

    if add_action:
        all_movie_id.append('Movies/7023181708619934815')
    all_movie_id.append('Movies/-6372550222147686303')
    all_movie_id.append('Movies/-3323239468660533929') #actionclipautoautotrain00616.mp4
    all_movie_id.append('Movies/-6372550222147686303')
    all_movie_id.append('Movies/889658032723458366')
    all_movie_id.append('Movies/-6576299517238034659')
    all_movie_id.append('Movies/-5723319113316714990')
    all_movie_id.append('Movies/2219594956981209558')
    all_movie_id.append('Movies/6293447408186786707')



    csv_file_name = 'scene_summarization_' + str(unique_run_name) + '_' + str(summarize_scene.prompting_type) + '_' + str(summarize_scene.gpt_type) +'.csv'

    for movie_id in all_movie_id:

        frame_boundary = []
        # if movie_id == 'Movies/-6372550222147686303':
        #     frame_boundary = [[834, 1181]]
        if movie_id == 'Movies/-6372550222147686303':  # dummy for debug
             frame_boundary = [[834, 1181], [14,272]]
        if movie_id == 'Movies/-5723319113316714990':
            frame_boundary = [[197, 320]]
        if movie_id == 'Movies/6293447408186786707':
            frame_boundary = [[1035, 1290]]

        scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary, caption_type='dense_caption')
        # scn_summ = summarize_scene.summarize_scene_forward(movie_id) # for all clip w/o frame boundaries 

        print("Movie: {} Scene summary : {}".format(movie_id, scn_summ))

        # results.append({'movie_id':movie_id, 'summary': scn_summ, 'movie_name':movie_name, 'prompt': prompt_prefix_then, 'mdf_no': mdf_no})
        results.append({'movie_id':movie_id, 'summary': scn_summ, 
                        'movie_name':summarize_scene.movie_name, 'prompt_prefix_caption' : summarize_scene.prompt_prefix_caption})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, csv_file_name), index=False)


if __name__ == '__main__':
    main()

"""
TODO : add option for more verbality conditioned on more unique(tokens) from all the MDFs in the scene

FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
'gpt-4-32k'
'gpt-4-32k-0314'
'gpt-4'

input_type = self.get_input_type_from_db(pipeline_id, "pipelines")
from database.arangodb import DBBase
def get_input_type_from_db(pipeline_id, collection):
    nre = DBBase()
    pipeline_data = nre.get_doc_by_key({'_key': pipeline_id}, collection)
    if pipeline_data:
        if "dataset" in pipeline_data["inputs"]["videoprocessing"]:
            input_type = pipeline_data["inputs"]["videoprocessing"]["dataset"]["type"]
        else:
            input_type = pipeline_data["inputs"]["videoprocessing"]["movies"][0]["type"]
    return input_type

"""