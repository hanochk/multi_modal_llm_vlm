# export PYTHONPATH=/notebooks/pip_install/
# pip install -U transformers


import os
from database.arangodb import NEBULA_DB
from typing import NamedTuple
from nebula3_experiments.prompts_utils import *
import time
from nebula3_experiments.vg_eval import VGEvaluation

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
FS_SAMPLES = 5                   # Samples for few-shot gpt

from abc import ABC, abstractmethod
import openai
from itertools import compress
from huggingface_hub.inference_api import InferenceApi
import requests

import importlib
print(importlib.metadata.version('openai'))


class LLMBase(ABC):
    @abstractmethod
    def completion(prompt_template: str, *args, n=1, **kwargs):
        pass
# pip install openai==0.27.0 --target /notebooks/pip_install/

# Movies/7417592353856606351
one_shot_context_ex_prefix_caption = '''Caption 1: Susan standing in front of a tv screen with a camera. 
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
                            Caption 13: Susan standing in front of a bookcase with a bookcase in the background. 
                            Video Summary: 3 persons in storage room, Susan, Michael, & Tom are all present inside a small dark room filled with boxes stacked up against one wall. The three people appear very tense as they look around frantically.'''
one_shot_context_ex_prefix_caption = one_shot_context_ex_prefix_caption.replace('\n ',' ').replace("  ", "")

one_shot_context_ex_prefix_then = '''Susan standing in front of a tv screen with a camera and then Michael with a body in the middle of the screen and then Tom standing next to a giant human in the shape of a human and then Susan standing in front of a camera and looking at the camera and then Susan in a blue dress and a man in a suit and then Susan in a room with a bike in the background and then Tom sitting on a chair in the middle of a room and then Michael with a mask on his face and Michael with a mask on his and then Tom in a suit and tie with a giant head and then Michael in a suit with the capt that says,'i'm ' and then Tom in a spider suit sitting on a train car and then Tom in a blue shirt and Tom in a black shirt and then Susan standing in front of a bookcase with a bookcase in the background.'''

def get_few_shot_prompt_paragraph_based_to_tuple_4K(query_paragraph, scene, n_uniq_ids, in_context_examples, **kwargs):
    few_shot_seperator = kwargs.pop('few_shot_seperator', None)
    prolog_refine = kwargs.pop('prolog_refine', '')
    
    # prolog = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place. Example: Video captions '''

    epilog = '''Video captions: {}. Video Summary: '''
    if n_uniq_ids > 0:
        prolog = '''Summarize the video {}given the captions that were taken place at {} with {} persons. Example of video captions: '''
        prolog = prolog.format(prolog_refine, scene, n_uniq_ids)
    else:
        prolog = '''Summarize the video {}given the captions that were taken place at {}. Example of video captions: '''
        prolog = prolog.format(prolog_refine, scene)


    epilog = epilog.format(query_paragraph)
    if few_shot_seperator:
        in_context_examples = in_context_examples  + '\n{}\n' .format(few_shot_seperator)   #
    prompt = '{} {} {}'.format(prolog, in_context_examples, epilog).strip()

    print(prompt)
    return prompt
"""
        prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : 
            Example:'' {} '' {} Video summary :'''.format(scene, n_uniq_ids, shot_example, seq_caption_w_caption)

"""
# Hosted Inference API :HF HTTP request
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Model Hub is where the members of the Hugging Face community can host all of their model checkpoints 
# hf = HuggingFaceHub(repo_id="google/flan-t5-xl")
if 0:
    hf_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    model = HuggingFaceLLM(hf_model, tokenizer)

api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


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

def flatten(lst): return [x for l in lst for x in l]

def get_dialog_caption (movie_id, frame_num):
    for res in nebula_db.nebula_db.db.collection("s4_visual_clues").find({'movie_id': movie_id, 'frame_num': frame_num}):
        pass

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

result_path = "/notebooks/nebula3_playground"
unique_run_name = str(int(time.time()))
prompting_type = 'few_shot' #'zeroshot'
evaluator = VGEvaluation()
add_action = True

results = list()
all_movie_id = list()
# all_movie_id.append('Movies/-7183176057624492662') # bad quality, bad ReID for indoor based summarization
if add_action:
    all_movie_id.append('Movies/7023181708619934815')
all_movie_id.append('Movies/-6372550222147686303')
all_movie_id.append('Movies/-6576299517238034659')
all_movie_id.append('Movies/889658032723458366')
all_movie_id.append('Movies/-5723319113316714990')
all_movie_id.append('Movies/2219594956981209558')
all_movie_id.append('Movies/6293447408186786707')
# all_movie_id.append('Movies/7417592353856606351') # used for one-shot

man_names = list(np.unique(['James', 'Michael', 'Tom', 'George' ,'Nicolas', 'John', 'daniel', 'Henry', 'Jack', 'Leo', 'Oliver']))
woman_names = list(np.unique(['Susan', 'Jennifer', 'Eileen', 'Sandra', 'Emma', 'Charlotte', 'Mia']))

places = 'indoor'
print("promting_type", prompting_type)


gpt_type = 'gpt-4'#'text-davinci-003' #'chat_gpt_3.5' #'HF_'
if gpt_type == 'HF_':
    InferenceApi(repo_id="gpt-j-6b-shakespeare", token=api_token)
elif gpt_type == 'chat_gpt_3.5' or gpt_type == 'gpt-4':
    chatgpt = ChatGptLLM()
elif gpt_type == 'text-davinci-003':
    context_win = 4096


csv_file_name = 'scene_summarization_' + str(unique_run_name) + '_' + str(prompting_type) + '_' + str(gpt_type) +'.csv'

for movie_id in all_movie_id:
# movie_id = all_movie_id[0]

# obj_LLM_OUTPUT_COLLECTION = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION) #s4_visual_clues/31275667
# cand = obj['candidate']

    all_caption = list()
    all_reid_caption = list()
    all_global_tokens = list()
    all_obj_LLM_OUTPUT_COLLECTION_cand = list()
    all_obj_LLM_OUTPUT_COLLECTION_cand_re_id = list()
    # cusrsor = nebula_db.get_doc_by_key2({'movie_id': movie_id}, MOVIES_COLLECTION)
    rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION)
    movie_name = os.path.basename(rc_movie_id['url_path'])
    rc_reid = nebula_db.get_doc_by_key({'movie_id': movie_id}, REID_CLUES_COLLECTION)
    # rc['mdfs']
    # frames_num_dict = dict(zip(flatten(rc_movie_id['mdfs']),rc_movie_id['mdfs_path']))
    all_ids = list()
    all_scene = list()
    id_prior_knowledge_among_many = dict()

    mdf_no = sorted(flatten(rc_movie_id['mdfs']))
    if movie_id == 'Movies/-6372550222147686303':
        frame_boundary = [834, 1181]
        mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]
    # Place voting
    for ix, frame_num in enumerate(mdf_no):
        mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
        obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
        scene = obj['global_scenes']['blip'][0][0]
        all_scene.append(scene)
    uniq_places, cnt = np.unique(all_scene, return_counts=True)
    n_scenes_by_length = 1+int(len(mdf_no)/50)
    # scene_top_k_frequent = uniq_places[np.argmax(cnt)] # take most frequent place
    scene_top_k_frequent = uniq_places[np.argsort(cnt)[::-1]][:n_scenes_by_length] 
    frequent_uniq_places = uniq_places[np.argsort(cnt)[::-1]]
    similarity_places = evaluator.compute_triplet_scores(src=[tuple([x]) for x in frequent_uniq_places], dst = [tuple([x]) for x in frequent_uniq_places])
    
    topk = 10
    np.fill_diagonal(similarity_places, 0)
    places_sim_th = 0.7
    simillar_places = list()
    for ix, ele in enumerate(similarity_places):
        if any(similarity_places[ix:, ix]>places_sim_th): # lower diagonal simillar ones will be removed 
            # print(frequent_uniq_places, similarity_places[:ix, ix])
            removed_places = frequent_uniq_places[np.where(similarity_places[ix:, ix]>places_sim_th)[0]+ix]
            # print(removed_places, ix)
            print("{} Like {} ".format(removed_places, frequent_uniq_places[ix]))
            simillar_places.extend(removed_places)
    
    simillar_places = [x.strip() for x in simillar_places]
    simillar_places = np.unique(simillar_places)
    g = list(frequent_uniq_places)
    [g.remove(x) for x in simillar_places]
    top_k_uniq_not_sim = g[:topk]


    is_indoor = any([True if x in  scene_top_k_frequent else False for x in ['lab', 'room', 'store', 'indoor', 'office', 'motel', 'home', 'house', 'bar', 'kitchen']])    #https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
    if is_indoor:
        reid = True
    scene_top_k_frequent = ' and or '.join(list(scene_top_k_frequent))

    for ix, frame_num in enumerate(mdf_no):
            
        mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
        obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
        caption = obj['global_caption']['blip']
        scene = obj['global_scenes']['blip'][0][0]
        all_scene.append(scene)
        all_global_tokens.extend([x[0] for x in obj['global_objects']['blip']])
        obj_LLM_OUTPUT_COLLECTION_cand = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)['candidate']
        all_obj_LLM_OUTPUT_COLLECTION_cand.append(obj_LLM_OUTPUT_COLLECTION_cand)
        # mdf_re_id_dict = rc_reid['frames'][ix]
        mdf_re_id_dict = [x  for x in rc_reid['frames'] if x['frame_num']==frame_num]
        if len(mdf_re_id_dict) >1:
            print('ka')
        if mdf_re_id_dict: #and is_indoor: #places == 'indoor':  # conditioned on man in the scene if places==indoor
            reid = True
            assert(mdf_re_id_dict[0]['frame_num'] == frame_num)
            for id_rec in mdf_re_id_dict: # match many2many girl lady, woman to IDs at first
                if 'face_no_id' in id_rec:
                    pass # TBD
                    
                if 're-id' in id_rec:
                    ids_n = id_rec['re-id']
    #TODO @@HK a woaman in 1st scene goes to Id where same ID can appears later under" persons" 
    # Movies/-6576299517238034659 'a man in a car looking at Susan in the back seat" However there only 2 IDs "a man in a car looking at a woman in the back seat" no woman!! ''two men in a red car, one is driving and the other is driving'' but only 1 ID is recognized so ? 
                    all_ids.extend([ids['id'] for ids in ids_n])
                    
                    male_str = ['man', 'person', 'boy', 'human']
                    female_str = ['woman', 'lady' , 'girl']
                    many_person_str = ['men', 'women', 'person']

                    is_male = list(compress(male_str, [caption.find(x)>0 for x in male_str]))
                    is_female = list(compress(female_str, [caption.find(x)>0 for x in female_str]))

                    if len(ids_n) > 1 or 1:
                        if 'men' in caption:
                            ids_phrase = ', ' + ' and '.join([man_names[ids['id']] for ids in ids_n]) + ', '
                            caption_re_id = caption.replace('men', 'men' + ids_phrase) 
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('men', 'men' + ids_phrase)
                        elif 'women' in caption:
                            ids_phrase = ', ' + ' and '.join([woman_names[ids['id']] for ids in ids_n]) + ', '
                            caption_re_id = caption.replace('women', 'women' + ids_phrase) 
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('women', 'women' + ids_phrase)
                        elif 'person' in caption:
                            ids_phrase = ', ' + ' and '.join([man_names[ids['id']] for ids in ids_n]) + ', '
                            caption_re_id = caption.replace('person', 'person' + ids_phrase)
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', 'person'  +ids_phrase)
                        # else:
                        #     print('Warning Multiple Ids were found but were not associated !!!!')
                    # Reduction when many Ids but only 1 ReID or single ID keyword/str " men in... one man"
                    if 1:
                        ids = id_rec['re-id'][0]  # TODO take the relavant Gender based ID out of the IDs in the MDF
                    # elif len(ids_n) == 1:
                        # ids = id_rec['re-id'][0]
                        if 'woman' in caption:
                            if 'a woman' in caption :                    
                                caption_re_id = caption.lower().replace('a woman', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a woman', woman_names[ids['id']])
                            else:
                                caption_re_id = caption.lower().replace('woman', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('woman', woman_names[ids['id']])
                        elif 'lady' in caption:
                            if 'a lady' in caption:
                                caption_re_id = caption.lower().replace('a lady', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a lady', woman_names[ids['id']])
                            else:
                                caption_re_id = caption.lower().replace('lady', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('lady', woman_names[ids['id']])
                        elif 'girl' in caption:
                            if 'a girl' in caption:
                                caption_re_id = caption.lower().replace('a girl', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a girl', woman_names[ids['id']])
                            else:
                                caption_re_id = caption.replace('girl', woman_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('girl', woman_names[ids['id']])
                        elif 'man' in caption:
                            if 'a man' in caption:
                                caption_re_id = caption.lower().replace('a man', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', man_names[ids['id']], 1)# TODO the obj_LLM_OUTPUT_COLLECTION_cand can chnage the a man to the man 
                            else:
                                caption_re_id = caption.replace('man', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
                        elif 'boy' in caption:
                            if 'a boy' in caption:
                                caption_re_id = caption.lower().replace('a boy', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a boy', man_names[ids['id']])
                            else:
                                caption_re_id = caption.replace('boy', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('boy', man_names[ids['id']])
                        # elif 'person' in caption:
                        #     if 'a person' in caption:
                        #         caption_re_id = caption.lower().replace('a person', man_names[ids['id']])
                        #         llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', woman_names[ids['id']])
                        #     else:
                        #         caption_re_id = caption.replace('person', man_names[ids['id']])
                        #         llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', woman_names[ids['id']])
                        elif 'person' in caption:
                            if 'a person' in caption:
                                caption_re_id = caption.lower().replace('a person', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', man_names[ids['id']])
                            else:
                                caption_re_id = caption.replace('person', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', man_names[ids['id']])
                        elif 'human' in caption:
                            if 'a human' in caption:
                                caption_re_id = caption.lower().replace('a human', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a human', man_names[ids['id']])
                            else:
                                caption_re_id = caption.lower().replace('human', man_names[ids['id']], 1)
                                # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('human', man_names[ids['id']])
                        else:
                            print('Warning Id was found but was not associated n IDS: {} !!!! Caption: {} movie name: {}'.format(len(ids_n), caption, movie_name))
                    
                    
            all_reid_caption.append(caption_re_id)
            # all_caption.append(caption)
            # all_obj_LLM_OUTPUT_COLLECTION_cand_re_id.append(llm_out_cand_re_id)
        else:
            all_reid_caption.append(caption)
    # obj = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)
        # get_dialog_caption(mid.movie_id,mid.frame_num)
    # uniq_places, cnt = np.unique(all_scene, return_counts=True)
    # scene_top_k_frequent = uniq_places[np.argmax(cnt)] # take most frequent place

    if all_reid_caption:
        seq_caption = ' and then '.join(all_reid_caption)
        n_uniq_ids = np.unique(all_ids).shape[0]
        seq_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x  for ix, x in enumerate(all_reid_caption)])
        seq_dense_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x for ix, x in enumerate(all_obj_LLM_OUTPUT_COLLECTION_cand_re_id)])
    else:
        seq_caption = ' and then '.join(all_caption)     # for ZS       
        seq_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x  for ix, x in enumerate(all_caption)])
        n_uniq_ids = 0
        


    if prompting_type == 'zeroshot':
        if 0:
            prompt = "Summarize the following video transcription of a scene given segmented captions separated by the word 'then':{} Summary :".format(seq_caption)
        elif 0:
            prompt = "Give a concise summary of the following video transcription of a scene given segmented captions separated by the word 'then':{} Summary :".format(seq_caption)
        elif 0:
            prompt = "What is the theme of the following video scene given captions separated by the word 'then':{} Summary :".format(seq_caption)
            prompt = "Summarize the captions out of a video scene separated by the word 'then':{} Summary :".format(seq_caption)
        elif 1:
            prompt = "Summarize the video scene by the shot captions separated by the word 'then', the scene is at the {} :{} Summary :".format(scene_top_k_frequent, seq_caption)
            prompt = "Summarize the video shots taken at the {} separated by the word 'then' :{} Summary :".format(scene_top_k_frequent, seq_caption)
            prompt = "Summarize the video that was taken at the {} by 2-3 sentences. The video shots are separated by the word 'then' :{} Summary :".format(scene_top_k_frequent, seq_caption)
            prompt = "Summarize the video that was taken at the place of {} by 2-3 sentences. The video shots are separated by the word 'then'. Start by telling how many persons and what place  :{} Summary :".format(scene_top_k_frequent, seq_caption)
            prompt = '''Summary, by only few sentences, the video that was taken place at {} with {} persons. The video shots are separated by the word 'then'. Start by telling how many persons and what place  : {}. Summary :'''.format(scene_top_k_frequent, n_uniq_ids, seq_caption)
            prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : {} Summary :'''.format(scene_top_k_frequent, n_uniq_ids, seq_caption_w_caption)
            
            if 0:
            # Dense caption
                prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : {} Summary :'''.format(scene_top_k_frequent, n_uniq_ids, seq_dense_caption_w_caption)
    # 'Summarize the video scene given the shots that were taken place at storage room with 3 persons. The video shot captions are separated by the word 'then'. Start by telling how many persons and where it was taken place  :'        

        else:
            prompt = "Give a concise summary of the following video scene captions separated by the word 'then':{} Summary :".format(seq_caption)
    elif prompting_type == 'few_shot':
        prompt_prefix_caption = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption_w_caption, scene_top_k_frequent, n_uniq_ids, 
                                                in_context_examples=one_shot_context_ex_prefix_caption, few_shot_seperator = '''###''',
                                                prolog_refine=', by 2-3 sentences, ')
        prompt_prefix_then = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption, scene_top_k_frequent, n_uniq_ids, 
                                                    in_context_examples=one_shot_context_ex_prefix_then, few_shot_seperator = '''###''',
                                                    prolog_refine=', by 2-3 sentences, ')
        
        
        # https://github.com/NEBULA3PR0JECT/nebula3_llm_task/blob/8254fb4bb1f81ae87ece51f91cf76d5a778ed6f1/llm_orchestration.py#LL545C31-L548C34
    else:
        raise
    # concise 
    if gpt_type == 'HF_':
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

    elif gpt_type == 'text-davinci-003':
        if len(prompt_prefix_caption) >4096-120: # MosaicML MPT-7B-Instruct 2K (https://huggingface.co/mosaicml/mpt-7b-instruct, https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)
            print('Prompt too long!!!')
        opportunities = 10
        while (opportunities):
            rc = gpt_execute(prompt_prefix_then, model='text-davinci-003', n=1, max_tokens=256)
            if rc == []:
                time.sleep(1)
                opportunities -= 1
                continue
            else:
                break

    elif gpt_type == 'chat_gpt_3.5' or gpt_type == 'gpt-4':
        if len(prompt_prefix_then) > 4096-256:
            print('Context window is too big')
        opportunities = 10
        while (opportunities):
            if gpt_type == 'gpt-4':
                rc = chatgpt.completion(prompt_prefix_then, n=1, max_tokens=256, model='gpt-4')
            else:
                rc = chatgpt.completion(prompt_prefix_then, n=1, max_tokens=256)
            if rc == []:
                time.sleep(1)
                opportunities -= 1
                continue
            else:
                break
    # chatgpt install
        # from chatgpt_wrapper import ChatGPT
        # bot = ChatGPT()
        # response = bot.ask(prompt)
        # print(response)  # prints the response from chatGPT
    
    results.append({'movie_id':movie_id, 'summary': rc[0], 'movie_name':movie_name, 'prompt': prompt_prefix_then, 'mdf_no': mdf_no})
df = pd.DataFrame(results)
df.to_csv(os.path.join(result_path, csv_file_name), index=False)


"""
TODO : add option for more verbality conditioned on more unique(tokens) from all the MDFs in the scene

FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
'gpt-4-32k'
'gpt-4-32k-0314'
'gpt-4'

                if len(ids_n) > 1:
                    if 'men' in caption:
                        ids_phrase = ', ' + ' and '.join([man_names[ids['id']] for ids in ids_n]) + ', '
                        caption_re_id = caption.replace('men', 'men' + ids_phrase) 
                        llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('men', 'men' + ids_phrase)
                    elif 'women' in caption:
                        ids_phrase = ', ' + ' and '.join([woman_names[ids['id']] for ids in ids_n]) + ', '
                        caption_re_id = caption.replace('women', 'women' + ids_phrase) 
                        llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('women', 'women' + ids_phrase)
                    elif 'person' in caption:
                        ids_phrase = ', ' + ' and '.join([man_names[ids['id']] for ids in ids_n]) + ', '
                        caption_re_id = caption.replace('person', 'person' + ids_phrase)
                        llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', 'person'  +ids_phrase)
                    else:
                        print('Warning Multiple Ids were found but were not associated !!!!')

                elif len(ids_n) == 1:
                    ids = id_rec['re-id'][0]
                    if 'woman' in caption:
                        if 'a woman' in caption :                    
                            caption_re_id = caption.lower().replace('a woman', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a woman', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.lower().replace('woman', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('woman', woman_names[ids['id']])
                    elif 'lady' in caption:
                        if 'a lady' in caption:
                            caption_re_id = caption.lower().replace('a lady', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a lady', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.lower().replace('lady', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('lady', woman_names[ids['id']])
                    elif 'girl' in caption:
                        if 'a girl' in caption:
                            caption_re_id = caption.lower().replace('a girl', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a girl', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.replace('girl', woman_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('girl', woman_names[ids['id']])
                    elif 'man' in caption:
                        if 'a man' in caption:
                            caption_re_id = caption.lower().replace('a man', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.replace('man', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', woman_names[ids['id']])
                    elif 'boy' in caption:
                        if 'a boy' in caption:
                            caption_re_id = caption.lower().replace('a boy', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a boy', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.replace('boy', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('boy', woman_names[ids['id']])
                    elif 'person' in caption:
                        if 'a person' in caption:
                            caption_re_id = caption.lower().replace('a person', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.replace('person', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', woman_names[ids['id']])
                    elif 'person' in caption:
                        if 'a person' in caption:
                            caption_re_id = caption.lower().replace('a person', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.replace('person', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', woman_names[ids['id']])
                    elif 'human' in caption:
                        if 'a human' in caption:
                            caption_re_id = caption.lower().replace('a human', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a human', woman_names[ids['id']])
                        else:
                            caption_re_id = caption.lower().replace('human', man_names[ids['id']])
                            llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('human', woman_names[ids['id']])
                    else:
                        print('Warning Id wsa found but was not associated n IDS {} !!!! Caption :{} '.format(len(ids_n), caption))

"""