# export PYTHONPATH=/notebooks/pip_install/

import os
from database.arangodb import NEBULA_DB
from typing import NamedTuple
from nebula3_experiments.prompts_utils import *
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
FS_SAMPLES = 5                   # Samples for few-shot gpt

from abc import ABC, abstractmethod
import openai
class LLMBase(ABC):
    @abstractmethod
    def completion(prompt_template: str, *args, n=1, **kwargs):
        pass
# pip install openai==0.27.0 --target /notebooks/pip_install/
FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'

import importlib
print(importlib.metadata.version('openai'))
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

all_movie_id = list()
all_movie_id.append('Movies/7417592353856606351')
all_movie_id.append('Movies/889658032723458366')
all_movie_id.append('Movies/889658032723458366')

man_names = ['James', 'Michael', 'Tom', 'George' ,'Nicolas', 'John']
woman_names = ['Susan', 'Jennifer', 'Eileen', 'Sandra', 'Emma']

places = 'indoor'
reid = True
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
    rc = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION)
    rc_reid = nebula_db.get_doc_by_key({'movie_id': movie_id}, REID_CLUES_COLLECTION)
    # rc['mdfs']
    # frames_num_dict = dict(zip(flatten(rc['mdfs']),rc['mdfs_path']))

    for ix, frame_num in enumerate(flatten(rc['mdfs'])):
        mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
        obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
        caption = obj['global_caption']['blip']
        scene = obj['global_scenes']['blip'][0][0]
        all_global_tokens.extend([x[0] for x in obj['global_objects']['blip']])
        obj_LLM_OUTPUT_COLLECTION_cand = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)['candidate']
        all_obj_LLM_OUTPUT_COLLECTION_cand.append(obj_LLM_OUTPUT_COLLECTION_cand)
        # mdf_re_id_dict = rc_reid['frames'][ix]
        mdf_re_id_dict = [x  for x in rc_reid['frames'] if x['frame_num']==frame_num]
        if len(mdf_re_id_dict) >1:
            print('ka')
        if mdf_re_id_dict and places == 'indoor':  # conditioned on man in the scene if places==indoor
            assert(mdf_re_id_dict[0]['frame_num'] == frame_num)
            for id_rec in mdf_re_id_dict: # match many2many girl lady, woman to IDs at first
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
                elif 'someone' in caption:
                    if 'a someone' in caption:
                        caption_re_id = caption.lower().replace('a someone', man_names[ids['id']])
                        llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a someone', woman_names[ids['id']])
                    else:
                        caption_re_id = caption.lower().replace('someone', man_names[ids['id']])
                        llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('someone', woman_names[ids['id']])
                else:
                    print('Warning Id wsa found but was not associated !!!!')
                
                    
            all_reid_caption.append(caption_re_id)
            all_caption.append(caption)
            all_obj_LLM_OUTPUT_COLLECTION_cand_re_id.append(llm_out_cand_re_id)
        
    # obj = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)
        # get_dialog_caption(mid.movie_id,mid.frame_num)
    if reid :
        seq_caption = ' then '.join(all_reid_caption)
    else:
        seq_caption = ' then '.join(all_caption)

    if 0:
        prompt = "Summarize the following video transcription of a scene given segmented captions seperated by the word 'then':{} Summary :".format(seq_caption)
    elif 0:
        prompt = "Give a concise summary of the following video transcription of a scene given segmented captions seperated by the word 'then':{} Summary :".format(seq_caption)
    elif 0:
        prompt = "What is the theme of the following video scene given captions seperated by the word 'then':{} Summary :".format(seq_caption)
        prompt = "Summarize the captions out of a video scene seperated by the word 'then':{} Summary :".format(seq_caption)
    elif 1:
        prompt = "Summarize the video scene by the shot captions seperated by the word 'then', the scene is at the {} :{} Summary :".format(scene, seq_caption)
        prompt = "Summarize the video shots taken at the {} seperated by the word 'then' :{} Summary :".format(scene, seq_caption)
        prompt = "Summarize the video that was taken at the {} by 2-3 sentences. The video shots are seperated by the word 'then' :{} Summary :".format(scene, seq_caption)

    else:
        prompt = "Give a concise summary of the following video scene captions seperated by the word 'then':{} Summary :".format(seq_caption)

    if len(prompt) >4096-120:
        print('Prompt too long!!!')
    # concise 
    if gpt_type == 'text-davinci-003':
        rc = gpt_execute(prompt, model='text-davinci-003', n=1, max_tokens=256)
    elif gpt_type == 'chat_gpt_3.5' or gpt_type == 'gpt-4':
        chatgpt = ChatGptLLM()
        chatgpt.completion(prompt, n=1, max_tokens=256)
    # chatgpt install
        # from chatgpt_wrapper import ChatGPT
        # bot = ChatGPT()
        # response = bot.ask(prompt)
        # print(response)  # prints the response from chatGPT
    pass

"""
FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
'gpt-4-32k'
'gpt-4-32k-0314'
'gpt-4'

"""