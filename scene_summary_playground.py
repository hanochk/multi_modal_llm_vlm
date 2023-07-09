import os
import sys
import pandas as pd
import time
from request_for_mdf_summary import SummarizeScene


# summarize_scene = SummarizeScene(gpt_type='gpt-4')
summarize_scene = SummarizeScene()

unique_run_name = str(int(time.time()))
result_path = "/notebooks/nebula3_playground"
csv_file_name = 'scene_summarization_' + str(unique_run_name) + '_' + str(summarize_scene.prompting_type) + '_' + str(summarize_scene.gpt_type) + '.csv'
add_action = False

results = list()
all_movie_id = list()
all_movie_id.append('Movies/-6576299517238034659') #Two men are sitting in a front-row of a driving car while 
all_movie_id.append('Movies/-7594388714349439611') # Kevin spacy with daugther photographed by her friend unrecognized

all_movie_id.append('Movies/889658032723458366') #Jensen Ackles, Jensen Atwood, and Naveen Andrews. They are seen sitting around a dining table, enjoying a meal together.
all_movie_id.append('Movies/-6372550222147686303')

# all_movie_id.append('Movies/7891527252242080923') # image just check robustness

all_movie_id.append('Movies/-3323239468660533929') #actionclipautoautotrain00616.mp4 man walking downa the strret passong newspaper stand and go to booth
if add_action:
    all_movie_id.append('Movies/7023181708619934815')


all_movie_id.append('Movies/-5723319113316714990')
all_movie_id.append('Movies/2219594956981209558')
all_movie_id.append('Movies/6293447408186786707')

for movie_id in all_movie_id:
    frame_boundary = []

    if movie_id == 'Movies/-7594388714349439611':
        frame_boundary = [[89, 1721] ,[1880, 2650]] #[[89, 1721]]#[[1880, 2650]] #[[89, 1721] ,[1880, 2650]]                  #[[89, 1721]] Kevin spacy  [1880, 2650] # his wife selling houses
    if movie_id == 'Movies/-6372550222147686303':
        frame_boundary = [[834, 1181]]
    if movie_id == 'Movies/-5723319113316714990':
        frame_boundary = [[197, 320]]
    if movie_id == 'Movies/6293447408186786707':
        frame_boundary = [[1035, 1290]]
    if movie_id == 'Movies/-3323239468660533929':
        frame_boundary = [[195, 1446]]




    # scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary, caption_type='blip2')
    scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary, caption_type='dense_caption')
    # scn_summ = summarize_scene.summarize_scene_forward(movie_id) # for all clip w/o frame boundaries 
    print("Movie: {} Scene summary : {}".format(movie_id, scn_summ))
    if scn_summ != -1:
        results.append({'movie_id':movie_id, 'summary': scn_summ, 'movie_name':summarize_scene.movie_name, 
                        'prompt_prefix_caption' : summarize_scene.prompt_prefix_caption, 
                        "scene_top_k": summarize_scene.scene_top_k_frequent,
                        "frame_boundary": frame_boundary})
df = pd.DataFrame(results)
df.to_csv(os.path.join(result_path, csv_file_name), index=False)
print("SAving to file : ", csv_file_name)

