import os
import sys
import pandas as pd
import time
from request_for_mdf_summary import SummarizeScene

sys.path.insert(0, "/notebooks/fast_demo/")
sys.path.insert(0, "/notebooks/fast_demo/vidarts_advanced_main/")

from vidarts_advanced_main.microservices.blip2.blip2_service import BLIP2Service


def callback_caption_extract(url):
    # # Inputs
    texts = [""]
    # bboxes = [[53.04999923706055, 199.6999969482422, 127.80999755859375, 396.29998779296875], [100.77999877929688, 209.41000366210938, 170.8800048828125, 380.7200012207031], [632.1799926757812, 124.19999694824219, 1019.760009765625, 743.0499877929688], [1599.1800537109375, 372.8900146484375, 1624.800048828125, 401.55999755859375], [1780.93994140625, 332.8999938964844, 1864.1199951171875, 438.6600036621094], [1406.1400146484375, 331.239990234375, 1527.050048828125, 444.1300048828125], [1610.6300048828125, 349.67999267578125, 1672.9599609375, 440.4100036621094], [1542.489990234375, 348.260009765625, 1655.1400146484375, 442.3699951171875], [744.3300170898438, 520.1500244140625, 843.3699951171875, 748.6900024414062], [996.4600219726562, 220.19000244140625, 1397.4300537109375, 940.010009765625], [979.1500244140625, 620.7100219726562, 1524.0, 948.0], [18.440000534057617, 727.5999755859375, 956.8499755859375, 948.969970703125], [928.1699829101562, 759.6500244140625, 1581.5799560546875, 945.5599975585938]]
    outputs = blip2_service.get_url_response(url, texts)[0]

    return outputs

summarize_scene = SummarizeScene(caption_callback=callback_caption_extract)

unique_run_name = str(int(time.time()))
result_path = "/notebooks/nebula3_playground"
csv_file_name = 'scene_summarization_' + str(unique_run_name) + '_' + str(summarize_scene.prompting_type) + '_' + str(summarize_scene.gpt_type) +'_blip2.csv'

add_action = True

results = list()
all_movie_id = list()
all_movie_id.append('Movies/-3323239468660533929') #actionclipautoautotrain00616.mp4
all_movie_id.append('Movies/-6372550222147686303')
if add_action:
    all_movie_id.append('Movies/7023181708619934815')
all_movie_id.append('Movies/889658032723458366')
all_movie_id.append('Movies/-6372550222147686303')
all_movie_id.append('Movies/-6576299517238034659')
all_movie_id.append('Movies/-5723319113316714990')
all_movie_id.append('Movies/2219594956981209558')
all_movie_id.append('Movies/6293447408186786707')

for movie_id in all_movie_id:
    frame_boundary = []
    if movie_id == 'Movies/-6372550222147686303':
        frame_boundary = [834, 1181]
    if movie_id == 'Movies/-5723319113316714990':
        frame_boundary = [197, 320]
    if movie_id == 'Movies/6293447408186786707':
        frame_boundary = [1035, 1290]

    blip2_service = BLIP2Service("http://209.51.170.37:8087/infer")



    scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary)
    # scn_summ = summarize_scene.summarize_scene_forward(movie_id) # for all clip w/o frame boundaries 
    print("Movie: {} Scene summary : {}".format(movie_id, scn_summ))

    results.append({'movie_id':movie_id, 'summary': scn_summ, 'movie_name':summarize_scene.movie_name, 'prompt_prefix_caption' : summarize_scene.prompt_prefix_caption})
df = pd.DataFrame(results)
df.to_csv(os.path.join(result_path, csv_file_name), index=False)
