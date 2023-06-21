from request_for_mdf_summary import SummarizeScene

summarize_scene = SummarizeScene()

add_action = True

results = list()
all_movie_id = list()
if add_action:
    all_movie_id.append('Movies/7023181708619934815')
all_movie_id.append('Movies/-6372550222147686303')
all_movie_id.append('Movies/-3323239468660533929') #actionclipautoautotrain00616.mp4
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
        # mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]
    if movie_id == 'Movies/-5723319113316714990':
        frame_boundary = [197, 320]
        # mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]
    if movie_id == 'Movies/6293447408186786707':
        frame_boundary = [1035, 1290]
        # mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]

    scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary)
    print("Movie: {} Scene summary : {}".format(movie_id, scn_summ))
