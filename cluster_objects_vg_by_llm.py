import json
path = '/notebooks/multi_modal/uniq_obj_vg.json'
with open(path, 'r') as f:
    count_obj_freq = json.load(f)
min_freq = 20
obj_freq_filtered = dict()
for k,v in count_obj_freq.items():
    if v > min_freq:
        obj_freq_filtered.update({k:v})


print("objects with more than {} Attributes {}".format(min_freq, len(obj_freq_filtered)))



