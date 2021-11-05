from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pkl
import json
import os
from sklearn.metrics import roc_auc_score


result = pkl.load(open("test_result.pkl", 'rb'), encoding='utf-8')
label = json.load(open('label.json','r'))


annotations = []    # 异常事件标注为0
frames = 0
for video_name in sorted(os.listdir("./mask")):
	path = "./mask/" + video_name
	anno = np.load(path)
	for p in anno:
		annotations.append(1-p)

	frames = frames + len(anno)
	
score = [1 for index in range(frames)]  # 全1列表



ious = np.zeros((len(label), 1),dtype=np.float32)
num_points = np.zeros((len(label), 1))
sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
vars = (sigmas * 2)**2


names = list(label.keys())   # 12_0175_0579
j = 0


for name in names: # 0
	name_json = name + ".json"
	gt = label[name]["label_index"]
	dt = result[name_json]
	xg = []
	yg = []
	xd = []
	yd = []
	vg = label[name]["flag"]
	s = label[name]["s"]
	k = np.count_nonzero(vg)
	for i in range(17):
		xg.append(gt[2*i])
		yg.append(gt[2*i+1])
		xd.append(dt[2*i])
		yd.append(dt[2*i+1])

	if k > 0:
		dx = np.subtract(xd,xg)
		dy = np.subtract(yd,yg)
		e = (dx**2 + dy**2) / vars / (s+np.spacing(1)) / 2
		e=e[np.nonzero(vg)]
		ious[j] = np.sum(np.exp(-e)) / e.shape[0]

	num_points[j] = k
	j = j+1

np.savetxt("1.txt",ious)


d = 0
L = 0

for video_name_1 in sorted(os.listdir("./mask")):
	path_1 = "./mask/" + video_name_1   # 01_0014.npy
	anno_1 = np.load(path_1)
	frames_1 = len(anno_1)
	f = 0
	for w in range(len(names)):
		if names[w][0:-5] == video_name_1[0:-4]:
			f = f+1

	print(f)    # 同一视频的label数量
	for num in range(frames_1):   # 一个视频的帧数
		list_frame = []
		for m in range(f):
			if (label[video_name_1[0:-4]+'_'+'%04d'%(m)]["label"] == num) and (label[video_name_1[0:-4]+'_'+'%04d'%(m)]["video"] == str(video_name_1[0:-4])):
				list_frame.append(m)

		if len(list_frame):
			order = []
			for n in list_frame:
				order.append(ious[n+d])

			min_score = min(order)

			score[num + L] = min_score

	d = d+f
	L = L + frames_1



np.savetxt("anno.txt",annotations)
np.savetxt("score.txt",score)
auc = roc_auc_score(annotations, score)
print(auc)





