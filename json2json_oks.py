import json
import functools
import os

check_add_one = lambda arr:functools.reduce(lambda x,y:(x+1==y if isinstance(x,int) else x[0] and x[1]+1==y, y),arr)[0]

def GetNumberOfK(data, k):
# write code here
	count = 0
	for i in data:
		if i == k:
			count += 1
	return count


path1 = './pose_choose/'
path2 = sorted(os.listdir(path1))
label_inform = {}

for g in range(len(path2)):
	path = path1 + path2[g]
	f = open(path,encoding='utf-8')
	points = json.load(f)

	n=4  #four frames
	#568

	image_id = []
	list_start = []

	for m in range(len(points)):
		image_id.append(points[m]['image_id'])

	list1 = list(set(image_id))

	for k in range(len(list1)-n):
		list2 = []
		for m in range(n+1):
			list2.append(list1[k+m])
		if check_add_one(list2):
			list_start.append(list2)


	list_frame = []


	for m in range(len(list_start)):
		list3 = list_start[m]
		list4 = []
		for k in range(len(points)):
			for p in range(n+1):
				if points[k]['image_id'] == list3[p]:
					list4.append(points[k]['track_id'])
		list5 = list(set(list4))
		for x in list5:
			num = GetNumberOfK(list4,x)
			if num == (n+1):
				list_frame.append(list_start[m])
				list_frame.append(x)


	path_final = './label/'+ 'label' +'.json'
	fl = open(path_final,'a')

	k = int(0.5 * len(list_frame))
	print(k)



	for m in range(k):

		frame = list_frame[2*m]
		trackid = list_frame[2*m+1]

		text = {}
		text["data"] = []
		text["frame_id"] = frame
		text["track_id"] = trackid
		text["video_id"] = path2[g][0:-5]

		path_data = './jsondata/'+ path2[g][0:-5] + '_' +'%04d'%(m)+'.json'
		f=open(path_data,'a')



		for x in range(len(points)):
			for p in range(n):
				if points[x]['image_id'] == frame[p]:
					if points[x]['track_id'] == trackid:
						keypoints = points[x]["keypoints"]
						pose = []
						score = []
						for q in range(17):
							pose.append(keypoints[3*q])
							pose.append(keypoints[3*q+1])
							score.append(keypoints[3*q+2])
						skeleton = {}
						skeleton["pose"] = pose
						skeleton["score"] = score
						data = {}
						data["frame"] = frame[p]
						data["frame_index"] = p
						data["skeleton"] = []
						data["skeleton"].append(skeleton)
						text["data"].append(data)

				
			if points[x]['image_id'] == frame[n]:
				if points[x]['track_id'] == trackid:
					label_data = {}
					label_data["has_skeleton"] = True
					label_data["label"] = frame[n]
					label_data["video"] = path2[g][0:-5]
					keypoints = points[x]["keypoints"]
					w = points[x]["w"]
					h = points[x]["h"]
					s = int(points[x]["s_area"])
					x0 = points[x]["point_x"]
					y0 = points[x]["point_y"]
					pose1 = [] 
					for q in range(17):
						pose1.append(keypoints[3*q])
						pose1.append(keypoints[3*q+1])
					flag = []
					for v in range(17):
						if keypoints[3*v+2] > 0.6:
							flag.append(1)
						else:
							flag.append(0)

					label_data["label_index"] = pose1
					label_data["flag"] = flag
					label_data["center"] = points[x]["center"]
					label_data["w"] = w
					label_data["h"] = h
					label_data["s"] = s
					label_data["point_x"] = x0
					label_data["point_y"] = y0
					name = path2[g][0:-5] + '_' +'%04d'%(m)
					label_inform[name] = label_data


		jsondata = json.dumps(text)
		f.write(jsondata)
		f.close()

labeldata = json.dumps(label_inform,indent=4)
fl.write(labeldata)

fl.close()
