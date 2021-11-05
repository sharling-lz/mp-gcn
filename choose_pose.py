import os 
import json

path = './pose/'
path_j = './pose_choose/'
data_name = os.listdir(path)

data_name.sort(key=lambda x:int(x[0:-5]))

for file in data_name:
	file_path = path + file
	result_path = path_j + file
	#print(file_path)
	f = open(file_path,encoding='utf-8')
	fj = open(result_path,'a')
	pose = json.load(f)
	select_data = []
	k = 0
	for n in range(len(pose)):
		if pose[n]['score'] >= 0.6:
			select_data.append(pose[n])
		else:
			k = k+1
	print(k)
	jsondata = json.dumps(select_data,indent=4)
	fj.write(jsondata)
	f.close()
	fj.close()