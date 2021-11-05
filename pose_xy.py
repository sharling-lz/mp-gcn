import os 
import json

path = './pose/'
path_j = './pose_xy/'
data_name = os.listdir(path)

data_name.sort(key=lambda x:int(x[0:-5]))

for file in data_name:
    file_path = path + file
    result_path = path_j + file

    f = open(file_path,encoding='utf-8')
    fj = open(result_path,'a')
    pose = json.load(f)
    select_data = []
    for n in range(len(pose)):
        x = pose[n]["point_x"]
        y = pose[n]["point_y"]


        for m in range(17):
            pose[n]["keypoints"][3*m] = pose[n]["keypoints"][3*m] - x
            pose[n]["keypoints"][3*m+1] = pose[n]["keypoints"][3*m+1] - y
        select_data.append(pose[n])

    jsondata = json.dumps(select_data,indent=4)
    fj.write(jsondata)
    f.close()
    fj.close()