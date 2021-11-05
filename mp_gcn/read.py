import numpy as np
import pickle as pkl

#a=np.load('/tangyao/test_frame_mask/01_001.npy')
#print(a)

result = pkl.load(open("test_result.pkl", 'rb'), encoding='utf-8')
print(result['1_0287.json'])