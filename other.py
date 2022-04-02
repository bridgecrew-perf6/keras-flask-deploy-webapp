from __future__ import division, print_function
import json
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import pandas as pd
import numpy as np
import biosppy
import matplotlib.pyplot as plt
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
output = []
def model_predict(uploaded_files, model):
    flag = 1
    json_filename= 'data.txt'
    print(uploaded_files)
    if True:
        path=uploaded_files
    # for path in uploaded_files:
        print(path)
        #index1 = str(path).find('sig-2') + 6
        #index2 = -4
        #ts = int(str(path)[index1:index2])
        APC, NORMAL, LBB, PVC, PAB, RBB, VEB = [], [], [], [], [], [], []
        output.append(str(path))
        result = {"APC": APC, "Normal": NORMAL, "LBB": LBB, "PAB": PAB, "PVC": PVC, "RBB": RBB, "VEB": VEB}
        print(output)
        
        indices = []
        
        kernel = np.ones((4,4),np.uint8)
        
        csv = pd.read_csv(path)
        print(csv)
        csv_data = csv[' Sample Value']
        data = np.array(csv_data)
        signals = []
        count = 1
        peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
        for i in (peaks[1:-1]):
           diff1 = abs(peaks[count - 1] - i)
           diff2 = abs(peaks[count + 1]- i)
           x = peaks[count - 1] + diff1//2
           y = peaks[count + 1] - diff2//2
           signal = data[x:y]
           signals.append(signal)
           count += 1
           indices.append((x,y))

        # print(signals)
        for count, i in enumerate(signals):
            fig = plt.figure(frameon=False)
            plt.plot(i) 
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            filename = 'fig' + '.png'
            fig.savefig(filename)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.erode(im_gray,kernel,iterations = 1)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
            im_gray = cv2.imread(filename)
            pred = model.predict(im_gray.reshape((1, 128, 128, 3)))
            pred_class = pred.argmax(axis=-1)
            if pred_class == 0:
                APC.append(indices[count]) 
            elif pred_class == 1:
                NORMAL.append(indices[count]) 
            elif pred_class == 2:    
                LBB.append(indices[count])
            elif pred_class == 3:
                PAB.append(indices[count])
            elif pred_class == 4:
                PVC.append(indices[count])
            elif pred_class == 5:
                RBB.append(indices[count]) 
            elif pred_class == 6:
                VEB.append(indices[count])
        


        result = sorted(result.items(), key = lambda y: len(y[1]))[::-1]   
        output.append(result)
        data = {}
        data['filename'+ str(flag)] = str(path)
        data['result'+str(flag)] = str(result)

        json_filename = 'data.txt'
        with open(json_filename, 'a+') as outfile:
            json.dump(data, outfile) 
        flag+=1 
    


    
    with open(json_filename, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('}{', ',')
    with open(json_filename, 'w') as file:
        file.write(filedata) 
    os.remove('fig.png')      
    return output
    
model = load_model('./ecgScratchEpoch2.hdf5')
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')
pred = model_predict("sample.csv", model)


# Process your result for human
            # Simple argmax
#pred_class = decode_predictions(pred, top=1)   # ImageNet Decode
#result = str(pred_class[0][0][1])               # Convert to string
result = str(pred)


print(result)




