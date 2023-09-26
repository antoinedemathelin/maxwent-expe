import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Sequential

from PIL import Image

path_to_array = "./datasets/citycam"
# C:/Users/antoi/Documents/PHD/Code/experiments/datasets/citycam_images
#WEBCAMS = ['398', 'bigbus', '164', '166', '170', '173', '181', '253', 
           
WEBCAMS = ['403', '410', '495', '511', '551', '572', '691',
           '846', '928']

if __name__ == "__main__":
    try:
        path_to_images = sys.argv[1]
    except:
        raise Exception("Please provide the path to the CityCam dataset as argument")
    os.makedirs(path_to_array, exist_ok=True)
    resnet50 = tf.keras.applications.ResNet50(include_top=False, pooling="avg")
    error_raised = []

    for WEBCAM in WEBCAMS:
        if (len(WEBCAM) == 3 and os.path.isdir(os.path.join(path_to_images, WEBCAM))) or WEBCAM=="bigbus":
            for r, d, f in os.walk(os.path.join(path_to_images, WEBCAM)):
                for direct in d:
                    if not "checkpoint" in direct and not "cache" in d:

                        HOUR = direct.split("-")[-1]
                        DAY = direct.split("-")[-2]

                        print(direct)
                        print(WEBCAM, DAY, HOUR)

                        if "big_bus" in direct:
                            save_path = os.path.join(path_to_array, "bigbus", direct)
                        else:
                            save_path = os.path.join(path_to_array,  WEBCAM, direct)

                        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

                        path = os.path.join(path_to_images, WEBCAM, direct)                      

                        X = []
                        y = []
                        datetime = []
                        wheather = []
                        flow = []

                        i = 1

                        for r, d, f in os.walk(path):
                            for filename in f:
                                if len(filename) == len("000001.jpg") and ".jpg" in filename:
                                    img_name = filename
                                    path_to_image = os.path.join(path, img_name)

                                    image = Image.open(path_to_image)
                                    croped_image_resize = image.resize((224, 224), Image.LANCZOS)

                                    final_image = np.array(croped_image_resize)

                                    #output = resnet50.predict(preprocess_input(np.array([final_image]))).ravel()

                                    count = 0

                                    xml_name = filename.replace(".jpg", ".xml")
                                    label_path = os.path.join(path, xml_name)

                                    file = open(label_path)
                                    for line in file:
                                        if "<vehicle>" in line:
                                            count += 1
                                        if "<time>" in line:
                                            time = line.split("<time>")[-1].split("</time>")[0]
                                        if "<weather>" in line:
                                            wheather_ = line.split("<weather>")[-1].split("</weather>")[0]
                                        if "<flow>" in line:
                                            flow_ = line.split("<flow>")[-1].split("</flow>")[0]

                                    X.append(final_image)
                                    y.append(count)
                                    datetime.append(time)
                                    wheather.append(wheather_)
                                    flow.append(flow_)

                                    print("image %i  -  Count: %i  -  Date: %s  -  Wheather: %s  -  flow: %s"%
                                          (i, count, time, wheather_, flow_))

                                    i += 1

                        # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                        # ax.imshow(final_image)
                        # plt.show()
                        
                        X = np.stack(X, axis=0)
                        X = resnet50.predict(preprocess_input(X), batch_size=64, verbose=1)

                        X = np.array(X)
                        y = np.array(y)
                        datetime = np.array(datetime)
                        wheather = np.array(wheather)
                        flow = np.array(flow)

                        try:
                            np.save(os.path.join(save_path, "X.npy"), X)
                            np.save(os.path.join(save_path, "y.npy"), y)
                            np.save(os.path.join(save_path, "time.npy"), datetime)
                            np.save(os.path.join(save_path, "wheather.npy"), wheather)
                            np.save(os.path.join(save_path, "flow.npy"), flow)
                        except:
                            raise ValueError("No save path")