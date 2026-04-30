import os
import numpy as np
from PIL import Image

def load_eurosat(data_dir):
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: i for i,name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir,class_name)

        for file in os.listdir(class_path):
            if file.endswith(".jpg"):
                img_path = os.path.join(class_path,file)
                img = Image.open(img_path).convert("RGB")
                img = np.array(img)
                img = img.reshape(-1)

                X.append(img)
                y.append(label_map[class_name])

    X = np.array(X,dtype=np.float32)
    y = np.array(y)
    X = X / 255.0

    return X, y
