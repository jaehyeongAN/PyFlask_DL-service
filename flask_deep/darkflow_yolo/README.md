# Requirements

Python 3.5, CUDA 9.0, cudnn 7.3.1 and other common packages listed in `requirements.txt`.


# Installation
### 1. Clone this repository

### 2. Install dependencies 
``` pip install -r requirements.txt```

### 3. Run setup from the repository root directory
``` python setup.py build_ext --inplace ```

### 4. Download yolo.weights
download data from https://pjreddie.com/media/files/yolov2.weights

### 5. Make directory
Make **/bin** directory from root and put the weights file into the bin.
Please rename **yolov2.weights** to **yolo.weights**.

# Sample test
### Run  Processing_Images.py
~~~
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

# define the model options and run

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

# read the color image and covert to RGB

img = cv2.imread('dog.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)

img.shape

# pull out some info from the results

tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
label = result[0]['label']


# add the box and label and display it
img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
plt.imshow(img)
plt.show()
~~~

# Result
![dog.png](https://postfiles.pstatic.net/MjAxOTAxMjJfMjgz/MDAxNTQ4MTM3Nzc2MzE1.qmu5CMeI62QOT4isoBknFmZBMC9JO9fqoEMla8_MiS4g.8wCu0dY6BM0iv6iDL3H3VkRBbjz_FSNT8iCRQ1wVl7Ag.PNG.nicetiger516/%EC%9D%B4%EB%AF%B8%EC%A7%80_1.png?type=w773)
