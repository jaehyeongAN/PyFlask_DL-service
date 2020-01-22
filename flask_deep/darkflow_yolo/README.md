# Requirements
Python 3.6, CUDA 10.0, cudnn 10.0 and other common packages listed in `requirements.txt`.

# Installation
### 1. Install dependencies 
``` pip install -r requirements.txt```

### 2. Run setup from the repository root directory
``` python setup.py build_ext --inplace ```
``` pip install . ```

### 3. Download yolo.weights
download data from https://pjreddie.com/media/files/yolov2.weights

### 4. Make directory
Make **/bin** directory from root and put the weights file into the bin.
Please rename **yolov2.weights** to **yolo.weights**.