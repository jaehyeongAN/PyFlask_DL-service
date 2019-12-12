import os, sys
# os.chdir('./darkflow_yolo')
sys.path.append('C:/Users/USER/Google_drive/pydev/workspace/pyflask/flask_deep/darkflow_yolo')
print(os.getcwd())

import numpy as np
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import save_img

def boxing(original_img, predictions):
	newImage = np.copy(original_img)

	# pull out some info from the results
	for result in predictions:
		tl = (result['topleft']['x'], result['topleft']['y'])			# topleft
		br = (result['bottomright']['x'], result['bottomright']['y'])	# bottomright
		
		confidence = result['confidence']
		label = result['label']+" "+str(round(confidence, 3))

		# add the box and label and display it
		if confidence > 0.3:
			newImage = cv2.rectangle(newImage, tl, br, (0, 255, 0), 3)
			newImage = cv2.putText(newImage, label, tl, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 250, 0), 1, cv2.LINE_AA)

	return newImage

def main(user_img_path):
	print('-------------------------------------')
	print(user_img_path)
	user_img_name = user_img_path.split('/')[-1].split('.')[0]
	print('-------------------------------------')

	# define the model options and run
	options = {
		'model': '.darkflow_yolo/cfg/yolo.cfg',
		'load': '.darkflow_yolo/bin/yolo.weights',
		'threshold': 0.3,
		'gpu': 0.5
	}

	tfnet = TFNet(options)

	# read the color image and covert to RGB
	img = cv2.imread(user_img_path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# use YOLO to predict the image
	result = tfnet.return_predict(img)
	detected_img = boxing(img, result)

	# save
	detected_img_path = 'flask_deep/static/images/'+str(user_img_name)+'_detected.png'
	save_img(detected_img_path, detected_img)
	# plt.imshow(detected_img)
	# plt.show()

	return detected_img_path

if __name__ == "__main__":
	main()