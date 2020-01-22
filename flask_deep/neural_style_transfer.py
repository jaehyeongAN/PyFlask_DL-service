'''
neural style transfer
1. 스타일 참조 이미지, 타깃 이미지, 생성된 이미지를 위해 VGG19의 층 활성화를 동시에 계산한느 네트워크를 설정
2. 세 이미지에서 계산한 층 활성화를 사용하여 콘텐츠 손실 및 스타일 손실 함수를 정의. 이 손실을 최소화하여 구현
3. 손실 함수를 최소화할 경사 하강법 과정을 설정
'''
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import time
import os

print(os.getcwd())

def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_height, img_width)) # (400, 381)
	'''The img_to_array() function adds channels: x.shape = (224, 224, 3) for RGB and (224, 224, 1) for gray image'''
	img = img_to_array(img) 			# (400, 381, 3)
	'''expand_dims() is used to add the number of images: x.shape = (1, 224, 224, 3)'''
	img = np.expand_dims(img, axis=0) 	# (1, 400, 381, 3)
	'''preprocess_input() subtracts the mean RGB channels of the imagenet dataset. 
	This is because the model you are using has been trained on a different dataset: x.shape is still (1, 224, 224, 3)'''
	img = vgg19.preprocess_input(img)

	return img 

def deprocess_image(x):
	'''vgg19.preprocess_input()에서 일어난 변환을 복원'''
	# imagenet의 평균 pixel 값을 더함 
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	# BGR -> RGB( )
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')

	return x 

def content_loss(base, combination):
	'''콘텐츠 손실'''
	return K.sum(K.square(combination - base))

def gram_matrix(x):
	'''feature map을 vector로 펼침'''
	features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
	gram = K.dot(features, K.transpose(features))

	return gram

def style_loss(style, combination):
	'''스타일 손실'''
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_height * img_width

	return K.sum(K.square(S-C))/(4.*(channels**2)*(size**2))

def total_variation_loss(x):
	'''생성된 이미지의 픽셀을 사용하여 계산하는 총 변위 손실'''
	a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
	b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

class Evaluator(object):

	def __init__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		x = x.reshape((1, img_height, img_width, 3))
		outs = fetch_loss_and_grads([x])
		loss_value = outs[0]
		grad_values = outs[1].flatten().astype('float64')
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


def main(refer_img_path, target_img_path):
	# image 파일명
	refer_img_path = refer_img_path.split('/')[-1]
	target_img_path = target_img_path.split('/')[-1]
	# image 경로
	style_reference_image_path = 'flask_deep/static/images/nst_get/'+ refer_img_path	# 스타일 참조 이미지 
	target_image_path = 'flask_deep/static/images/'+ target_img_path 			# 타깃 이미지 

	# 모든 이미지를 fixed-size(400pixel)로 변경
	width, height = load_img(target_image_path).size
	global img_height; global img_width;
	img_height = 400
	img_width = int(width * img_height / height)

	target_image = K.constant(preprocess_image(target_image_path)) # creates img to a constant tensor
	style_reference_image = K.constant(preprocess_image(style_reference_image_path))
	combination_image = K.placeholder((1, img_height, img_width, 3)) # 생성된 이미지를 담을 placeholder

	# 3개의 이미지를 하나의 배치로 합침
	input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

	# 3개 이미지의 배치를 입력으로 받는 VGGNet 생성
	model = vgg19.VGG19(input_tensor=input_tensor, 
						weights='imagenet', # pre-trained ImageNet 가중치 로드 
						include_top=False) # FC layer 제외 


	# 층 이름과 활성화 텐서를 매핑한 dict
	outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

	# content 손실에 사용할 layer
	content_layer = 'block5_conv2'
	# style 손실에 사용할 layer
	style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

	# 손실 항목의 가중치 평균에 사용할 가중치
	total_variation_weight = 1e-4
	style_weight = 1.
	content_weight = 0.025

	loss = K.variable(0.)
	layer_features = outputs_dict[content_layer]
	target_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]
	loss = loss + (content_weight * content_loss(target_image_features, combination_features))

	# 각 타깃 층에 대한 스타일 손실을 더함
	for layer_name in style_layers:
		layer_features = outputs_dict[layer_name]
		style_reference_features = layer_features[1, :, :, :]
		combination_features = layer_features[2, :, :, :]
		sl = style_loss(style_reference_features, combination_features)
		loss = loss + ((style_weight / len(style_layers)) * sl)

	# 총 변위 손실을 더함 
	loss = loss + (total_variation_weight * total_variation_loss(combination_image))

	# 손실에 대한 생성된 이미지의 그래디언트를 구합니다
	grads = K.gradients(loss, combination_image)[0]

	# 현재 손실과 그래디언트의 값을 추출하는 케라스 Function 객체입니다
	global fetch_loss_and_grads
	fetch_loss_and_grads = K.function([combination_image], [loss, grads])



	evaluator = Evaluator()
	refer_img_name = refer_img_path.split('.')[0].split('/')[-1]
	result_prefix = 'flask_deep/static/images/nst_result_'+refer_img_name
	iterations = 20

	# 뉴럴 스타일 트랜스퍼의 손실을 최소화하기 위해 생성된 이미지에 대해 L-BFGS 최적화를 수행합니다
	x = preprocess_image(target_image_path)	# 초기 값은 타깃 이미지입니다
	x = x.flatten()	# scipy.optimize.fmin_l_bfgs_b 함수가 벡터만 처리할 수 있기 때문에 이미지를 펼칩니다.
	for i in range(iterations):
		print('반복 횟수:', i)
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
		
		print('현재 손실 값:', min_val)
		# 생성된 현재 이미지를 저장합니다
		img = x.copy().reshape((img_height, img_width, 3))
		img = deprocess_image(img)
		fname = result_prefix + '.png'

	save_img(fname, img)

	return fname

if __name__ == "__main__":
	main()