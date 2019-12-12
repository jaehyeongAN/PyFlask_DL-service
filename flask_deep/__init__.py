import os, sys
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from werkzeug import secure_filename
from . import neural_style_transfer
from .darkflow_yolo import processing_images

app = Flask(__name__)
app.debug = True

@app.route('/info')
def info():
	return render_template('info.html')

# Fisrt page
@app.route('/')
def index():
	return render_template('index.html')

''' Neural Style Transfer '''
# user로부터 reference img와 target img를 선택받는 화면
@app.route('/get_image')
def get_nst_img():
	return render_template('get_image.html')

# NST 결과를 출력하는 화면 
@app.route('/post_image', methods=['GET','POST'])
def post_nst_img():
	if request.method == 'POST':
		# Reference Image
		refer_img = request.form['refer_img']
		refer_img_path = 'images/'+str(refer_img)

		# User Image (target image)
		user_img = request.files['user_img']
		user_img.save('./flask_deep/static/images/'+str(user_img.filename))
		user_img_path = 'images/'+str(user_img.filename)

		# Neural Style Transfer 
		transfer_img = neural_style_transfer.main(refer_img_path, user_img_path)
		transfer_img_path = 'images/'+str(transfer_img.split('/')[-1])

	return render_template('post_image.html', 
					refer_img=refer_img_path, user_img=user_img_path, transfer_img=transfer_img_path)


''' Obejct Detection '''
@app.route('/object_detection')
def object_detection():
	return render_template('object_detection.html')

@app.route('/post_object_detection', methods=['GET','POST'])
def post_object_detection():
	if request.method == 'POST':
		# User Image
		user_img = request.files['object_img']
		user_img.save('./flask_deep/static/images/'+str(user_img.filename))
		user_img_path = 'images/'+str(user_img.filename)

		# Object Detection by darkflow with YOLO V2
		detected_img = processing_images.main(user_img_path)
		detected_img_path = 'images/'+str(detected_img.split('/')[-1])


	return render_template('post_object_detection.html', detected_img=detected_img_path)