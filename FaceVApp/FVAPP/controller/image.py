from flask import Flask, flash, request, redirect, url_for, render_template, Blueprint
from FVAPP.models import Image, User
import os
from FVAPP.__init__ import db
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
# from random import choice
# from numpy import load
# from numpy import expand_dims
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from matplotlib import pyplot
# import mtcnn
# # function for face detection with mtcnn
# # from PIL import Image
# from numpy import asarray
# from mtcnn.mtcnn import MTCNN
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from os import listdir
# import os.path as path
# import numpy as np
# from keras.models import load_model

#model = load_model('facenet_keras.h5')
img = Blueprint('img', __name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'D:/Tai-Lieu-Hoc/TNCKH/Graduation_Thesis/FaceVApp/FVAPP/static/uploads/'
# img.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@img.route('/', methods=['POST', 'GET'])
@login_required
def upload_image():

    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    #file = request.files['files']
    files = request.files.getlist("files")
    # for file in files:
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    #
    if files.index == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            check_image = Image.query.filter_by(path=os.path.join(UPLOAD_FOLDER, filename)).first()
            if check_image:
                flash('This image already exists.', category='error')
            # print('upload_image filename: ' + filename)
            else:
                pic = Image(path=os.path.join(UPLOAD_FOLDER, filename), user_id=current_user.id)
                db.session.add(pic)
                db.session.commit()
                flash('Image successfully uploaded and displayed below', category='success')
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('home.html', filename=filename, user=current_user)

#
# def load_data(data_path):
#     # load faces
#     data = load(data_path)
#     testX_faces = data['arr_2']
#     testy_faces = data['arr_3']
#     # load face embeddings
#     data = load('5-celebrity-faces-embeddings.npz')
#     trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
#     return trainX, trainy, testX, testy
#
#
# def normalize_data(trainX, trainy, testX, testy):
#     # Normalize input vectors
#     in_encoder = Normalizer(norm='l2')
#     trainX = in_encoder.transform(trainX)
#     testX = in_encoder.transform(testX)
#
#     # Label encode outputs
#     out_encoder = LabelEncoder()
#     out_encoder.fit(trainy)
#     trainy = out_encoder.transform(trainy)
#     testy = out_encoder.transform(testy)
#     return trainX, trainy, testX, testy
#
#
# def extract_face(filename, required_size=(160, 160)):
# 	# load image from file
# 	image = Image.open(filename)
# 	# convert to RGB, if needed
# 	image = image.convert('RGB')
# 	# convert to array
# 	pixels = asarray(image)
# 	# create the detector, using default weights
# 	detector = MTCNN()
# 	# detect faces in the image
# 	results = detector.detect_faces(pixels)
# 	# extract the bounding box from the first face
# 	x1, y1, width, height = results[0]['box']
# 	# bug fix
# 	x1, y1 = abs(x1), abs(y1)
# 	x2, y2 = x1 + width, y1 + height
# 	# extract the face
# 	face = pixels[y1:y2, x1:x2]
# 	# resize pixels to the model size
# 	image = Image.fromarray(face)
# 	image = image.resize(required_size)
# 	face_array = asarray(image)
# 	return face_array
#
# # load images and extract faces for all images in a directory
# def load_faces(directory):
# 	faces = list()
# 	# enumerate files
# 	for filename in listdir(directory):
# 		# path
# 		path = directory + filename
# 		# get face
# 		face = extract_face(path)
# 		# store
# 		faces.append(face)
# 	return faces
#
# def load_dataset(directory):
# 	X, y = list(), list()
# 	# enumerate folders, on per class
# 	for subdir in listdir(directory):
# 		# path
# 		path1 = directory + subdir + '/'
# 		# skip any files that might be in the dir
# 		if not path.isdir(path1):
# 			continue
# 		# load all faces in the subdirectory
# 		faces = load_faces(path1)
# 		# create labels
# 		labels = [subdir for _ in range(len(faces))]
# 		# summarize progress
# 		print('>loaded %d examples for class: %s' % (len(faces), subdir))
# 		# store
# 		X.extend(faces)
# 		y.extend(labels)
#
# 	return asarray(X), asarray(y)

@img.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)