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
            print(filename)
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


@img.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)