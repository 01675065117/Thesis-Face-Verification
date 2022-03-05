from flask import Blueprint,render_template
from flask_login import login_required, current_user

views = Blueprint('views', __name__)

# @views.route('/home')
@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/verify')
@login_required
def verify():
    return render_template("verify.html", user=current_user)

# from flask import Flask, flash, request, redirect, url_for, render_template, Blueprint
# import urllib.request
# import os
# from werkzeug.utils import secure_filename
# #img = Blueprint('img', __name__)
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# UPLOAD_FOLDER = 'static/uploads/'
# views.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# @views.route('/', methods=['POST', 'GET'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(views.config['UPLOAD_FOLDER'], filename))
#         # print('upload_image filename: ' + filename)
#         flash('Image successfully uploaded and displayed below')
#         return render_template('home.html', filename=filename)
#     else:
#         flash('Allowed image types are - png, jpg, jpeg, gif')
#         return redirect(request.url)
#
# @views.route('/display/<filename>')
# def display_image(filename):
#     # print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)