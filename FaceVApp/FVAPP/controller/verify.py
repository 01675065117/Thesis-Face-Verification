import base64
import io

from flask import Flask, flash, request, redirect, url_for, render_template, Blueprint
from FVAPP.models import Image, User
import os
from FVAPP.__init__ import db
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from numpy import asarray
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image as Image1
import matplotlib.image as mpimg
from os import listdir
import os.path as path
import numpy as np
from keras.models import load_model
from scipy import spatial
from keras.models import load_model
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

model = load_model('D:/Tai-Lieu-Hoc/TNCKH/Graduation_Thesis/facenet_keras.h5')
imgV = Blueprint('imgV', __name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'D:/Tai-Lieu-Hoc/TNCKH/Graduation_Thesis/FaceVApp/FVAPP/static/uploads/'


# img.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@imgV.route('/verify', methods=['POST', 'GET'])
@login_required
def verify_image():
    # if 'files1' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    # if 'files2' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    #
    file1 = request.files['files1']
    file2 = request.files['files2']

    # if file1.index == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    #
    # if file2.index == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)

    if file1 and allowed_file(file1.filename):
        filename1 = secure_filename(file1.filename)
        file1.save(os.path.join(UPLOAD_FOLDER, filename1))
        path1 = os.path.join(UPLOAD_FOLDER, filename1)
        pixels1 = extract_face(path1)
        newTrainX1 = list()
        embedding1 = get_embedding(model, pixels1)
        newTrainX1.append(embedding1)
        newTrainX1 = np.asarray(newTrainX1)
        flash('The first image successfully uploaded', category='success')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

    if file2 and allowed_file(file2.filename):
        filename2 = secure_filename(file2.filename)
        file2.save(os.path.join(UPLOAD_FOLDER, filename2))
        path2 = os.path.join(UPLOAD_FOLDER, filename2)
        pixels2 = extract_face(path2)
        newTrainX2 = list()
        embedding2 = get_embedding(model, pixels2)
        newTrainX2.append(embedding2)
        newTrainX2 = np.asarray(newTrainX2)
        print("shape: ",newTrainX2.shape)
        flash('The second image successfully uploaded', category='success')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    user1 = str(current_user.id)
    Ims = Image.query.filter_by(user_id=user1).order_by(Image.path).all()
    for im in Ims:
        print(im.path)

    compare_images(pixels1, pixels2, newTrainX1, newTrainX2, current_user.first_name)

    pic = current_user.first_name + '.jpg'
    img1 = Image1.open(os.path.join(UPLOAD_FOLDER, pic))
    data = io.BytesIO()
    img1.save(data,"JPEG")

    encode_image = base64.b64encode(data.getvalue())
    return render_template('verify.html', pic = encode_image.decode("UTF-8"), user=current_user)


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image1.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image1.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(path1, path2, imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    spicy = 1 - spatial.distance.cosine(imageA, imageB)
    imgB_T = imageB.T
    nump = imageA.dot(imgB_T)/ (np.linalg.norm(imageA, axis=1) * np.linalg.norm(imgB_T))
    skl = 1 - cosine_distances(imageA,imageB)
    # s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.5f, Spicy: %.5f, Numpy: %.5f, Sklearn: %.5f" % (m, spicy, nump, skl))
    # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(path1, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(path2, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.savefig(UPLOAD_FOLDER + title + '.jpg')


# @imgV.route('/verify/<pic>')
# def display_image_verify(pic):
#     # print('display_image filename: ' + filename)
#     return redirect(url_for('static', pic='uploads/' + pic), code=301)
