from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

db = SQLAlchemy()
DB_NAME = "faceVData.db"
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'tranminhkhoa123456'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    db.init_app(app)

    UPLOAD_FOLDER = 'static/uploads/'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    from .views import views
    from .auth import auth
    from FVAPP.controller.image import img
    from FVAPP.controller.verify import imgV

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, urlprefix='/')
    app.register_blueprint(img, urlprefix='/')
    app.register_blueprint(imgV, urlprefix='/')

    from .models import User, Image

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    return app

def create_database(app):
    if not path.exists('FVAPP/'+DB_NAME):
        db.create_all(app=app)
        print('Created Database!')

