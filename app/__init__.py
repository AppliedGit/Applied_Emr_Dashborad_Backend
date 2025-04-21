import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
from .services.database import Database
from .services.authentication import Authentication


jwt = JWTManager()

def create_app():
    app = Flask(__name__)
    CORS(app)  

    app.database = Database()
    app.authentication = Authentication(app)

    jwt.init_app(app)

    from .routes.auth_routes import auth_bp
    from .routes.train_model import train_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(train_bp)

    return app
