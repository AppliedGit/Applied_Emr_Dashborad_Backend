import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
from .services.database import Database
from .services.authentication import Authentication
from flask_executor import Executor
from app.services.background_tasks import BackgroundTask


jwt = JWTManager()

def create_app():
    app = Flask(__name__)
    CORS(app)  

    app.database = Database()
    app.authentication = Authentication(app)

    jwt.init_app(app)
    # executor = Executor(app)
    
    # # Initialize the background task runner
    # app.background_runner = BackgroundTask(executor)
    executor = Executor(app)
    
        # Initialize the background task runner
    app.background_runner = BackgroundTask(executor)

    # Make sure that background_runner is part of the app context
    with app.app_context():
        app.background_runner = BackgroundTask(executor)

    from .routes.auth_routes import auth_bp
    from .routes.train_model import train_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(train_bp)

    return app
