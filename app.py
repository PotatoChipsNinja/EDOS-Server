from flask import Flask, request, render_template
from flask_cors import CORS
from predictor import Predictor
import pandas as pd

print('loading predictor')
predictor = Predictor()
examples = pd.read_csv('./examples.csv')
print('predictor loaded')

app = Flask(__name__, static_folder="dist", static_url_path="", template_folder = "dist")
CORS(app)

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['GET'])
def predict_api():
    text = request.args.get('text')
    result = predictor.predict(text)
    return result

@app.route("/example", methods=['GET'])
def example_api():
    return examples.sample(5)['text'].to_list()

@app.route('/')
def index():
    return render_template("index.html")

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html')
