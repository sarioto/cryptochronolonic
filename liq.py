from tinydb import TinyDB
from flask import Flask, jsonify

app = Flask(__name__)

db = TinyDB('./live_hist/memories.json')

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/latest_hist')
def get_paper_hist():
    return jsonify(db.all())

