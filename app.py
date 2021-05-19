from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import random
from os import listdir
import os
import pymongo
from pymongo import MongoClient
import plotly as py
import json
from datetime import datetime

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def launch():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)