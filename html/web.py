from flask import Flask
app = Flask(__name__)
import random

@app.route('/')
def index():
    with open("example1.html") as f:
        html=f.read()
    return html

@app.route('/dynamic')
def dynamic():
    with open("dynamic.html") as f:
        html=f.read()
    return html

@app.route('/dynamic_response')
def dynamic_response():
    return str(random.random())

if __name__ == '__main__':
    app.run()