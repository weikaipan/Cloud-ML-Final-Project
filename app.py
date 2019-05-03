from flask import Flask, jsonify, request,make_response
import os
from settings import APP_STATIC,APP_ROOT
import contextlib
import subprocess
from train.train import *

app = Flask(__name__)

@app.route('/')
def default_route():
    return app.send_static_file('index.html')

@app.route('/logs2', methods=['GET'])
def showLogs2():
    with open(os.path.join(APP_STATIC, 'dummyFile')) as f:
        content=f.read()
        f.close()
    return jsonify(content)

@app.route('/train', methods=['POST'])
def startTraining():
    req=request.get_json()
    modelMapping={1:"BASELINE",2:"RNN",3:"CNN"}
    if req["id"] not in modelMapping:
      return make_response('You need to select a model', 405)

    with open('training.log','w') as f:
        with contextlib.redirect_stdout(f):
            main(stop=True,topology=modelMapping[req["id"]])
    # fwrite = open("blah.txt", "w")
    # subprocess.call(["python", "train/train.py", "-stop", "True", "-topology","BASELINE"],stdout=fwrite)
    return jsonify('')

@app.route('/logs', methods=['GET'])
def showLogs():
    # fwrite = open("blah.txt", "w")
    # subprocess.call(["python", "train/train.py", "-stop", "True", "-topology","BASELINE"],stdout=fwrite)

    with open(os.path.join(APP_ROOT, 'training.log')) as f:
        content=f.read()
        f.close()
    return jsonify(content)



if __name__ == '__main__':
    app.run('0.0.0.0', 8000, debug=True)