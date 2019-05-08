from flask import Flask, jsonify, request, make_response, url_for
from settings import APP_STATIC,APP_ROOT
from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

import os
import contextlib
import subprocess
import random
import time

app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

@app.route('/train', methods=['POST'])
def train_model():
    from train.train import main
    task = main.apply_async((), {'stop': True})
    print('Task id is {}'.format(task.id))
    response = jsonify()
    response.status_code = 202
    response.headers['location'] = url_for('model_status', task_id=task.id)
    return response

@app.route('/log/<task_id>', methods=['GET'])
def model_status(task_id):
    from train.train import main
    print('in status task id is {}'.format(task_id))
    task = main.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'status': 'Pending...',
            'verbose': True
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'verbose': True
        }
        if task.state == 'READING':
            response['reading'] = task.info.get('INFO', 'none')
        elif task.state == 'TRAIN':
            response['train'] = task.info.get('INFO', 'none')
        elif task.state == 'PROGRESS':
            response['epoch'] = task.info.get('Epoch', 'none')
            response['train'] = task.info.get('Train', 'none')
            response['val'] = task.info.get('Val', 'none')
        elif task.state == 'END':
            response['test'] = task.info.get('Test', 'none')
        else:
            response['success'] = 'RUNNING'
            print(task.info)
            response['epoch'] = task.info.get('Epoch', 'none')
            response['train'] = task.info.get('Train', 'none')
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }
    print(response)
    return jsonify(response)

@app.route('/')
def default_route():
    return app.send_static_file('index.html')

@app.route('/tasks')
def tasks():
    return app.send_static_file('tasks.html')

@app.route('/product')
def product():
    return app.send_static_file('product.html')

@app.route('/logs2', methods=['GET'])
def showLogs2():
    with open(os.path.join(APP_STATIC, 'dummyFile')) as f:
        content=f.read()
        f.close()
    return jsonify(content)

# @app.route('/train', methods=['POST'])
# def startTraining():
#     req=request.get_json()
#     modelMapping={1:"BASELINE",2:"RNN",3:"CNN"}
#     if req["id"] not in modelMapping:
#       return make_response('You need to select a model', 405)

#     with open('training.log','w') as f:
#         with contextlib.redirect_stdout(f):
#             main(stop=True,topology=modelMapping[req["id"]])
#     # fwrite = open("blah.txt", "w")
#     # subprocess.call(["python", "train/train.py", "-stop", "True", "-topology","BASELINE"],stdout=fwrite)
#     return jsonify('')

# @app.route('/logs', methods=['GET'])
# def showLogs():
#     # fwrite = open("blah.txt", "w")
#     # subprocess.call(["python", "train/train.py", "-stop", "True", "-topology","BASELINE"],stdout=fwrite)

#     with open(os.path.join(APP_ROOT, 'training.log')) as f:
#         content=f.read()
#         f.close()
#     return jsonify(content)

# @app.route('/logs/periodic-get', methods=['GET'])
# def get_logs():


@app.route('/test', methods=['GET'])
def test():
    from train.test import deploy_model
    if request.args['sentence'] is None:
        ret = { 'sentiment': deploy_model() }
    else:
        ret = { 'sentiment': deploy_model(sentence=request.args['sentence']) }
    return jsonify(ret)


if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)