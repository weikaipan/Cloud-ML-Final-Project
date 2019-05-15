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
    print(request.json)
    from train.train import main

    print(request.json['model'])
    packed = False if request.json['model'] == 'BASELINE' else True
    task = main.apply_async((), {'stop': True,
                                 'packed': packed,
                                 'embedding_size': request.json['embedding'],
                                 'learning_rate': request.json['lr'],
                                 'topology': request.json['model'],
                                 'optim_option': request.json['optim']})

    print('Task id is {}'.format(task.id))
    response = jsonify()
    response.status_code = 202
    response.headers['location'] = url_for('model_status', task_id=task.id)
    response.headers['task_id'] = task.id
    return response

@app.route('/cancel/<task_id>', methods=['DELETE'])
def cancel_task(task_id):
    print("cancel")
    print(task_id)
    from celery.task.control import revoke
    try:
        revoke(task_id, terminate=True)
        return jsonify({ 'success': 'cancel the task' })
    except Exception as e:
        print(e)
        return jsonify({ 'fail': 'cancel fail' })

@app.route('/log/<task_id>', methods=['GET'])
def model_status(task_id):
    from train.train import main
    print('in status task id is {}'.format(task_id))
    task = main.AsyncResult(task_id)
    print(task.state)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'status': 'Pending...',
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'task': task.info
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
            'error': True
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

@app.route('/test', methods=['GET'])
def test():
    from train.test import deploy_model
    if request.args['sentence'] is None:
        ret = { 'sentiment': deploy_model() }
    else:
        ret = { 'sentiment': deploy_model(sentence=request.args['sentence'], topology=request.args['topology']) }
    return jsonify(ret)


if __name__ == '__main__':
    app.run('0.0.0.0', 8000, debug=True)