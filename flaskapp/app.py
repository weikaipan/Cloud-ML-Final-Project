from flask import Flask, jsonify, request
import os
from settings import APP_STATIC

app = Flask(__name__)

@app.route('/')
def default_route():
    return app.send_static_file('index.html')

@app.route('/logs', methods=['GET'])
def showLogics():
    with open(os.path.join(APP_STATIC, 'dummyFile')) as f:
        content=f.read()
        f.close()
    return jsonify('hello logs'+content)



if __name__ == '__main__':
    app.run('0.0.0.0', 8000, debug=True)