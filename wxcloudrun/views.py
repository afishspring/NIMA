from datetime import datetime
from flask import request
from run import app
from wxcloudrun.dao import delete_counterbyid, query_counterbyid, insert_counter, update_counterbyid
from wxcloudrun.model import Counters
from wxcloudrun.utils.response import make_succ_empty_response, make_succ_response, make_err_response

import os
import numpy as np
from pathlib import Path
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from wxcloudrun.utils.model import model
from wxcloudrun.utils.score_utils import mean_score, std_score


@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    photo = request.files['image']

    idx = np.random.randint(1, 1000)

    save_path = os.path.join('./wxcloudrun/img/', f'temp_{idx}.jpg')

    photo.save(save_path)

    x = load_img(save_path, target_size=(224, 224))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    scores = model.predict(x, batch_size=1, verbose=0)[0]

    mean = mean_score(scores)
    std = std_score(scores)

    file_name = Path(save_path).name.lower()
    score_list=(file_name, mean, std)

    print("Evaluating : ", save_path)
    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
    
    return make_succ_response(score_list)

@app.route('/xxx')
def index():
    """
    :return: 返回index页面
    """
    return 'hello'


@app.route('/api/count', methods=['POST'])
def count():
    """
    :return:计数结果/清除结果
    """

    # 获取请求体参数
    params = request.get_json()

    # 检查action参数
    if 'action' not in params:
        return make_err_response('缺少action参数')

    # 按照不同的action的值，进行不同的操作
    action = params['action']

    # 执行自增操作
    if action == 'inc':
        counter = query_counterbyid(1)
        if counter is None:
            counter = Counters()
            counter.id = 1
            counter.count = 1
            counter.created_at = datetime.now()
            counter.updated_at = datetime.now()
            insert_counter(counter)
        else:
            counter.id = 1
            counter.count += 1
            counter.updated_at = datetime.now()
            update_counterbyid(counter)
        return make_succ_response(counter.count)

    # 执行清0操作
    elif action == 'clear':
        delete_counterbyid(1)
        return make_succ_empty_response()

    # action参数错误
    else:
        return make_err_response('action参数错误')


@app.route('/api/count', methods=['GET'])
def get_count():
    """
    :return: 计数的值
    """
    counter = Counters.query.filter(Counters.id == 1).first()
    return make_succ_response(0) if counter is None else make_succ_response(counter.count)
