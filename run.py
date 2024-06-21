import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from utils import *
from pathlib import Path
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.mobilenet import preprocess_input

app = Flask(__name__)

# 设置上传文件保存的目录
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    try:
        data = request.json  # 假设前端发送的是 JSON 格式数据
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # 获取 base64 编码的图片数据
        image_base64 = data['image']

        # 解码 base64 字符串
        image_data = base64.b64decode(image_base64.split(',')[1])

        # 生成一个随机的文件名
        idx = np.random.randint(1, 1000)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{idx}.jpg')

        # 保存解码后的图片数据为文件
        with open(save_path, 'wb') as f:
            f.write(image_data)

        # 加载图像并进行预处理
        x = load_img(save_path, target_size=(224, 224))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 这里需要根据你的模型来进行预测
        # 假设你的模型是在全局变量中定义的，例如 model
        scores = nima.predict(x, batch_size=1, verbose=0)[0]

        # 计算均值和标准差
        mean = mean_score(scores)
        std = std_score(scores)

        # 获取文件名并转换为小写
        file_name = Path(save_path).name.lower()

        print("Evaluating:", save_path)
        print("NIMA Score: %0.3f +- (%0.3f)" % (mean, std))

        # 返回评估结果
        return jsonify({'file_name': file_name, 'mean': mean, 'std': std}), 200

    except Exception as e:
        print("Error evaluating image:", e)
        return jsonify({'error': 'Error evaluating image'}), 500

if __name__ == '__main__':
    app.run(debug=True)