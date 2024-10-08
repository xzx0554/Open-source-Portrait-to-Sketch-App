from flask import Flask, render_template, request, redirect, url_for, make_response, send_from_directory, jsonify
import json
import os
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from watermarker.marker import add_mark
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import torchvision
import time
import torch
if torch.cuda.is_available():
    # 指定要使用的 GPU 设备
    torch.cuda.set_device(3)
def wait_for_file(file_path, retries=3, wait_seconds=3):

    for attempt in range(1, retries + 1):
        if os.path.isfile(file_path):
            return True
        if attempt < retries:
            time.sleep(wait_seconds)
    raise FileNotFoundError(f"File not saved: {file_path}")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3 for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): 
        self.hooks.remove()

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

app = Flask(__name__)

# Global model loading
portrait_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting',device='gpu:3')
learn = load_learner('./', 'ArtLine_920.pkl')
learn.model = learn.model.to(torch.device('cuda:3'))
# File paths
CARDS_FILE = 'cards.json'
UPLOAD_FOLDER = 'static/upload'
DOWNLOAD_FOLDER = 'static/download'

# Ensure upload and download directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def load_cards():
    with open(CARDS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_cards(cards):
    with open(CARDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=4, ensure_ascii=False)

def get_card(card_id):
    cards = load_cards()
    for card in cards:
        if card['card_id'] == card_id.strip():
            return card
    return None

@app.route('/', methods=['GET'])
def index():
    card = None
    message = None
    card_id = request.cookies.get('card_id')
    if card_id:
        card = get_card(card_id)
        if not card:
            message = "卡密无效，请重新输入。"
    return render_template('index.html', card=card, message=message)

@app.route('/process', methods=['POST'])
def process_card():
    card_id = request.form.get('card_id')
    card = get_card(card_id)
    message = None
    if card:
        resp = make_response(redirect(url_for('index')))
        resp.set_cookie('card_id', card_id)
        return resp
    else:
        return render_template('index.html', card=None, message="卡密无效，请重新输入。")

@app.route('/upload', methods=['POST'])
def upload():
    card_id = request.cookies.get('card_id')
    if not card_id:
        return redirect(url_for('index'))
    card = get_card(card_id)
    if not card:
        return redirect(url_for('index'))

    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Only allow image formats
    if not (file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))):
        return redirect(url_for('index'))

    user_upload_folder = os.path.join(UPLOAD_FOLDER, card_id)
    os.makedirs(user_upload_folder, exist_ok=True)

    filename = file.filename
    upload_path = os.path.join(user_upload_folder, filename)
    file.save(upload_path)

    # Processing Steps
    # 1. Portrait Matting
    result = portrait_matting(upload_path)
    result_array = np.where(result[OutputKeys.OUTPUT_IMG][:, :, 3:4] != 0, 1,255)
    result_image = result[OutputKeys.OUTPUT_IMG][:, :, :3] * result_array

    processed_image_no_watermark_path = os.path.join(user_upload_folder, f'processed_{filename}')
    cv2.imwrite(processed_image_no_watermark_path, result_image)
    wait_for_file(processed_image_no_watermark_path)
    assert os.path.isfile(processed_image_no_watermark_path), "File not saved"

    img = PIL.Image.open(processed_image_no_watermark_path).convert("RGB")
    im_new = add_margin(img, 250, 250, 250, 250, (255, 255, 255))
    im_new.save(processed_image_no_watermark_path, quality=95)
    wait_for_file(processed_image_no_watermark_path)

    assert os.path.isfile(processed_image_no_watermark_path), "File not saved"

    # 2. Create Simplified Drawing using FastAI
    img = open_image(processed_image_no_watermark_path)
    p, img_hr, b = learn.predict(img)

    result_image = img_hr.cpu().numpy()
    img_hr = (img_hr - img_hr.min()) / (img_hr.max() - img_hr.min())
    torchvision.utils.save_image(img_hr, processed_image_no_watermark_path)

    wait_for_file(processed_image_no_watermark_path)
    assert os.path.isfile(processed_image_no_watermark_path), "File not saved"

    # 3. Add Watermark
    processed_folder_with_watermark_path = os.path.join(user_upload_folder, 'processed_mask')
    processed_image_with_watermark_path = os.path.join(processed_folder_with_watermark_path, f'processed_{filename}')
    os.makedirs(processed_folder_with_watermark_path, exist_ok=True)
    add_mark(file=processed_image_no_watermark_path, out=processed_folder_with_watermark_path, mark="only for test", opacity=0.2, angle=45, space=30)

    wait_for_file(processed_image_with_watermark_path)
    assert os.path.isfile(processed_image_with_watermark_path), "File not saved"

    # Return the processed image with watermark for display
    processed_image_url = os.path.join('static', 'upload', card_id, 'processed_mask', f'processed_{filename}')
    return render_template('index.html', card=card, processed_image=processed_image_url, original_filename=filename)

@app.route('/download/<card_id>/<filename>', methods=['GET'])
def download(card_id, filename):
    # Validate card
    card = get_card(card_id)
    if not card:
        return redirect(url_for('index'))

    # Check balance
    if card['amount'] < 3:
        return "余额不足，无法下载。", 400

    # Decrement balance
    cards = load_cards()
    for c in cards:
        if c['card_id'] == card_id:
            c['amount'] -= 3
            break
    save_cards(cards)

    # Create download directory
    user_download_folder = os.path.join(DOWNLOAD_FOLDER, card_id)
    os.makedirs(user_download_folder, exist_ok=True)

    # Use the processed image without watermark for download
    processed_image_no_watermark_path = os.path.join(UPLOAD_FOLDER, card_id, f'processed_{filename}')
    if not os.path.exists(processed_image_no_watermark_path):
        return "处理后的图片不存在。", 404
    download_path = os.path.join(user_download_folder, filename)
    cv2.imwrite(download_path, cv2.imread(processed_image_no_watermark_path))

    return send_from_directory(user_download_folder, filename, as_attachment=True)
@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=7777)
