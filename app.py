from flask import Flask, request, Response, jsonify
import numpy as np
from CrossNetRes2 import CrossNet
import io
from PIL import Image
import base64
import torchvision.transforms as transforms
import cv2
import torch
import torch.nn.functional as F
MODEL_CHECKPOINT ='./model.pt'
##############################################################
#                                                            #
#    env:denoise                                             #
#                                                            #
#                                                            #
##############################################################

# Initialize the Flask application
app = Flask(__name__)
model = CrossNet()
model.eval()
model.double()
model.load_state_dict(torch.load(MODEL_CHECKPOINT))
def imageTob64StringEncode(pil_im,pil_im_format):
    b = io.BytesIO()
    pil_im.save(b,pil_im_format)
    im_bytes = b.getvalue()
    return base64.b64encode(im_bytes).decode('utf-8')
def b64StringToImageDecode(b64):
    img  = Image.open(io.BytesIO(base64.b64decode(b64)))
    return img
def paddingNum(size):
    padding = 0 if size%8==0 else 8-size%8
    first = padding//2
    last = padding-first
    return last,first
def preprocess(img,padding):
    return F.pad(img,padding,)
def denoise(img,model):
    t= t = transforms.Compose([
        transforms.ToTensor(),
        ])
    img = cv2.imread('./kkk.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (t(img)*255).double().unsqueeze(0)
    padding =(*paddingNum(img.shape[3]),*(paddingNum(img.shape[2])))
    img = preprocess(img,padding)
    img = model(img)
    img[img<0]=0
    img[img>255]=255
    img = np.transpose((img[0]).detach().numpy().astype('uint8'),(1,2,0))
    img = img[padding[2]:img.shape[0]-padding[3],padding[0]:img.shape[1]-padding[1],:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./mt2cv1.jpg',img)

# route http posts to this method
@app.route('/load', methods=['POST'])
def test1():
    img = b64StringToImageDecode(request.form['data'])
    img.save('./kkk.jpg')
    imgformat = img.format
    denoise(img,model)
    denoised = Image.open('./mt2cv1.jpg')
    op = imageTob64StringEncode(denoised,imgformat)
    return jsonify(image= op)
@app.route('/', methods=['GET'])
def test():
    return "<h1>Welcome to our server new !!</h1>"
if __name__ =='__main__':
    app.run(threaded=True)
