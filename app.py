from flask import Flask, render_template,request
from scipy.misc import imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import base64

sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model, graph
model, graph = init()

def convertImage(imgData1):
	imgstr = re.search(rb'base64,(.*)',imgData1).group(1) #cleaning the metadata to convert it into an image file
	with open('output.png','wb') as output: #saving it for testing puposes
		output.write(base64.decodestring(imgstr))

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x = imread('output.png',mode='L')
	x = np.invert(x) #inverting the image so that white turns into black and black into white for better performance
	x = imresize(x,(28,28)) #resizing the image from 280x280 which was retrieved from the canvas to 28x28
	x = x.reshape(1,28,28,1) #reshaping the image to 28x28 pixels
	with graph.as_default():
		out = model.predict(x) #out contains an np.ndarray
		response = np.array_str(np.argmax(out,axis=1)) #to return a valid number we converted the ndarray and save the output in response
		return response


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
