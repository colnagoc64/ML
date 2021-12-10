from fastapi import FastAPI ,File,UploadFile
from PIL import Image, ImageOps
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras


app=FastAPI

@app.get("/")
async def root():
    return{"message":"Hello World"}

@app.get('/test')
async def test():
    return{"message":"test"}

loaded_model=keras.models.load_model('my_mnist')
print("Loaded model",loaded_model)
@app.post('/uploadfile/')
async def create_upload_file(file:UploadFile = File(...)):

    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    img_gray= pil_image.covert('L')
    img_inverted = ImageOps.invert(img_gray)

    pixel_list = np.asfarray(img_inverted).flatten().tolist()
    input = (np.asfarray(pixel_list)/255.0 *0.99)+0.01

    preds =loaded_model.predict(input.reshape((1,28,28,1)), batch_size=1)

    return {'result_cnn':str(preds[0].argmax()),'scores_cnn':str(preds[0].tolist())}