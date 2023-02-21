from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from LeWaGAN.model.registery import load_model
from LeWaGAN.postprocessing.generate import generate_noise, display_sample, eigenvectors

import numpy as np
import io

from PIL import Image


app = FastAPI()



#Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
 )

app.state.model = load_model()
app.state.noise = generate_noise(1)

#alpha = [1,1,2,2]

@app.get("/")
def index():
    return {"status": "ok"}

@app.get('/image', response_class=Response)
def get_image(alpha):
    # loading image from disk
    # im = Image.open('test.png')


    img = display_sample(app.state.model, app.state.noise,alpha)
    # using an in-memory image
    im = Image.fromarray(img)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')
