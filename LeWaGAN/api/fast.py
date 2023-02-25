from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from LeWaGAN.interface.main import generate_model
from LeWaGAN.model.registery import load_model, load_eigenvectors
from LeWaGAN.postprocessing.generate import generate_noise, image_with_eigenvectors
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
model = generate_model()
app.state.model = load_model(model)
app.state.noise = generate_noise(1, 1)
app.state.vector = load_eigenvectors()

@app.get("/")
def index():
    return {"status": "ok"}

@app.get('/image', response_class=Response)
def get_image(v1 = int,
              v2 = int,
              v3 = int,
              v4 = int,
              v5 = int,
              seed = int
              ):
    # loading image from disk
    # im = Image.open('test.png')
    app.state.noise = generate_noise(1,int(seed))
    alpha = [int(v1), int(v2), int(v3), int(v4), int(v5)]
    img = image_with_eigenvectors(app.state.model, app.state.noise,alpha,app.state.vector)
    # using an in-memory image
    im = Image.fromarray(img)
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')
