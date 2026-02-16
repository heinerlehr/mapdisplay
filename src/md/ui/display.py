import os
from pathlib import Path
import panel as pn

from fastapi import FastAPI
from panel.io.fastapi import add_application

# Configuration
from iconfig.iconfig import iConfig

from loguru import logger


app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@add_application('/panel', app=app, title='My Panel App')
def create_panel_app():
    slider = pn.widgets.IntSlider(name='Slider', start=0, end=10, value=3)
    return slider.rx() * '⭐'

def start():
    host = os.getenv("MAPDISPLAY_HOST", default="localhost")
    host = "0.0.0.0" if Path("/.dockerenv").exists() else host
    port = int(os.getenv("MAPDISPLAY_PORT", default="5006"))

    pn.serve(create_panel_app, title="Login", autoreload=False, address=host, port=port, show=True)