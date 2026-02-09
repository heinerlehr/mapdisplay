import panel as pn

from fastapi import FastAPI
from panel.io.fastapi import add_application

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@add_application('/panel', app=app, title='My Panel App')
def create_panel_app():
    slider = pn.widgets.IntSlider(name='Slider', start=0, end=10, value=3)
    return slider.rx() * '⭐'