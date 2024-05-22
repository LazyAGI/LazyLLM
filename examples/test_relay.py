import lazyllm
from lazyllm import FastapiApp as app
from typing import Any
from starlette.responses import RedirectResponse
import pydantic
from pydantic import BaseModel
import time


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

class Manager():
    def test1(self):
        print('test1')

    @app.get('/', response_model=BaseResponse, summary='docs')
    def document(self):
        return RedirectResponse(url='/docs')

    @app.post('/generate1')
    def test2(self):
        return 'test2'

    @app.post('/generate2')
    def test3(self):
        print('test3')

    def __call__(self, inp):
        return f'get inp'


m = lazyllm.ServerModule(Manager())
m.start()
    
time.sleep(1000)