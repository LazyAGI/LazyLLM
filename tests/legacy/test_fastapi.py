import time
import json
import requests
from typing import Any

import pydantic
from pydantic import BaseModel
from starlette.responses import RedirectResponse

import lazyllm
from lazyllm import FastapiApp as app


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

    @app.get('/', response_model=BaseResponse, summary='docs')
    def document(self):
        return RedirectResponse(url='/docs')

    @app.post('/getres')
    def test(self):
        return 'test'
    
    def __call__(self, inp):
        return 'inps'

class TestFn_FastAPI(object):

    def test_fastapi(self):
        m = lazyllm.ServerModule(Manager())
        m.start()

        err_url = m._url
        org_url = err_url.replace("generate", "getres")
        doc_url = err_url.replace("generate", "docs")

        response = requests.get(doc_url)
        assert response.status_code == 200

        response = requests.post(org_url, data=json.dumps('ww'))
        assert response.json() == "test"
