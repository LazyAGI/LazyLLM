import os
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

class Manager(object):

    @app.get('/', response_model=BaseResponse, summary='docs')
    def document(self):
        return RedirectResponse(url='/docs')

    @app.post('/getres')
    def test(self):
        return 'test'

    @app.post('/getres2')
    def test_overwrite(self):
        return 'test_overwrite'

    def __call__(self, inp):
        return 'inps'


class Manager2(object):
    @app.post('/manager2')
    def test_manager2(self):
        return 'manager2'


class Derived(Manager, Manager2):
    def test_overwrite(self):
        return 'test_overwrite'

    @app.post('/subclass_api')
    def test_subclass_api(self):
        return 'subclass_api'


class TestServerModule(object):

    def test_base_module(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        m = lazyllm.ServerModule(Manager(), pythonpath=current_dir)
        m.start()

        response = requests.get(m._url.replace("generate", "docs"))
        assert response.status_code == 200

        response = requests.post(m._url.replace("generate", "getres"), data=json.dumps('ww'))
        assert response.json() == "test"

        response = requests.post(m._url.replace("generate", "getres2"), data=json.dumps('ww'))
        assert response.json() == "test_overwrite"

    def test_derived_module(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        m = lazyllm.ServerModule(Derived(), pythonpath=current_dir)
        m.start()

        response = requests.get(m._url.replace("generate", "docs"))
        assert response.status_code == 200

        response = requests.post(m._url.replace("generate", "getres"), data=json.dumps('ww'))
        assert response.json() == "test"

        response = requests.post(m._url.replace("generate", "manager2"), data=json.dumps('ww'))
        assert response.json() == "manager2"

        response = requests.post(m._url.replace("generate", "getres2"), data=json.dumps('ww'))
        assert response.status_code == 404

        response = requests.post(m._url.replace("generate", "subclass_api"), data=json.dumps('ww'))
        assert response.json() == "subclass_api"
