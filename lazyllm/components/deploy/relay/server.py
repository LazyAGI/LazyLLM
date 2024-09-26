import cloudpickle
import uvicorn
import argparse
import base64
import os
import sys
import inspect
import traceback
from types import GeneratorType
from lazyllm import kwargs, package
from lazyllm import FastapiApp, globals, decode_request
import pickle
import codecs
import asyncio
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import requests

# TODO(sunxiaoye): delete in the future
lazyllm_module_dir = os.path.abspath(__file__)
for _ in range(5):
    lazyllm_module_dir = os.path.dirname(lazyllm_module_dir)
sys.path.append(lazyllm_module_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--open_ip", type=str, default="0.0.0.0",
                    help="IP: Receive for Client")
parser.add_argument("--open_port", type=int, default=17782,
                    help="Port: Receive for Client")
parser.add_argument("--function", required=True)
parser.add_argument("--before_function")
parser.add_argument("--after_function")
parser.add_argument("--pythonpath")
args = parser.parse_args()

def load_func(f):
    return cloudpickle.loads(base64.b64decode(f.encode('utf-8')))

if args.pythonpath:
    sys.path.append(args.pythonpath)

func = load_func(args.function)
if args.before_function:
    before_func = load_func(args.before_function)
if args.after_function:
    after_func = load_func(args.after_function)


app = FastAPI()
FastapiApp.update()

async def async_wrapper(func, *args, **kwargs):
    loop = asyncio.get_running_loop()

    def impl(func, sid, global_data, *args, **kw):
        globals._init_sid(sid)
        globals._update(global_data)
        return func(*args, **kw)

    result = await loop.run_in_executor(None, partial(impl, func, globals._sid, globals._data, *args, **kwargs))
    return result

@app.post("/generate")
async def generate(request: Request): # noqa C901
    try:
        globals._init_sid(decode_request(request.headers.get('Session-ID')))
        globals._update(decode_request(request.headers.get('Global-Parameters')))
        input, kw = (await request.json()), {}
        try:
            input, kw = decode_request(input)
        except Exception: pass
        origin = input

        if args.before_function:
            assert (callable(before_func)), 'before_func must be callable'
            r = inspect.getfullargspec(before_func)
            if isinstance(input, kwargs) or (
                    isinstance(input, dict) and set(r.args[1:] if r.args[0] == 'self' else r.args) == set(input.keys())):
                assert len(kw) == 0, 'Cannot provide kwargs-input and kwargs at the same time'
                input = before_func(**input)
            else:
                input = func(*input, **kw) if isinstance(input, package) else before_func(input, **kw)
                kw = {}
        if isinstance(input, kwargs):
            kw.update(input)
            ags = ()
        else:
            ags = input if isinstance(input, package) else (input,)
        output = await async_wrapper(func, *ags, **kw)

        def impl(o):
            return codecs.encode(pickle.dumps(o), 'base64')

        if isinstance(output, GeneratorType):
            def generate_stream():
                for o in output:
                    yield impl(o)
            return StreamingResponse(generate_stream(), media_type='text_plain')
        elif args.after_function:
            assert (callable(after_func)), 'after_func must be callable'
            r = inspect.getfullargspec(after_func)
            assert len(r.args) > 0 and r.varargs is None and r.varkw is None
            # TODO(wangzhihong): specify functor and real function
            new_args = r.args[1:] if r.args[0] == 'self' else r.args
            if len(new_args) == 1:
                output = after_func(output) if len(r.kwonlyargs) == 0 else \
                    after_func(output, **{r.kwonlyargs[0]: origin})
            elif len(new_args) == 2:
                output = after_func(output, origin)
        return Response(content=impl(output))
    except requests.RequestException as e:
        return Response(content=f'{str(e)}', status_code=500)
    except Exception as e:
        return Response(content=f'{str(e)}\n--- traceback ---\n{traceback.format_exc()}', status_code=500)
    finally:
        globals.clear()


def find_services(cls):
    if '__relay_services__' not in dir(cls): return
    if '__relay_services__' in cls.__dict__:
        for (method, path), (name, kw) in cls.__relay_services__.items():
            if getattr(func.__class__, name) is getattr(cls, name):
                getattr(app, method)(path, **kw)(getattr(func, name))
    for base in cls.__bases__:
        find_services(base)

find_services(func.__class__)

if __name__ == "__main__":
    uvicorn.run(app, host=args.open_ip, port=args.open_port)
