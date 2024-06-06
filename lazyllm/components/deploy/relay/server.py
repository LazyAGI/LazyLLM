import cloudpickle
import uvicorn
import argparse
import base64
import os
import sys
import inspect
import traceback
from types import GeneratorType
from lazyllm import LazyLlmResponse, ReqResHelper, LazyLlmRequest
from lazyllm import FastapiApp
import pickle
import codecs

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
args = parser.parse_args()

def load_func(f):
    return cloudpickle.loads(base64.b64decode(f.encode('utf-8')))

func = load_func(args.function)
if args.before_function:
    before_func = load_func(args.before_function)
if args.after_function:
    after_func = load_func(args.after_function)


app = FastAPI()
FastapiApp.update()

@app.post("/generate")
async def generate(request: Request): # noqa C901
    try:
        origin = input = (await request.json())
        kw = dict()
        try:
            input = pickle.loads(codecs.decode(input.encode('utf-8'), "base64"))
            assert isinstance(input, LazyLlmRequest)
            kw = input.kwargs
        except Exception: input = origin
        finally: origin = input

        h = ReqResHelper()
        origin = input = h.make_request(input).input
        if args.before_function:
            assert (callable(before_func)), 'before_func must be callable'
            r = inspect.getfullargspec(before_func)
            if isinstance(input, dict) and set(r.args[1:] if r.args[0] == 'self' else r.args) == set(input.keys()):
                assert len(kw) == 0, f'Duplicate kwargs provide, keys are {kw.keys()}'
                input = before_func(**input)
            else:
                input = before_func(input, **kw)
                kw = dict()
        if getattr(getattr(func, '_meta', func.__class__), '__enable_request__', False):
            output = func(h.make_request(input, **kw))
        else:
            output = func(input, **kw)
        output = h.make_request(output).input

        def impl(o):
            o = h.make_response(o, force=True)
            assert isinstance(o, LazyLlmResponse), 'output of func must be LazyLlmResponse'
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

if '__relay_services__' in dir(func.__class__):
    for (method, path), (name, kw) in func.__class__.__relay_services__.items():
        getattr(app, method)(path, **kw)(getattr(func, name))

if __name__ == "__main__":
    uvicorn.run(app, host=args.open_ip, port=args.open_port)
