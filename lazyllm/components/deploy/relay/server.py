from lazyllm.common.utils import str2obj
import uvicorn
import argparse
import os
import sys
import inspect
import traceback
from types import GeneratorType
import lazyllm
from lazyllm import kwargs, package, load_obj
from lazyllm import FastapiApp, globals
from lazyllm.common import _trim_traceback, _register_trim_module
import time
import pickle
import codecs
import asyncio
import functools
from functools import partial
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import requests

# TODO(sunxiaoye): delete in the future
lazyllm_module_dir = os.path.abspath(__file__)
for _ in range(5):
    lazyllm_module_dir = os.path.dirname(lazyllm_module_dir)
sys.path.append(lazyllm_module_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--open_ip', type=str, default='0.0.0.0',
                    help='IP: Receive for Client')
parser.add_argument('--open_port', type=int, default=17782,
                    help='Port: Receive for Client')
parser.add_argument('--function', required=True)
parser.add_argument('--before_function')
parser.add_argument('--after_function')
parser.add_argument('--pythonpath')
parser.add_argument('--num_replicas', type=int, default=1, help='num of ray replicas')
parser.add_argument('--security_key', type=str, default=None, help='security key')
parser.add_argument('--defined_pos', type=str, default=None, help='user defined positional')
args = parser.parse_args()

if args.pythonpath:
    sys.path.append(args.pythonpath)

func = load_obj(args.function)
if args.before_function:
    before_func = load_obj(args.before_function)
if args.after_function:
    after_func = load_obj(args.after_function)


_register_trim_module({'__main__': ['async_wrapper', 'impl']})
_err_msg = ('service of ServerModule execuate failed.\n\nThe above exception was the direct cause '
            'of the following exception in service of ServerModule')
_err_msg += (f' defined at `{load_obj(args.defined_pos)}`' if args.defined_pos else '') + ':\n'


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

def security_check(f: Callable):
    @functools.wraps(f)
    async def wrapper(request: Request):
        if args.security_key and args.security_key != request.headers.get('Security-Key'):
            return Response(content='Authentication failed', status_code=401)
        return (await f(request)) if inspect.iscoroutinefunction(f) else f(request)
    return wrapper

@app.post('/_call')
@security_check
async def lazyllm_call(request: Request):
    try:
        fname, args, kwargs = await request.json()
        args, kwargs = load_obj(args), load_obj(kwargs)
        r = getattr(func, fname)(*args, **kwargs)
        return Response(content=codecs.encode(pickle.dumps(r), 'base64'))
    except requests.RequestException as e:
        return Response(content=f'{str(e)}', status_code=500)
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        formatted = ''.join(traceback.format_exception(exc_type, exc_value, _trim_traceback(exc_tb)))
        return Response(content=f'{_err_msg}\n{formatted}', status_code=500)

@app.post('/generate')
@security_check
async def generate(request: Request): # noqa C901
    try:
        input, kw = (await request.json()), {}
        try:
            input, kw = str2obj(input)
        except Exception: pass
        origin = input

        # TODO(wangzhihong): `update` should come after the `await`, otherwise it may cause strange errors.
        #                    The reason is that when multiple coroutines use the same Session-ID, the function
        #                    clears the globals at the end, which causes some coroutines to mistakenly remove
        #                    data from globals after finishing execution.
        globals._init_sid(request.headers.get('Session-ID'))
        globals.unpickle_and_update_data(request.headers.get('Global-Parameters'))

        if args.before_function:
            assert (callable(before_func)), 'before_func must be callable'
            r = inspect.getfullargspec(before_func)
            if isinstance(input, kwargs) or (
                    isinstance(input, dict) and set(r.args[1:] if r.args[0] == 'self' else r.args) == set(input.keys())):
                assert len(kw) == 0, 'Cannot provide kwargs-input and kwargs at the same time'
                input = before_func(**input)
            else:
                input = before_func(*input, **kw) if isinstance(input, package) else before_func(input, **kw)
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
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        formatted = ''.join(traceback.format_exception(exc_type, exc_value, _trim_traceback(exc_tb)))
        return Response(content=f'{_err_msg}\n{formatted}', status_code=500)
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

class _Dummy: pass

if __name__ == '__main__':
    if lazyllm.config['use_ray']:
        import ray
        from ray import serve
        ray.init()
        _Dummy = serve.deployment(serve.ingress(app)(_Dummy), num_replicas=args.num_replicas)
        serve.start(http_options={'host': args.open_ip, 'port': args.open_port})
        serve.run(_Dummy.bind())

        printed = False
        while True:
            if not printed:
                status = serve.status()
                app_status = status.applications.get('default')
                if app_status and app_status.status == 'RUNNING':
                    lazyllm.LOG.success(f'Deployment is ready on {args.open_ip}:{args.open_port}!')
                    printed = True
            time.sleep(2)
    else:
        uvicorn.run(app, host=args.open_ip, port=args.open_port)
