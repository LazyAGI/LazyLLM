from lazyllm.thirdparty import cloudpickle
import httpx
import uvicorn
import argparse
import base64
import os
import sys
import inspect

from fastapi import FastAPI, Request
from fastapi.responses import Response

# TODO(sunxiaoye): delete in the future
lazyllm_module_dir=os.path.abspath(__file__)
for _ in range(5):
    lazyllm_module_dir = os.path.dirname(lazyllm_module_dir)
sys.path.append(lazyllm_module_dir)

app = FastAPI()

prompt = '{input}'
response_split = None

@app.post("/prompt")
async def set_prompt(request: Request):
    try:
        input = await request.json()
        global prompt, response_split
        prompt, response_split = input['prompt'], input['response_split']
        return Response(content='set prompt done!')
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.post("/generate")
async def generate(request: Request):
    try:
        origin = input = (await request.json())

        if args.before_function:
            assert(callable(before_func)), 'before_func must be callable'
            input = before_func(**input)
        output = func(prompt.format(**input))
        if response_split is not None:
            output = output.split(response_split)[-1]
        if args.after_function:
            assert(callable(after_func)), 'after_func must be callable'
            r = inspect.getfullargspec(after_func)
            assert len(r.args) > 0 and r.varargs is None and r.varkw is None
            # TODO(wangzhihong): specify functor and real function
            new_args = r.args[1:] if r.args[0] == 'self' else r.args
            if len(new_args) == 1:
                output = after_func(output) if len(r.kwonlyargs) == 0 else \
                         after_func(output, **{r.kwonlyargs[0]: origin}) 
            elif len(new_args) == 2:
                output = after_func(output, origin)
        return Response(content=output)

    except Exception as e:
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_ip", type=str, default="0.0.0.0", 
                        help="IP: Receive for Client")
    parser.add_argument("--open_port", type=int, default=17782,
                        help="Port: Receive for Client")
    parser.add_argument("--function", required=True)
    parser.add_argument("--before_function")
    parser.add_argument("--after_function")
    args = parser.parse_args()

    # TODO(search/implement a new encode & decode method)
    def load_func(f):
        return cloudpickle.loads(base64.b64decode(f.encode('utf-8')))

    func = load_func(args.function)
    if args.before_function:
        before_func = load_func(args.before_function)
    if args.after_function:
        after_func = load_func(args.after_function)

    uvicorn.run(app, host=args.open_ip, port=args.open_port)