import cloudpickle
import httpx
import uvicorn
import argparse
import base64
import os
import sys
import copy

from fastapi import FastAPI, Request
from fastapi.responses import Response

# TODO(sunxiaoye): delete in the future
lazyllm_module_dir=os.path.abspath(__file__)
for _ in range(5):
    lazyllm_module_dir = os.path.dirname(lazyllm_module_dir)
sys.path.append(lazyllm_module_dir)

app = FastAPI()


@app.post("/generate")
async def generate(request: Request):
    try:
        input = await request.json()
        input = input['input']
        origin = input

        if args.before_function:
            assert(callable(before_func)), 'before_func must be callable'
            input = before_func(input)
        output = func(input)
        if args.after_function:
            assert(callable(after_func)), 'after_func must be callable'
            output = after_func(origin, output)
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