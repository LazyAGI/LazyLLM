import cloudpickle
import httpx
import uvicorn
import argparse


from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    try:
        json_data = await request.json()
        headers = {'Content-Type': 'application/json'}

        input_json = json_data.copy()
        if args.before_function:
            # json_data = await before_func(json_data)
            json_data = before_func(json_data)

        async with httpx.AsyncClient(timeout=90) as client:
           response = await client.post(args.target_url, json=json_data, headers=headers)

        if args.after_function:
            # res = await after_func(input_json, response.content)
            res = after_func(input_json, response.content)
            response.headers["Content-Length"] = str(len(res))
            return Response(content=res, status_code=response.status_code, headers=response.headers)
        else:
            return Response(content=response.content, status_code=response.status_code, headers=response.headers)

    except Exception as e:
        err_str = str(e)
        return Response(content=err_str, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_url", type=str,
                        help="Url of llm Server")
    parser.add_argument("--open_ip", type=str, default="0.0.0.0", 
                        help="IP: Receive for Client")
    parser.add_argument("--open_port", type=int, default=17782,
                        help="Port: Receive for Client")
    parser.add_argument("--before_function")
    parser.add_argument("--after_function")
    args = parser.parse_args()

    # TODO(search/implement a new encode & decode method)
    if args.before_function:
        exec(f'r = {args.before_function[1:]}')
        b = bytes(r, encoding='raw_unicode_escape')
        before_func = cloudpickle.loads(b)
    if args.after_function:
        exec(f'r = {args.after_function[1:]}')
        b = bytes(r, encoding='raw_unicode_escape')
        after_func = cloudpickle.loads(b)

    uvicorn.run(app, host=args.open_ip, port=args.open_port)