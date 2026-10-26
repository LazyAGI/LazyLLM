import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import lazyllm


@pytest.fixture
def endpoint():  # noqa: C901 - scripted local HTTP scenarios.
    seen, release, received = [], threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            received.set()
            if body['model'] == 'slow-head':
                release.wait(3)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body['stream'] else 'application/json')
            self.end_headers()
            try:
                if body['model'] in ('slow-stream', 'long-stream'):
                    count = 0
                    while not release.wait(0.02):
                        frame = {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'partial '},
                                              'finish_reason': None}]}
                        self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                        self.wfile.flush()
                        count += 1
                        if body['model'] == 'long-stream' and count == 12:
                            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
                            break
                elif body['model'] == 'slow-body':
                    self.wfile.write(b'{')
                    self.wfile.flush()
                    release.wait(3)
                elif body['model'] == 'slow-read':
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n')
                    self.wfile.flush()
                    release.wait(3)
                elif self.path.endswith('/messages'):
                    finish = 'max_tokens' if body['model'] == 'length' else 'end_turn'
                    if body['stream']:
                        frames = [
                            {'type': 'message_start', 'message': {'usage': {'input_tokens': 11}}},
                            {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'prepared'}},
                            {'type': 'message_delta', 'delta': {'stop_reason': finish}, 'usage': {'output_tokens': 7}},
                            {'type': 'message_stop'},
                        ]
                        for frame in frames:
                            encoded = 'event: ' + frame['type'] + '\ndata: ' + json.dumps(frame) + '\n\n'
                            self.wfile.write(encoded.encode())
                    else:
                        self.wfile.write(json.dumps({
                            'type': 'message', 'content': [{'type': 'text', 'text': 'prepared'}],
                                                    'stop_reason': finish,
                                                    'usage': {'input_tokens': 11, 'output_tokens': 7}}).encode())
                else:
                    frame = {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'prepared'},
                                          'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 11, 'completion_tokens': 7}}
                    self.wfile.write((('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n') if body['stream']
                                      else json.dumps(frame)).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/', seen, received, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def model(endpoint, *, source='openai', name='gpt-test', stream=False):
    return lazyllm.OnlineChatModule(source=source, model=name, url=endpoint[0], api_key='fixture-key', stream=stream)
