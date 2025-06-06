import atexit
import os
import json
import signal
import socket
import sys
import requests
import traceback
from lazyllm.thirdparty import gradio as gr, PIL
import time
import re
from pathlib import Path
from typing import List, Union

import lazyllm
from lazyllm import LOG, globals, FileSystemQueue, OnlineChatModule, TrainableModule
from lazyllm.components.formatter import decode_query_with_filepaths, encode_query_with_filepaths
from ...module.module import ModuleBase


css = """
#logging {background-color: #FFCCCB}

#module {
  font-family: 'Courier New', Courier, monospace;
  font-size: 16px;
  white-space: pre !important;
}
"""

class WebModule(ModuleBase):
    class Mode:
        Dynamic = 0
        Refresh = 1
        Appendix = 2

    def __init__(self, m, *, components=dict(), title='对话演示终端', port=None,
                 history=[], text_mode=None, trace_mode=None, audio=False, stream=False,
                 files_target=None, static_paths: Union[str, Path, List[str | Path]] = None,
                 encode_files=False) -> None:
        super().__init__()
        # Set the static directory of gradio so that gradio can access local resources in the directory
        if isinstance(static_paths, (str, Path)):
            self._static_paths = [static_paths]
        elif isinstance(static_paths, list) and all(isinstance(p, (str, Path)) for p in static_paths):
            self._static_paths = static_paths
        elif static_paths is None:
            self._static_paths = []
        else:
            raise ValueError(f"static_paths only supported str, path or list types. Not supported {static_paths}")
        self.m = lazyllm.ActionModule(m) if isinstance(m, lazyllm.FlowBase) else m
        self.pool = lazyllm.ThreadPoolExecutor(max_workers=50)
        self.title = title
        self.port = port or range(20500, 20799)
        components = sum([[([k._module_id, k._module_name] + list(v)) for v in vs]
                         for k, vs in components.items()], [])
        self.ckeys = [[c[0], c[2]] for c in components]
        if isinstance(m, (OnlineChatModule, TrainableModule)) and not history:
            history = [m]
        self.history = [h._module_id for h in history]
        if trace_mode:
            LOG.warn('trace_mode is deprecated')
        self.text_mode = text_mode if text_mode else WebModule.Mode.Dynamic
        self.cach_path = self._set_up_caching()
        self.audio = audio
        self.stream = stream
        self.files_target = files_target if isinstance(files_target, list) or files_target is None else [files_target]
        self.encode_files = encode_files
        self.demo = self.init_web(components)
        self.url = None
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _get_all_file_submodule(self):
        if self.files_target: return
        self.files_target = []
        self.for_each(
            lambda x: getattr(x, 'template_message', None),
            lambda x: self.files_target.append(x)
        )

    def _signal_handler(self, signum, frame):
        LOG.info(f"Signal {signum} received, terminating subprocess.")
        atexit._run_exitfuncs()
        sys.exit(0)

    def _set_up_caching(self):
        if 'GRADIO_TEMP_DIR' in os.environ:
            cach_path = os.environ['GRADIO_TEMP_DIR']
        else:
            cach_path = os.path.join(lazyllm.config['temp_dir'], 'gradio_cach')
            os.environ['GRADIO_TEMP_DIR'] = cach_path
        if not os.path.exists(cach_path):
            os.makedirs(cach_path)
        return cach_path

    def init_web(self, component_descs):
        gr.set_static_paths(self._static_paths)
        with gr.Blocks(css=css, title=self.title, analytics_enabled=False) as demo:
            sess_data = gr.State(value={
                'sess_titles': [''],
                'sess_logs': {},
                'sess_history': {},
                'sess_num': 1,
                'curr_sess': '',
                'frozen_query': '',
            })
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        with lazyllm.config.temp('repr_show_child', True):
                            gr.Textbox(elem_id='module', interactive=False, show_label=True,
                                       label="模型结构", value=repr(self.m))
                    with gr.Row():
                        chat_use_context = gr.Checkbox(interactive=True, value=False, label="使用上下文")
                    with gr.Row():
                        stream_output = gr.Checkbox(interactive=self.stream, value=self.stream, label="流式输出")
                        text_mode = gr.Checkbox(interactive=(self.text_mode == WebModule.Mode.Dynamic),
                                                value=(self.text_mode != WebModule.Mode.Refresh), label="追加输出")
                    components = []
                    for _, gname, name, ctype, value in component_descs:
                        if ctype in ('Checkbox', 'Text'):
                            components.append(getattr(gr, ctype)(interactive=True, value=value, label=f'{gname}.{name}'))
                        elif ctype == 'Dropdown':
                            components.append(getattr(gr, ctype)(interactive=True, choices=value,
                                                                 label=f'{gname}.{name}'))
                        else:
                            raise KeyError(f'invalid component type: {ctype}')
                    with gr.Row():
                        dbg_msg = gr.Textbox(show_label=True, label='处理日志',
                                             elem_id='logging', interactive=False, max_lines=10)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
                with gr.Column(scale=6):
                    with gr.Row():
                        add_sess_btn = gr.Button("添加新会话")
                        sess_drpdn = gr.Dropdown(choices=sess_data.value['sess_titles'], label="选择会话：", value='')
                        del_sess_btn = gr.Button("删除当前会话")
                    chatbot = gr.Chatbot(height=700)
                    query_box = gr.MultimodalTextbox(show_label=False, placeholder='输入内容并回车!!!', interactive=True)
                    recordor = gr.Audio(sources=["microphone"], type="filepath", visible=self.audio)

            query_box.submit(self._init_session, [query_box, sess_data, recordor],
                                                 [sess_drpdn, chatbot, dbg_msg, sess_data, recordor], queue=True
                ).then(lambda: gr.update(interactive=False), None, query_box, queue=False
                ).then(lambda: gr.update(interactive=False), None, add_sess_btn, queue=False
                ).then(lambda: gr.update(interactive=False), None, sess_drpdn, queue=False
                ).then(lambda: gr.update(interactive=False), None, del_sess_btn, queue=False
                ).then(self._prepare, [query_box, chatbot, sess_data], [query_box, chatbot], queue=True
                ).then(self._respond_stream, [chat_use_context, chatbot, stream_output, text_mode] + components,
                                             [chatbot, dbg_msg], queue=chatbot
                ).then(lambda: gr.update(interactive=True), None, query_box, queue=False
                ).then(lambda: gr.update(interactive=True), None, add_sess_btn, queue=False
                ).then(lambda: gr.update(interactive=True), None, sess_drpdn, queue=False
                ).then(lambda: gr.update(interactive=True), None, del_sess_btn, queue=False)
            clear_btn.click(self._clear_history, [sess_data], outputs=[chatbot, query_box, dbg_msg, sess_data])

            sess_drpdn.change(self._change_session, [sess_drpdn, chatbot, dbg_msg, sess_data],
                                                    [sess_drpdn, chatbot, query_box, dbg_msg, sess_data])
            add_sess_btn.click(self._add_session, [chatbot, dbg_msg, sess_data],
                                                  [sess_drpdn, chatbot, query_box, dbg_msg, sess_data])
            del_sess_btn.click(self._delete_session, [sess_drpdn, sess_data],
                                                     [sess_drpdn, chatbot, query_box, dbg_msg, sess_data])
            recordor.change(self._sub_audio, recordor, query_box)
            return demo

    def _sub_audio(self, audio):
        if audio:
            return {'text': '', 'files': [audio]}
        else:
            return {}

    def _init_session(self, query, session, audio):
        audio = None
        session['frozen_query'] = query
        if session['curr_sess'] != '':  # remain unchanged.
            return gr.Dropdown(), gr.Chatbot(), gr.Textbox(), session, audio

        if "text" in query and query["text"] is not None:
            id_name = query['text']
        else:
            id_name = id(id_name)
        session['curr_sess'] = f"({session['sess_num']})  {id_name}"
        session['sess_num'] += 1
        session['sess_titles'][0] = session['curr_sess']

        session['sess_logs'][session['curr_sess']] = []
        session['sess_history'][session['curr_sess']] = []
        return gr.update(choices=session['sess_titles'], value=session['curr_sess']), [], '', session, audio

    def _add_session(self, chat_history, log_history, session):
        if session['curr_sess'] == '':
            LOG.warning('Cannot create new session while current session is empty.')
            return gr.Dropdown(), gr.Chatbot(), {}, gr.Textbox(), session

        self._save_history(chat_history, log_history, session)

        session['curr_sess'] = ''
        session['sess_titles'].insert(0, session['curr_sess'])
        return gr.update(choices=session['sess_titles'], value=session['curr_sess']), [], {}, '', session

    def _save_history(self, chat_history, log_history, session):
        if session['curr_sess'] in session['sess_titles']:
            session['sess_history'][session['curr_sess']] = chat_history
            session['sess_logs'][session['curr_sess']] = log_history

    def _change_session(self, session_title, chat_history, log_history, session):
        if session['curr_sess'] == '':  # new session
            return gr.Dropdown(), [], {}, '', session

        if session_title not in session['sess_titles']:
            LOG.warning(f'{session_title} is not an existing session title.')
            return gr.Dropdown(), gr.Chatbot(), {}, gr.Textbox(), session

        self._save_history(chat_history, log_history, session)

        session['curr_sess'] = session_title
        return (gr.update(choices=session['sess_titles'], value=session['curr_sess']),
                session['sess_history'][session['curr_sess']], {},
                session['sess_logs'][session['curr_sess']], session)

    def _delete_session(self, session_title, session):
        if session_title not in session['sess_titles']:
            LOG.warning(f'session {session_title} does not exist.')
            return gr.Dropdown(), session
        session['sess_titles'].remove(session_title)

        if session_title != '':
            del session['sess_history'][session_title]
            del session['sess_logs'][session_title]
            session['curr_sess'] = session_title
        else:
            session['curr_sess'] = 'dummy session'
            # add_session and change_session cannot accept an uninitialized session.
            # Here we need to imitate removal of a real session so that
            # add_session and change_session could skip saving chat history.

        if len(session['sess_titles']) == 0:
            return self._add_session(None, None, session)
        else:
            return self._change_session(session['sess_titles'][0], None, {}, session)

    def _prepare(self, query, chat_history, session):
        if not query.get('text', '') and not query.get('files', []):
            query = session['frozen_query']
        if chat_history is None:
            chat_history = []
        for x in query["files"]:
            chat_history.append([[x,], None])
        if "text" in query and query["text"]:
            chat_history.append([query['text'], None])
        return {}, chat_history

    def _respond_stream(self, use_context, chat_history, stream_output, append_text, *args):  # noqa C901
        try:
            # TODO: move context to trainable module
            files = []
            chat_history[-1][1], log_history = '', []
            for file in chat_history[::-1]:
                if file[-1]: break  # not current chat
                if isinstance(file[0], (tuple, list)):
                    files.append(file[0][0])
                elif isinstance(file[0], str) and file[0].startswith('lazyllm_img::'):  # Just for pytest
                    files.append(file[0][13:])
            if isinstance(chat_history[-1][0], str):
                string = chat_history[-1][0]
            else:
                string = ''
            if self.files_target is None and not self.encode_files:
                self._get_all_file_submodule()
            if self.encode_files and files:
                string = encode_query_with_filepaths(string, files)
            if files and self.files_target:
                for module in self.files_target:
                    assert isinstance(module, ModuleBase)
                    if module._module_id in globals['lazyllm_files']:
                        globals['lazyllm_files'][module._module_id].extend(files)
                    else:
                        globals['lazyllm_files'][module._module_id] = files
                string += f' ## Get attachments: {os.path.basename(files[-1])}'
            elif self.files_target:
                for module in self.files_target:
                    assert isinstance(module, ModuleBase)
                    globals['lazyllm_files'][module._module_id] = []
            input = string
            history = chat_history[:-1] if use_context and len(chat_history) > 1 else list()

            for k, v in zip(self.ckeys, args):
                if k[0] not in globals['global_parameters']: globals['global_parameters'][k[0]] = dict()
                globals['global_parameters'][k[0]][k[1]] = v

            if use_context:
                for h in self.history:
                    if h not in globals['chat_history']: globals['chat_history'][h] = list()
                    globals['chat_history'][h] = history

            if FileSystemQueue().size() > 0: FileSystemQueue().clear()
            kw = dict(stream_output=stream_output) if isinstance(self.m, (TrainableModule, OnlineChatModule)) else {}
            func_future = self.pool.submit(self.m, input, **kw)
            while True:
                if value := FileSystemQueue().dequeue():
                    chat_history[-1][1] += ''.join(value) if append_text else ''.join(value)
                    if stream_output: yield chat_history, ''
                elif value := FileSystemQueue.get_instance('lazy_error').dequeue():
                    log_history.append(''.join(value))
                elif value := FileSystemQueue.get_instance('lazy_trace').dequeue():
                    log_history.append(''.join(value))
                elif func_future.done(): break
                time.sleep(0.01)
            result = func_future.result()
            if FileSystemQueue().size() > 0: FileSystemQueue().clear()

            def get_log_and_message(s):
                if isinstance(s, dict):
                    s = s.get("message", {}).get("content", "")
                else:
                    try:
                        r = decode_query_with_filepaths(s)
                        if isinstance(r, str):
                            r = json.loads(r)
                        if 'choices' in r:
                            if "type" not in r["choices"][0] or (
                                    "type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                                delta = r["choices"][0]["delta"]
                                if "content" in delta:
                                    s = delta["content"]
                                else:
                                    s = ""
                        elif isinstance(r, dict) and 'files' in r and 'query' in r:
                            return r['query'], ''.join(log_history), r['files'] if len(r['files']) > 0 else None
                        else:
                            s = s
                    except (ValueError, KeyError, TypeError):
                        s = s
                    except Exception as e:
                        LOG.error(f"Uncaptured error `{e}` when parsing `{s}`, please contact us if you see this.")
                return s, "".join(log_history), None

            def contains_markdown_image(text: str):
                pattern = r"!\[.*?\]\((.*?)\)"
                return bool(re.search(pattern, text))

            def extract_img_path(text: str):
                pattern = r"!\[.*?\]\((.*?)\)"
                urls = re.findall(pattern, text)
                return urls

            file_paths = None
            if isinstance(result, (str, dict)):
                result, log, file_paths = get_log_and_message(result)
            if file_paths:
                for i, file_path in enumerate(file_paths):
                    suffix = os.path.splitext(file_path)[-1].lower()
                    file = None
                    if suffix in PIL.Image.registered_extensions().keys():
                        file = gr.Image(file_path)
                    elif suffix in ('.mp3', '.wav'):
                        file = gr.Audio(file_path)
                    elif suffix in ('.mp4'):
                        file = gr.Video(file_path)
                    else:
                        LOG.error(f'Not supported typr: {suffix}, for file: {file}')
                    if i == 0:
                        chat_history[-1][1] = file
                    else:
                        chat_history.append([None, file])
                if result:
                    chat_history.append([None, result])
            else:
                assert isinstance(result, str), f'Result should only be str, but got {type(result)}'
                show_result = result
                if contains_markdown_image(show_result):
                    urls = extract_img_path(show_result)
                    for url in urls:
                        suffix = os.path.splitext(url)[-1].lower()
                        if suffix in PIL.Image.registered_extensions().keys() and os.path.exists(url):
                            show_result = show_result.replace(url, "file=" + url)
                if result:
                    count = (len(match.group(1)) if (match := re.search(r'(\n+)$', result)) else 0) + len(result) + 1
                    if not (result in chat_history[-1][1][-count:]):
                        chat_history[-1][1] += "\n\n" + show_result
                    elif show_result != result:
                        chat_history[-1][1] = chat_history[-1][1].replace(result, show_result)
        except requests.RequestException as e:
            chat_history = None
            log = str(e)
        except Exception as e:
            chat_history = None
            log = f'{str(e)}\n--- traceback ---\n{traceback.format_exc()}'
            LOG.error(log)
        globals['chat_history'].clear()
        yield chat_history, log

    def _clear_history(self, session):
        session['sess_history'][session['curr_sess']] = []
        session['sess_logs'][session['curr_sess']] = []
        return [], {}, '', session

    def _work(self):
        if isinstance(self.port, (range, tuple, list)):
            port = self._find_can_use_network_port()
        else:
            port = self.port
            assert self._verify_port_access(port), f'port {port} is occupied'

        self.url = f'http://127.0.0.1:{port}'
        self.broadcast_url = f'http://0.0.0.0:{port}'

        self.demo.queue().launch(server_name="0.0.0.0", server_port=port, prevent_thread_lock=True)
        LOG.success('LazyLLM webmodule launched successfully: Running on: '
                    f'{self.broadcast_url}, local URL: {self.url}', flush=True)

    def _update(self, *, mode=None, recursive=True):
        super(__class__, self)._update(mode=mode, recursive=recursive)
        self._work()
        return self

    def wait(self):
        self.demo.block_thread()

    def stop(self):
        if self.demo:
            self.demo.close()
            del self.demo
            self.demo, self.url = None, ''

    @property
    def status(self):
        return 'running' if self.url else 'waiting' if self.url is None else 'Cancelled'

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Web', name=self._module_name, subs=[repr(self.m)])

    def _find_can_use_network_port(self):
        for port in self.port:
            if self._verify_port_access(port):
                return port
        raise RuntimeError(
            f'The ports in the range {self.port} are all occupied. '
            'Please change the port range or release the relevant ports.'
        )

    def _verify_port_access(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('127.0.0.1', port))
            return result != 0
