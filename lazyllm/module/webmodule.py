import socket
import requests
import traceback
import multiprocessing
from .module import ModuleBase
from lazyllm.thirdparty import gradio as gr
from lazyllm import LazyLlmResponse, LazyLlmRequest
from lazyllm.flow import Pipeline
import lazyllm
from types import GeneratorType


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

    def __init__(self, m, *, components=dict(), title='对话演示终端', port=range(20500,20799),
                 text_mode=None, trace_mode=None) -> None:
        super().__init__()
        self.m = m
        self.title = title
        self.port = port 
        components = sum([[([k._module_id, k._module_name] + list(v)) for v in vs]
                           for k, vs in components.items()], [])
        self.ckeys = [[c[0], c[2]] for c in components]
        self.trace_mode = trace_mode if trace_mode else WebModule.Mode.Refresh
        self.text_mode = text_mode if text_mode else WebModule.Mode.Dynamic
        self.demo = self.init_web(components)

    def init_web(self, component_descs):
        with gr.Blocks(css=css, title=self.title) as demo:
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Textbox(elem_id='module', interactive=False, show_label=True, label="模型结构", value=repr(self.m))
                    with gr.Row():
                        chat_use_context = gr.Checkbox(interactive=True, value=False, label="使用上下文")
                    with gr.Row():
                        stream_output = gr.Checkbox(interactive=True, value=True, label="流式输出")
                        text_mode = gr.Checkbox(interactive=(self.text_mode==WebModule.Mode.Dynamic),
                                                value=(self.text_mode!=WebModule.Mode.Refresh), label="追加输出")
                    components = []
                    for _, gname, name, ctype, value in component_descs:
                        if ctype in ('Checkbox', 'Text'):
                            components.append(getattr(gr, ctype)(interactive=True, value=value, label=f'{gname}.{name}'))
                        else:
                            raise KeyError(f'invalid component type: {ctype}')
                    with gr.Row():
                        dbg_msg = gr.Textbox(show_label=True, label='处理日志', elem_id='logging', interactive=False, max_lines=10)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(height=900)
                    query_box = gr.Textbox(show_label=False, placeholder='输入内容并回车!!!')

            query_box.submit(self._prepare, [query_box, chatbot], [query_box, chatbot], queue=False
                ).then(self._respond_stream, [chat_use_context, chatbot, stream_output, text_mode] + components,
                                             [chatbot, dbg_msg], queue=chatbot
                ).then(lambda: gr.update(interactive=True), None, query_box, queue=False)
            clear_btn.click(self._clear_history, None, outputs=[chatbot, query_box, dbg_msg])
        return demo

    def _prepare(self, query, chat_history):
        if chat_history is None:
            chat_history = []
        return '', chat_history + [[query, None]]
        
    def _respond_stream(self, use_context, chat_history, stream_output, append_text, *args):
        try:
            # TODO: move context to trainable module
            input = ('\<eos\>'.join([f'{h[0]}\<eou\>{h[1]}' for h in chat_history]).rsplit('\<eou\>', 1)[0]
                     if use_context else chat_history[-1][0])

            kwargs = dict()
            for k, v in zip(self.ckeys, args):
                if k[0] not in kwargs:
                    kwargs[k[0]] = dict()
                kwargs[k[0]][k[1]] = v
            result = self.m(LazyLlmRequest(input=input, global_parameters=kwargs))

            log_history = []
            if isinstance(result, (LazyLlmResponse, str)):
                result, log = get_log_and_message(result)

            def get_log_and_message(s):
                if isinstance(s, LazyLlmResponse):
                    if not self.trace_mode == WebModule.Mode.Appendix:
                        log_history.clear()
                    if s.err[0] != 0: log_history.append(s.err[1])
                    if s.trace: log_history.append(s.trace)
                    s = s.messages
                return s, ''.join(log_history)

            if isinstance(result, str):
                chat_history[-1][1] = result
            elif isinstance(result, GeneratorType):
                chat_history[-1][1] = ''
                for s in result:
                    if isinstance(s, (LazyLlmResponse, str)):
                        s, log = get_log_and_message(s)
                    chat_history[-1][1] = (chat_history[-1][1] + s) if append_text else s
                    if stream_output: yield chat_history, log
            else:
                raise TypeError(f'function result should only be LazyLlmResponse or str, but got {type(result)}')
        except requests.RequestException as e:
            chat_history = None
            log = str(e)
        except Exception as e:
            chat_history = None
            log = f'{str(e)}\n--- traceback ---\n{traceback.format_exc()}'
        yield chat_history, log

    def _clear_history(self):
        return [], '', ''

    def _work(self):
        if isinstance(self.port, (range, tuple, list)):
            port = self._find_can_use_network_port()
        else:
            port = self.port
            assert self._verify_port_access(port), f'port {port} is occupied'
        def _impl():    
            self.demo.queue().launch(server_name='localhost', server_port=port)
        self.p = multiprocessing.Process(target=_impl)
        self.p.start()
        self.url = f'http://localhost:{port}'

    def _get_deploy_tasks(self):
        return Pipeline(self._work)

    def wait(self):
        return self.p.join()

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
            result = s.connect_ex(('localhost', port))
            return result != 0