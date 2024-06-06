import os
import socket
import requests
import multiprocessing
import json

import lazyllm
from lazyllm import ModuleBase, ServerModule
import gradio as gr
from lazyllm.flow import Pipeline

class WebUi():
    def __init__(self, base_url) -> None:
        self.base_url = base_url

    def basic_headers(self, content_type=True):
        return {"accept": "application/json", "Content-Type": "application/json" if content_type else None}

    def muti_headers(self, ):
        return {"accept": "application/json"}

    def post_request(self, url, data):
        response = requests.post(url, headers=self.basic_headers(), data=json.dumps(data))
        return response.json()

    def get_request(self, url):
        response = requests.get(url, headers=self.basic_headers(False))
        return response.json()

    def new_group(self, group_name: str):
        response = requests.post(f"{self.base_url}/new_group?group_name={group_name}",
                                 headers=self.basic_headers(True))
        return response.json()["msg"]

    def delete_group(self, group_name: str):
        response = requests.post(f"{self.base_url}/delete_group?group_name={group_name}",
                                 headers=self.basic_headers(True))
        return response.json()["msg"]

    def list_groups(self):
        response = requests.get(f"{self.base_url}/list_groups", headers=self.basic_headers(False))
        return response.json()["data"]

    def upload_files(self, group_name: str, override: bool = True):
        response = requests.post(f"{self.base_url}/upload_files?group_name={group_name}&override={override}",
                                 headers=self.basic_headers(True))
        return response.json()["data"]

    def list_files(self, group_name: str):
        response = requests.get(f"{self.base_url}/list_files?group_name={group_name}", headers=self.basic_headers(False))
        return response.json()["data"]

    def delete_file(self, group_name: str, file_name: str):
        response = requests.post(f"{self.base_url}/delete_file?group_name={group_name}&file_name={file_name}",
                                 headers=self.basic_headers(True))
        return response.json()["msg"]

    def gr_show_list(self, str_list: list[str], list_name: str):
        return gr.DataFrame(
            headers=["index", list_name],
            value=[[index, str_list[index]] for index in range(len(str_list))]
        )

    def create_ui(self):
        with gr.Blocks() as demo:
            with gr.Tabs():
                select_group_list = []
                with gr.TabItem("创建分组"):
                    create_group_text = gr.Textbox(label="分组名称：")
                    create_group_btn = gr.Button("创建")

                with gr.TabItem("删除分组"):
                    del_select_group = gr.Dropdown(self.list_groups(), label="选择分组")
                    delete_group_btn = gr.Button("删除")
                    select_group_list.append(del_select_group)

                with gr.TabItem("分组列表"):
                    select_group = self.gr_show_list(self.list_groups(), list_name="group_name")
                    select_group_list.append(select_group)

                with gr.TabItem("上传文件"):
                    def _upload_files(group_name, files):

                        files_to_upload = [
                            ("files", (os.path.basename(file), open(file, "rb")))
                            for file in files
                        ]

                        url = f"{self.base_url}/upload_files?group_name={group_name}&override=true"
                        response = requests.post(url, files=files_to_upload, headers=self.muti_headers())
                        response.raise_for_status()
                        response_data = response.json()
                        gr.Info(str(response_data["data"]))

                        for _, (_, file_obj) in files_to_upload:
                            file_obj.close()

                    select_group = gr.Dropdown(self.list_groups(), label="选择分组")
                    select_group.change(lambda x: x, inputs=select_group, outputs=None)

                    up_files = gr.Files(label="上传文件")
                    up_btn = gr.Button("上传")
                    up_btn.click(
                        _upload_files,
                        inputs=[select_group, up_files],
                        outputs=None,
                    )

                    select_group_list.append(select_group)

                with gr.TabItem("文件列表"):
                    def _list_group_files(group_name):
                        file_list = self.list_files(group_name)
                        return gr.update(value=[[index, file_list[index]] for index in range(len(file_list))])

                    select_group = gr.Dropdown(self.list_groups(), label="选择分组")
                    show_list = self.gr_show_list([], list_name="file_name")
                    select_group.change(fn=_list_group_files, inputs=select_group, outputs=show_list)
                    select_group_list.append(select_group)

                with gr.TabItem("删除文件"):
                    def _list_group_files(group_name):
                        file_list = self.list_files(group_name)
                        return gr.update(choices=file_list)

                    select_group = gr.Dropdown(self.list_groups(), label="选择分组")
                    select_file = gr.Dropdown([], label="选择文件")
                    select_group.change(fn=_list_group_files, inputs=select_group, outputs=select_file)
                    delete_btn = gr.Button("删除")

                    def _delete_file(group_name, file_name):
                        gr.Info(self.delete_file(group_name, file_name))
                        return _list_group_files(group_name)

                    delete_btn.click(fn=_delete_file, inputs=[select_group, select_file], outputs=select_file)
                    select_group_list.append(select_group)

                def _create_group(group_name):
                    gr.Info(self.new_group(group_name))
                    curt_groups = self.list_groups()
                    return [
                        gr.update(choices=curt_groups) if isinstance(select, gr.Dropdown)
                        else gr.update(value=[[index, curt_groups[index]] for index in range(len(curt_groups))])
                        for select in select_group_list
                    ]

                def _del_group(group_name):
                    gr.Info(self.delete_group(group_name))
                    curt_groups = self.list_groups()
                    return [
                        gr.update(choices=curt_groups) if isinstance(select, gr.Dropdown)
                        else gr.update(value=[[index, curt_groups[index]] for index in range(len(curt_groups))])
                        for select in select_group_list
                    ]

                create_group_btn.click(
                    fn=_create_group,
                    inputs=create_group_text,
                    outputs=select_group_list
                )

                delete_group_btn.click(
                    fn=_del_group,
                    inputs=del_select_group,
                    outputs=select_group_list
                )

        return demo

class DocWebModule(ModuleBase):
    class Mode:
        Dynamic = 0
        Refresh = 1
        Appendix = 2

    def __init__(
        self,
        doc_server: ServerModule,
        title='文档管理演示终端',
        port=range(20800, 20999),
        history=[],
        text_mode=None,
        trace_mode=None
    ) -> None:
        super().__init__()
        self.title = title
        self.port = port
        self.history = history
        self.trace_mode = trace_mode if trace_mode else DocWebModule.Mode.Refresh
        self.text_mode = text_mode if text_mode else DocWebModule.Mode.Dynamic
        self.doc_server = doc_server
        self._deploy_flag = lazyllm.once_flag()
        self.api_url = ""
        self.url = ""

    def _prepare(self, query, chat_history):
        if chat_history is None:
            chat_history = []
        return '', chat_history + [[query, None]]

    def _clear_history(self):
        return [], '', ''

    def _work(self):
        if isinstance(self.port, (range, tuple, list)):
            port = self._find_can_use_network_port()
        else:
            port = self.port
            assert self._verify_port_access(port), f'port {port} is occupied'

        def _impl():
            self.demo.queue().launch(server_name='0.0.0.0', server_port=port)

        self.api_url = self.doc_server._url.rsplit('/', 1)[0]
        self.web_ui = WebUi(self.api_url)
        self.demo = self.web_ui.create_ui()
        self.p = multiprocessing.Process(target=_impl)
        self.p.start()
        self.url = f'http://0.0.0.0:{port}'

    def _get_deploy_tasks(self):
        return Pipeline(self._work)

    def _get_post_process_tasks(self):
        return Pipeline(self._print_url)

    def wait(self):
        return self.p.join()

    def _find_can_use_network_port(self):
        for port in self.port:
            if self._verify_port_access(port):
                return port
        raise RuntimeError(
            f'The ports in the range {self.port} are all occupied. '
            'Please change the port range or release the relevant ports.'
        )

    def _print_url(self):
        lazyllm.LOG.success(f'LazyLLM DocWebModule launched successfully: Running on local URL: {self.url}', flush=True)

    def _verify_port_access(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            return result != 0

    def __repr__(self):
        return lazyllm.make_repr('Module', 'DocWebModule')
