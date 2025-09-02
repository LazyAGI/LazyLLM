from lazyllm.module import ServerModule
from lazyllm.components.deploy.mineru.mineru_server_module import MineruServerBase

class MineruServer(ServerModule):
    def __init__(self,
                 cache_dir: str = None,
                 image_save_dir: str = None,
                 default_backend: str = 'pipeline',
                 default_lang: str = 'ch_server',
                 default_parse_method: str = 'auto',
                 default_formula_enable: bool = True,
                 default_table_enable: bool = True,
                 default_return_md: bool = False,
                 default_return_content_list: bool = True,
                 *args, **kwargs):
        mineru_server = MineruServerBase(
            cache_dir=cache_dir, image_save_dir=image_save_dir, default_backend=default_backend,
            default_lang=default_lang, default_parse_method=default_parse_method,
            default_formula_enable=default_formula_enable, default_table_enable=default_table_enable,
            default_return_md=default_return_md, default_return_content_list=default_return_content_list)
        super().__init__(mineru_server, *args, **kwargs)
