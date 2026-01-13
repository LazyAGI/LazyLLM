import os
import time
import argparse
import asyncio

import lazyllm


def deploy(commands):
    if commands and commands[0] == 'mcp_server':
        commands = commands[1:]
        parser = argparse.ArgumentParser(
            description='lazyllm deploy command for deploying an MCP server.',
            epilog=(
                'Examples:\n'
                '  lazyllm deploy mcp_server uvx mcp-server-fetch\n'
                '  lazyllm deploy mcp_server -e GITHUB_PERSONAL_ACCESS_TOKEN your_token '
                '--sse-port 8080 npx -- -y @modelcontextprotocol/server-github'
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument('command', help='Command to spawn the server. Do not provide an HTTP URL.')
        parser.add_argument('args', nargs='*', help='Extra arguments for the command to spawn the server')
        parser.add_argument('-e', '--env', nargs=2, action='append', metavar=('KEY', 'VALUE'),
                            help='Environment variables for spawning the server. Can be used multiple times.',
                            default=[])
        parser.add_argument('--pass-environment', action=argparse.BooleanOptionalAction,
                            help='Pass through all environment variables when spawning the server.',
                            default=False)
        parser.add_argument('--sse-port', type=int, default=0,
                            help='Port to expose an SSE server on. Default is a random port')
        parser.add_argument('--sse-host', default='127.0.0.1',
                            help='Host to expose an SSE server on. Default is 127.0.0.1')
        parser.add_argument(
            '--allow-origin',
            nargs='+',
            default=[],
            help='Allowed origins for the SSE server. Can be used multiple times. Default is no CORS allowed.'
        )
        args = parser.parse_args(commands)

        env = {}
        if args.pass_environment:
            env.update(os.environ)
        env.update(dict(args.env))

        lazyllm.LOG.debug('Starting stdio client and SSE server')

        from lazyllm.tools.mcp.deploy import SseServerSettings
        client = lazyllm.tools.MCPClient(command_or_url=args.command, args=args.args, env=env)
        asyncio.run(
            client.deploy(
                sse_settings=SseServerSettings(
                    bind_host=args.sse_host,
                    port=args.sse_port,
                    allow_origins=args.allow_origin,
                )
            )
        )
    elif commands and commands[0] == 'mineru':
        commands = commands[1:]
        parser = argparse.ArgumentParser(description='lazyllm deploy command for deploying a mineru server.')
        parser.add_argument('--port', help='Port for the mineru server.', default=None)
        parser.add_argument('--cache_dir', help='Cache directory for the mineru server.', default=None)
        parser.add_argument('--image_save_dir', help='Image save directory for the mineru server.', default=None)
        parser.add_argument('--default_backend', help='Default backend for the mineru server.\
            (pipeline|vlm-vllm-async-engine|vlm-transformers)', default='pipeline')
        parser.add_argument('--model_source', help='Model source for the mineru server.(huggingface|modelscope)',
                            default='huggingface')
        args = parser.parse_args(commands)
        os.environ['MINERU_MODEL_SOURCE'] = args.model_source

        lazyllm.LOG.info('Starting mineru server')
        lazyllm.LOG.info(f'Current model source: {args.model_source}')
        from lazyllm.tools.servers.mineru.mineru_server_module import MineruServer
        server = MineruServer(
            cache_dir=args.cache_dir,
            image_save_dir=args.image_save_dir,
            default_backend=args.default_backend,
            port=args.port,
        )
        server.start()
        server.wait()
    elif commands and commands[0] == 'graphrag':
        commands = commands[1:]
        parser = argparse.ArgumentParser(description='lazyllm deploy command for deploying a graphrag server.')
        parser.add_argument('--kg_dir', help='graphrag knowledge graph directory', required=True, default=None)
        args = parser.parse_args(commands)

        lazyllm.LOG.info('Starting graphrag server')
        from lazyllm.tools.servers.graphrag_server_module import GraphRagServerModule
        server = GraphRagServerModule(kg_dir=args.kg_dir)
        server.start()
        server.wait()
    else:
        parser = argparse.ArgumentParser(description='lazyllm deploy command for deploying a model.')
        parser.add_argument('model', help='model name')
        parser.add_argument('--framework', help='deploy framework', default='auto',
                            choices=['auto', 'vllm', 'lightllm', 'lmdeploy', 'infinity', 'embedding', 'mindie'])
        parser.add_argument('--chat', help='chat ', default='false',
                            choices=['ON', 'on', '1', 'true', 'True', 'OFF', 'off', '0', 'False', 'false'])

        args, unknown = parser.parse_known_args(commands)
        kwargs = {'options_keys': []}
        for arg in unknown:
            assert arg.startswith('--'), f'Argument {arg} must start with --'
            key = arg[2:]
            if '=' in key:
                key, value = key.split('=', 1)
                kwargs[key] = value
            else:
                kwargs['options_keys'].append(key)

        lazyllm.LOG.debug(f'Use arguments: {kwargs}')
        t = lazyllm.TrainableModule(args.model).deploy_method(getattr(lazyllm.deploy, args.framework), **kwargs)
        if args.chat in ['ON', 'on', '1', 'true', 'True']:
            t = lazyllm.WebModule(t)
        t.start()
        if args.chat in ['ON', 'on', '1', 'true', 'True']:
            t.wait()
        else:
            lazyllm.LOG.success(f'LazyLLM TrainableModule launched successfully:\n  URL: {t._url}\n  '
                                f'Framework: {t._deploy_type.__name__}')
            while True:
                time.sleep(10)
