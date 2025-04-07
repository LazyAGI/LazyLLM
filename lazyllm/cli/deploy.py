import os
import sys
import time
import argparse
import asyncio

import lazyllm
from lazyllm.common import str2bool


def deploy(commands):
    parser = argparse.ArgumentParser(
        description=(
            "lazyllm deploy command for deploying a model or a mcp server."
        ),
        epilog=(
            "Examples:\n"
            "lazyllm deploy model internlm2-chat-20b\n"
            "lazyllm deploy model internlm2-chat-20b --framework vllm\n"
            "lazyllm deploy mcp_server uvx mcp-server-fetch\n"
            "lazyllm deploy mcp_server -e GITHUB_PERSONAL_ACCESS_TOKEN your_token "
            "--sse-port 8080 npx -- -y @modelcontextprotocol/server-github"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="deploy_type", required=True, help="Deployment type")

    # subcommand: deploy a model
    model_parser = subparsers.add_parser("model", help="Deploy a model")
    model_parser.add_argument("model", help="Model name (for model deployment)")
    model_parser.add_argument("--framework", help="Deploy framework", default="auto",
                              choices=["auto", "vllm", "lightllm", "lmdeploy"])
    model_parser.add_argument("--chat", help="Enable chat", default=False, type=str2bool)

    # subcommand: deploy an MCP server
    mcp_parser = subparsers.add_parser("mcp_server", help="Deploy an MCP server")
    mcp_parser.add_argument("command", help="Command to spawn the server. Do not provide an HTTP URL.")
    mcp_parser.add_argument("args", nargs="*", help="Extra arguments for the command to spawn the server")
    mcp_parser.add_argument("-e", "--env", nargs=2, action="append", metavar=("KEY", "VALUE"),
                            help="Environment variables for spawning the server. Can be used multiple times.",
                            default=[])
    mcp_parser.add_argument("--pass-environment", action=argparse.BooleanOptionalAction,
                            help="Pass through all environment variables when spawning the server.",
                            default=False)
    mcp_parser.add_argument("--sse-port", type=int, default=0,
                            help="Port to expose an SSE server on. Default is a random port")
    mcp_parser.add_argument("--sse-host", default="127.0.0.1",
                            help="Host to expose an SSE server on. Default is 127.0.0.1")
    mcp_parser.add_argument(
        "--allow-origin",
        nargs="+",
        default=[],
        help="Allowed origins for the SSE server. Can be used multiple times. Default is no CORS allowed."
    )

    args = parser.parse_args(commands)

    if args.deploy_type == "model":
        # deploy a model
        deploy_framework = getattr(lazyllm.deploy, args.framework)
        t = lazyllm.TrainableModule(args.model).deploy_method(deploy_framework)
        if args.chat:
            t = lazyllm.WebModule(t)
        t.start()
        if args.chat:
            t.wait()
        else:
            lazyllm.LOG.success(
                (
                    f"LazyLLM TrainableModule launched successfully:\n"
                    f"  URL: {t._url}\n"
                    f"  Framework: {t._deploy_type.__name__}"
                ),
                flush=True,
            )
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                sys.exit(0)

    elif args.deploy_type == "mcp_server":
        # deploy an MCP server
        env: dict[str, str] = {}
        if args.pass_environment:
            env.update(os.environ)
        env.update(dict(args.env))

        lazyllm.LOG.debug("Starting stdio client and SSE server")

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
    else:
        parser.error(f"Unsupported deploy type: {args.deploy_type}")
