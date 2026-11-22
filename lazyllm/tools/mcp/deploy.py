from lazyllm.thirdparty import uvicorn

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Any, Optional, List

from lazyllm.thirdparty import mcp

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.routing import Mount, Route


@dataclass
class SseServerSettings:
    '''Settings for the SSE server.'''
    bind_host: str
    port: int
    allow_origins: Optional[List[str]] = None
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'


def _create_starlette_app(mcp_server, *, allow_origins=None, debug=False) -> Starlette:
    '''
    Create a Starlette application to serve the provided MCP server with SSE.

    Args:
        mcp_server: The MCP server instance.
        allow_origins: Allowed origins for CORS middleware.
        debug: Flag indicating whether to enable debug mode.

    Returns:
        A configured Starlette application.
    '''
    sse = mcp.server.sse.SseServerTransport('/messages/')

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    middleware: List[Middleware] = []
    if allow_origins:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_methods=['*'],
                allow_headers=['*'],
            )
        )

    return Starlette(
        debug=debug,
        middleware=middleware,
        routes=[
            Route('/sse', endpoint=handle_sse),
            Mount('/messages/', app=sse.handle_post_message),
        ],
    )


async def _create_proxy_server(remote_app): # noqa C901
    '''
    Create a proxy server instance based on a remote client session.

    Args:
        remote_app: A client session for a remote MCP application.

    Returns:
        An instance of Server with request and notification handlers mapped.
    '''
    response = await remote_app.initialize()
    capabilities = response.capabilities

    # mcp 2.x removed Server.request_handlers/notification_handlers (handlers are passed to the constructor as
    # on_* callbacks taking (ctx, params)), no longer wraps results in ServerResult, and uses snake_case fields.
    is_v2 = hasattr(mcp.server.Server, 'add_request_handler')
    wrap_result = (lambda result: result) if is_v2 else mcp.types.ServerResult
    v2_handlers = {}
    if not is_v2:
        server_instance: mcp.server.Server[Any] = mcp.server.Server(name=response.serverInfo.name)  # noqa NID003

    def register(request_type, v2_name, handler):
        if is_v2:
            async def v2_handler(ctx, params):
                return await handler(SimpleNamespace(params=params))
            v2_handlers[v2_name] = v2_handler
        elif request_type is mcp.types.ProgressNotification:
            server_instance.notification_handlers[request_type] = handler
        else:
            server_instance.request_handlers[request_type] = handler

    if capabilities.prompts:
        async def _list_prompts(_: Any) -> mcp.types.ServerResult:
            result = await remote_app.list_prompts()
            return wrap_result(result)

        async def _get_prompt(req: mcp.types.GetPromptRequest) -> mcp.types.ServerResult:
            result = await remote_app.get_prompt(req.params.name, req.params.arguments)
            return wrap_result(result)

        register(mcp.types.ListPromptsRequest, 'on_list_prompts', _list_prompts)
        register(mcp.types.GetPromptRequest, 'on_get_prompt', _get_prompt)

    if capabilities.resources:
        async def _subscribe_resource(req: mcp.types.SubscribeRequest) -> mcp.types.ServerResult:
            await remote_app.subscribe_resource(req.params.uri)
            return wrap_result(mcp.types.EmptyResult())

        async def _unsubscribe_resource(req: mcp.types.UnsubscribeRequest) -> mcp.types.ServerResult:
            await remote_app.unsubscribe_resource(req.params.uri)
            return wrap_result(mcp.types.EmptyResult())

        async def _list_resources(_: Any) -> mcp.types.ServerResult:
            result = await remote_app.list_resources()
            return wrap_result(result)

        async def _read_resource(req: mcp.types.ReadResourceRequest) -> mcp.types.ServerResult:
            result = await remote_app.read_resource(req.params.uri)
            return wrap_result(result)

        register(mcp.types.SubscribeRequest, 'on_subscribe_resource', _subscribe_resource)
        register(mcp.types.UnsubscribeRequest, 'on_unsubscribe_resource', _unsubscribe_resource)
        register(mcp.types.ListResourcesRequest, 'on_list_resources', _list_resources)
        register(mcp.types.ReadResourceRequest, 'on_read_resource', _read_resource)

    if capabilities.logging:
        async def _set_logging_level(req: mcp.types.SetLevelRequest) -> mcp.types.ServerResult:
            await remote_app.set_logging_level(req.params.level)
            return wrap_result(mcp.types.EmptyResult())

        register(mcp.types.SetLevelRequest, 'on_set_logging_level', _set_logging_level)

    if capabilities.tools:
        async def _list_tools(_: Any) -> mcp.types.ServerResult:
            tools = await remote_app.list_tools()
            return wrap_result(tools)

        async def _call_tool(req: mcp.types.CallToolRequest) -> mcp.types.ServerResult:
            try:
                result = await remote_app.call_tool(
                    req.params.name,
                    req.params.arguments or {},
                )
                return wrap_result(result)
            except Exception as e:
                return wrap_result(
                    mcp.types.CallToolResult(
                        content=[mcp.types.TextContent(type='text', text=str(e))],
                        isError=True,
                    )
                )

        register(mcp.types.ListToolsRequest, 'on_list_tools', _list_tools)
        register(mcp.types.CallToolRequest, 'on_call_tool', _call_tool)

    async def _send_progress_notification(req: mcp.types.ProgressNotification) -> None:
        await remote_app.send_progress_notification(
            req.params.progress_token if is_v2 else req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    register(mcp.types.ProgressNotification, 'on_progress', _send_progress_notification)

    async def _complete(req: mcp.types.CompleteRequest) -> mcp.types.ServerResult:
        result = await remote_app.complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return wrap_result(result)

    register(mcp.types.CompleteRequest, 'on_completion', _complete)

    if is_v2:
        return mcp.server.Server(response.server_info.name, **v2_handlers)
    return server_instance


async def start_sse_server(
    client_session,
    sse_settings: SseServerSettings,
) -> None:
    '''
    Start the SSE server by creating a proxy MCP server and serving it via Starlette.

    Args:
        client_session: The client session for the remote MCP app.
        sse_settings: The settings for configuring the SSE server.
    '''
    mcp_server = await _create_proxy_server(client_session)

    # Create the Starlette app with SSE routes and middleware.
    starlette_app = _create_starlette_app(
        mcp_server,
        allow_origins=sse_settings.allow_origins,
        debug=(sse_settings.log_level == 'DEBUG'),
    )

    # Configure and start the HTTP server using uvicorn.
    config = uvicorn.Config(
        starlette_app,
        host=sse_settings.bind_host,
        port=sse_settings.port,
        log_level=sse_settings.log_level.lower(),
    )
    http_server = uvicorn.Server(config)
    await http_server.serve()
