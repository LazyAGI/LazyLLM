import uvicorn

from dataclasses import dataclass
from typing import Literal, Any, Optional, List

from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.client.session import ClientSession

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.routing import Mount, Route


@dataclass
class SseServerSettings:
    """Settings for the SSE server."""
    bind_host: str
    port: int
    allow_origins: Optional[List[str]] = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def _create_starlette_app(
    mcp_server: Server[Any],
    *,
    allow_origins: Optional[List[str]] = None,
    debug: bool = False,
) -> Starlette:
    """
    Create a Starlette application to serve the provided MCP server with SSE.
    
    Args:
        mcp_server: The MCP server instance.
        allow_origins: Allowed origins for CORS middleware.
        debug: Flag indicating whether to enable debug mode.
    
    Returns:
        A configured Starlette application.
    """
    sse = SseServerTransport("/messages/")

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
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )

    return Starlette(
        debug=debug,
        middleware=middleware,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


async def _create_proxy_server(remote_app: ClientSession) -> Server[Any]:
    """
    Create a proxy server instance based on a remote client session.
    
    Args:
        remote_app: A client session for a remote MCP application.
    
    Returns:
        An instance of Server with request and notification handlers mapped.
    """
    response = await remote_app.initialize()
    capabilities = response.capabilities

    server_instance: Server[Any] = Server(name=response.serverInfo.name)

    if capabilities.prompts:
        async def _list_prompts(_: Any) -> types.ServerResult:
            result = await remote_app.list_prompts()
            return types.ServerResult(result)

        async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
            result = await remote_app.get_prompt(req.params.name, req.params.arguments)
            return types.ServerResult(result)

        server_instance.request_handlers[types.ListPromptsRequest] = _list_prompts
        server_instance.request_handlers[types.GetPromptRequest] = _get_prompt

    if capabilities.resources:
        async def _subscribe_resource(req: types.SubscribeRequest) -> types.ServerResult:
            await remote_app.subscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        async def _unsubscribe_resource(req: types.UnsubscribeRequest) -> types.ServerResult:
            await remote_app.unsubscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        async def _list_resources(_: Any) -> types.ServerResult:
            result = await remote_app.list_resources()
            return types.ServerResult(result)

        async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
            result = await remote_app.read_resource(req.params.uri)
            return types.ServerResult(result)

        server_instance.request_handlers[types.SubscribeRequest] = _subscribe_resource
        server_instance.request_handlers[types.UnsubscribeRequest] = _unsubscribe_resource
        server_instance.request_handlers[types.ListResourcesRequest] = _list_resources
        server_instance.request_handlers[types.ReadResourceRequest] = _read_resource

    if capabilities.logging:
        async def _set_logging_level(req: types.SetLevelRequest) -> types.ServerResult:
            await remote_app.set_logging_level(req.params.level)
            return types.ServerResult(types.EmptyResult())

        server_instance.request_handlers[types.SetLevelRequest] = _set_logging_level

    if capabilities.tools:
        async def _list_tools(_: Any) -> types.ServerResult:
            tools = await remote_app.list_tools()
            return types.ServerResult(tools)

        async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
            try:
                result = await remote_app.call_tool(
                    req.params.name,
                    req.params.arguments or {},
                )
                return types.ServerResult(result)
            except Exception as e:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(type="text", text=str(e))],
                        isError=True,
                    )
                )

        server_instance.request_handlers[types.ListToolsRequest] = _list_tools
        server_instance.request_handlers[types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: types.ProgressNotification) -> None:
        await remote_app.send_progress_notification(
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    server_instance.notification_handlers[types.ProgressNotification] = _send_progress_notification

    async def _complete(req: types.CompleteRequest) -> types.ServerResult:
        result = await remote_app.complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return types.ServerResult(result)

    server_instance.request_handlers[types.CompleteRequest] = _complete

    return server_instance


async def start_sse_server(
    client_session: ClientSession,
    sse_settings: SseServerSettings,
) -> None:
    """
    Start the SSE server by creating a proxy MCP server and serving it via Starlette.
    
    Args:
        client_session: The client session for the remote MCP app.
        sse_settings: The settings for configuring the SSE server.
    """
    mcp_server = await _create_proxy_server(client_session)

    # Create the Starlette app with SSE routes and middleware.
    starlette_app = _create_starlette_app(
        mcp_server,
        allow_origins=sse_settings.allow_origins,
        debug=(sse_settings.log_level == "DEBUG"),
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
