import uvicorn

from dataclasses import dataclass
from typing import Literal, Any, Optional, List

from lazyllm.thirdparty import mcp

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


def _create_starlette_app(mcp_server, *, allow_origins=None, debug=False) -> Starlette:
    """
    Create a Starlette application to serve the provided MCP server with SSE.

    Args:
        mcp_server: The MCP server instance.
        allow_origins: Allowed origins for CORS middleware.
        debug: Flag indicating whether to enable debug mode.

    Returns:
        A configured Starlette application.
    """
    sse = mcp.server.sse.SseServerTransport("/messages/")

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


async def _create_proxy_server(remote_app): # noqa C901
    """
    Create a proxy server instance based on a remote client session.

    Args:
        remote_app: A client session for a remote MCP application.

    Returns:
        An instance of Server with request and notification handlers mapped.
    """
    response = await remote_app.initialize()
    capabilities = response.capabilities

    server_instance: mcp.server.Server[Any] = mcp.server.Server(name=response.serverInfo.name)

    if capabilities.prompts:
        async def _list_prompts(_: Any) -> mcp.types.ServerResult:
            result = await remote_app.list_prompts()
            return mcp.types.ServerResult(result)

        async def _get_prompt(req: mcp.types.GetPromptRequest) -> mcp.types.ServerResult:
            result = await remote_app.get_prompt(req.params.name, req.params.arguments)
            return mcp.types.ServerResult(result)

        server_instance.request_handlers[mcp.types.ListPromptsRequest] = _list_prompts
        server_instance.request_handlers[mcp.types.GetPromptRequest] = _get_prompt

    if capabilities.resources:
        async def _subscribe_resource(req: mcp.types.SubscribeRequest) -> mcp.types.ServerResult:
            await remote_app.subscribe_resource(req.params.uri)
            return mcp.types.ServerResult(mcp.types.EmptyResult())

        async def _unsubscribe_resource(req: mcp.types.UnsubscribeRequest) -> mcp.types.ServerResult:
            await remote_app.unsubscribe_resource(req.params.uri)
            return mcp.types.ServerResult(mcp.types.EmptyResult())

        async def _list_resources(_: Any) -> mcp.types.ServerResult:
            result = await remote_app.list_resources()
            return mcp.types.ServerResult(result)

        async def _read_resource(req: mcp.types.ReadResourceRequest) -> mcp.types.ServerResult:
            result = await remote_app.read_resource(req.params.uri)
            return mcp.types.ServerResult(result)

        server_instance.request_handlers[mcp.types.SubscribeRequest] = _subscribe_resource
        server_instance.request_handlers[mcp.types.UnsubscribeRequest] = _unsubscribe_resource
        server_instance.request_handlers[mcp.types.ListResourcesRequest] = _list_resources
        server_instance.request_handlers[mcp.types.ReadResourceRequest] = _read_resource

    if capabilities.logging:
        async def _set_logging_level(req: mcp.types.SetLevelRequest) -> mcp.types.ServerResult:
            await remote_app.set_logging_level(req.params.level)
            return mcp.types.ServerResult(mcp.types.EmptyResult())

        server_instance.request_handlers[mcp.types.SetLevelRequest] = _set_logging_level

    if capabilities.tools:
        async def _list_tools(_: Any) -> mcp.types.ServerResult:
            tools = await remote_app.list_tools()
            return mcp.types.ServerResult(tools)

        async def _call_tool(req: mcp.types.CallToolRequest) -> mcp.types.ServerResult:
            try:
                result = await remote_app.call_tool(
                    req.params.name,
                    req.params.arguments or {},
                )
                return mcp.types.ServerResult(result)
            except Exception as e:
                return mcp.types.ServerResult(
                    mcp.types.CallToolResult(
                        content=[mcp.types.TextContent(type="text", text=str(e))],
                        isError=True,
                    )
                )

        server_instance.request_handlers[mcp.types.ListToolsRequest] = _list_tools
        server_instance.request_handlers[mcp.types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: mcp.types.ProgressNotification) -> None:
        await remote_app.send_progress_notification(
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    server_instance.notification_handlers[mcp.types.ProgressNotification] = _send_progress_notification

    async def _complete(req: mcp.types.CompleteRequest) -> mcp.types.ServerResult:
        result = await remote_app.complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return mcp.types.ServerResult(result)

    server_instance.request_handlers[mcp.types.CompleteRequest] = _complete

    return server_instance


async def start_sse_server(
    client_session,
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
