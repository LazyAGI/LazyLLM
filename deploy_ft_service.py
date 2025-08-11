#!/usr/bin/env python3
"""
训练服务部署脚本
启动LazyLLM训练服务API服务器
"""

import os
import sys
import uvicorn
import argparse
from datetime import datetime
from lazyllm.tools.train_service.serveV2 import TrainServer
from lazyllm import FastapiApp
from fastapi import FastAPI

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_app():
    """创建FastAPI应用实例"""
    # 创建FastAPI应用
    fastapi_app = FastAPI(
        title="LazyLLM训练服务API",
        description="用于部署和管理AI模型训练服务的RESTful API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 创建训练服务实例
    train_server = TrainServer()

    # 更新FastapiApp装饰器信息
    FastapiApp.update()

    # 从TrainServer类中获取路由信息并注册
    if hasattr(TrainServer, '__relay_services__'):
        for (method, path), (func_name, kw) in TrainServer.__relay_services__.items():
            # 获取方法对象
            func = getattr(train_server, func_name)
            
            # 设置标签和描述
            tags = ["训练服务"]
            summary = kw.get('summary', func_name)
            description = kw.get('description', f"{method.upper()} {path}")
            
            # 注册路由
            fastapi_app.add_api_route(
                path,
                func,
                methods=[method.upper()],
                tags=tags,
                summary=summary,
                description=description,
                **{k: v for k, v in kw.items() if k not in ['summary', 'description']}
            )

    return fastapi_app


def main():
    parser = argparse.ArgumentParser(description="部署LazyLLM训练服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=31341, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="日志级别")

    args = parser.parse_args()

    # 设置环境变量
    os.environ.setdefault("LAZYLLM_TRAIN_LOG_ROOT", "./logs/train")

    # 创建日志目录
    log_dir = os.environ["LAZYLLM_TRAIN_LOG_ROOT"]
    os.makedirs(log_dir, exist_ok=True)

    print(f"[{datetime.now()}] 启动LazyLLM训练服务...")
    print(f"[{datetime.now()}] 服务地址: http://{args.host}:{args.port}")
    print(f"[{datetime.now()}] API文档: http://{args.host}:{args.port}/docs")
    print(f"[{datetime.now()}] 日志目录: {log_dir}")

    # 创建应用
    app = create_app()

    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
