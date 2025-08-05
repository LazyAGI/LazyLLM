#!/usr/bin/env python3
"""
推理服务部署脚本
启动LazyLLM推理服务API服务器
"""

import os
import sys
import uvicorn
import argparse
from datetime import datetime

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lazyllm.tools.infer_service.serve import InferServer
from fastapi import FastAPI

def create_app():
    """创建FastAPI应用实例"""
    # 创建FastAPI应用
    fastapi_app = FastAPI(
        title="LazyLLM推理服务API",
        description="用于部署和管理AI模型推理服务的RESTful API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 创建推理服务实例
    infer_server = InferServer()
    
    # 手动注册路由，因为FastapiApp装饰器需要特殊处理
    # 我们需要从InferServer类中获取装饰器信息并手动注册
    
    # 获取InferServer类的所有方法
    methods = dir(infer_server)
    
    # 注册POST /v1/inference_services
    fastapi_app.add_api_route(
        "/v1/inference_services",
        infer_server.create_job,
        methods=["POST"],
        tags=["推理服务"],
        summary="创建推理任务",
        description="创建一个新的AI模型推理任务"
    )
    
    # 注册DELETE /v1/inference_services/{job_id}
    fastapi_app.add_api_route(
        "/v1/inference_services/{job_id}",
        infer_server.cancel_job,
        methods=["DELETE"],
        tags=["推理服务"],
        summary="取消推理任务",
        description="取消指定的推理任务"
    )
    
    # 注册GET /v1/inference_services
    fastapi_app.add_api_route(
        "/v1/inference_services",
        infer_server.list_jobs,
        methods=["GET"],
        tags=["推理服务"],
        summary="获取任务列表",
        description="获取所有推理任务的列表"
    )
    
    # 注册GET /v1/inference_services/{job_id}
    fastapi_app.add_api_route(
        "/v1/inference_services/{job_id}",
        infer_server.get_job_info,
        methods=["GET"],
        tags=["推理服务"],
        summary="获取任务详情",
        description="获取指定推理任务的详细信息"
    )
    
    # 注册GET /v1/inference_services/{job_id}/events
    fastapi_app.add_api_route(
        "/v1/inference_services/{job_id}/events",
        infer_server.get_job_log,
        methods=["GET"],
        tags=["推理服务"],
        summary="获取任务日志",
        description="获取指定推理任务的日志信息"
    )
    
    return fastapi_app

def main():
    parser = argparse.ArgumentParser(description="部署LazyLLM推理服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=31340, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="日志级别")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ.setdefault("LAZYLLM_INFER_LOG_ROOT", "./logs/infer")
    
    # 创建日志目录
    log_dir = os.environ["LAZYLLM_INFER_LOG_ROOT"]
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[{datetime.now()}] 启动LazyLLM推理服务...")
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