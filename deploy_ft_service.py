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

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lazyllm.tools.train_service.serveV2 import TrainServer
from fastapi import FastAPI

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
    
    # 手动注册路由，因为FastapiApp装饰器需要特殊处理
    # 我们需要从TrainServer类中获取装饰器信息并手动注册
    
    # 注册POST /v1/finetuneTasks
    fastapi_app.add_api_route(
        "/v1/finetuneTasks",
        train_server.create_job,
        methods=["POST"],
        tags=["训练服务"],
        summary="创建训练任务",
        description="创建一个新的AI模型训练任务"
    )
    
    # 注册DELETE /v1/finetuneTasks/{job_id}
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}",
        train_server.cancel_job,
        methods=["DELETE"],
        tags=["训练服务"],
        summary="取消训练任务",
        description="取消指定的训练任务"
    )
    
    # 注册GET /v1/finetuneTasks/jobs
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/jobs",
        train_server.list_jobs,
        methods=["GET"],
        tags=["训练服务"],
        summary="获取任务列表",
        description="获取所有训练任务的列表"
    )
    
    # 注册GET /v1/finetuneTasks/{job_id}
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}",
        train_server.get_job_info,
        methods=["GET"],
        tags=["训练服务"],
        summary="获取任务详情",
        description="获取指定训练任务的详细信息"
    )
    
    # 注册GET /v1/finetuneTasks/{job_id}/log
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}/log",
        train_server.get_job_log,
        methods=["GET"],
        tags=["训练服务"],
        summary="获取任务日志",
        description="获取指定训练任务的日志信息"
    )
    
    # 注册POST /v1/finetuneTasks/{job_id}:pause
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}:pause",
        train_server.pause_job,
        methods=["POST"],
        tags=["训练服务"],
        summary="暂停训练任务",
        description="暂停指定的训练任务"
    )
    
    # 注册POST /v1/finetuneTasks/{job_id}:resume
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}:resume",
        train_server.resume_job,
        methods=["POST"],
        tags=["训练服务"],
        summary="恢复训练任务",
        description="恢复指定的训练任务"
    )
    
    # 注册GET /v1/finetuneTasks/{job_id}runningMetrics
    fastapi_app.add_api_route(
        "/v1/finetuneTasks/{job_id}runningMetrics",
        train_server.get_running_metrics,
        methods=["GET"],
        tags=["训练服务"],
        summary="获取运行指标",
        description="获取指定训练任务的运行指标"
    )
    
    # 注册GET /v1/models:all
    fastapi_app.add_api_route(
        "/v1/models:all",
        train_server.get_support_model,
        methods=["GET"],
        tags=["训练服务"],
        summary="获取支持的模型",
        description="获取所有支持的模型列表"
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