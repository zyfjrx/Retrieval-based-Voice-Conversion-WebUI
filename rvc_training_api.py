#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC训练API服务
提供FastAPI接口来启动RVC模型训练
"""

import os
import sys
import subprocess
import logging
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RVC Training API",
    description="RVC模型训练接口",
    version="1.0.0"
)

class TrainingRequest(BaseModel):
    """训练请求参数"""
    experiment_dir: str = Field(..., description="实验目录名称", alias="e")
    batch_size: int = Field(..., description="批次大小", alias="bs", gt=0)
    total_epoch: int = Field(..., description="总训练轮数", alias="te", gt=0)
    save_every_epoch: int = Field(..., description="保存检查点的频率(轮数)", alias="se", gt=0)
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "experiment_dir": "test0000",
                "batch_size": 8,
                "total_epoch": 10,
                "save_every_epoch": 5
            }
        }

class TrainingResponse(BaseModel):
    """训练响应"""
    status: str
    message: str
    experiment_dir: Optional[str] = None
    process_id: Optional[int] = None

# 存储正在运行的训练进程
running_processes = {}

@app.get("/")
async def root():
    """根路径"""
    return {"message": "RVC Training API Server", "status": "running"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """启动训练"""
    try:
        # 验证实验目录是否存在config.json
        experiment_path = os.path.join("./logs", request.experiment_dir)
        config_path = os.path.join(experiment_path, "config.json")
        
        if not os.path.exists(config_path):
            raise HTTPException(
                status_code=400, 
                detail=f"实验目录 {request.experiment_dir} 下不存在 config.json 文件，请先完成数据预处理和特征提取"
            )
        
        # 检查是否已有相同实验在运行
        if request.experiment_dir in running_processes:
            process = running_processes[request.experiment_dir]
            if process.poll() is None:  # 进程仍在运行
                raise HTTPException(
                    status_code=409,
                    detail=f"实验 {request.experiment_dir} 已在运行中"
                )
            else:
                # 进程已结束，清理记录
                del running_processes[request.experiment_dir]
        
        # 构建训练命令
        cmd = [
            sys.executable,
            "rvc_training_pipeline.py",
            "-e", request.experiment_dir,
            "-bs", str(request.batch_size),
            "-te", str(request.total_epoch),
            "-se", str(request.save_every_epoch)
        ]
        
        logger.info(f"启动训练命令: {' '.join(cmd)}")
        
        # 创建日志文件路径
        log_dir = os.path.join(experiment_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training.log")
        
        # 启动训练进程
        # 将输出同时写入控制台和日志文件
        process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 启动一个线程来处理输出重定向
        
        def log_output():
            """将进程输出同时写入控制台和日志文件"""
            try:
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            # 输出到控制台
                            print(line.rstrip())
                            # 写入日志文件
                            log_file.write(line)
                            log_file.flush()
                process.stdout.close()
            except Exception as e:
                logger.error(f"日志输出处理错误: {e}")
        
        # 启动日志处理线程
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        # 记录进程
        running_processes[request.experiment_dir] = process
        
        logger.info(f"训练进程已启动，PID: {process.pid}")
        
        return TrainingResponse(
            status="started",
            message=f"训练已启动，实验目录: {request.experiment_dir}",
            experiment_dir=request.experiment_dir,
            process_id=process.pid
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

@app.get("/status/{experiment_dir}")
async def get_training_status(experiment_dir: str):
    """获取训练状态"""
    if experiment_dir not in running_processes:
        return {"status": "not_found", "message": "未找到该实验的训练进程"}
    
    process = running_processes[experiment_dir]
    
    if process.poll() is None:
        return {
            "status": "running",
            "message": "训练正在进行中",
            "process_id": process.pid
        }
    else:
        # 进程已结束
        return_code = process.returncode
        del running_processes[experiment_dir]
        
        if return_code == 0:
            return {
                "status": "completed",
                "message": "训练已完成",
                "return_code": return_code
            }
        else:
            return {
                "status": "failed",
                "message": "训练失败",
                "return_code": return_code
            }

@app.post("/stop/{experiment_dir}")
async def stop_training(experiment_dir: str):
    """停止训练"""
    if experiment_dir not in running_processes:
        raise HTTPException(status_code=404, detail="未找到该实验的训练进程")
    
    process = running_processes[experiment_dir]
    
    if process.poll() is None:
        # 进程仍在运行，终止它
        process.terminate()
        try:
            process.wait(timeout=10)  # 等待10秒
        except subprocess.TimeoutExpired:
            process.kill()  # 强制杀死
        
        del running_processes[experiment_dir]
        return {"status": "stopped", "message": f"训练进程已停止: {experiment_dir}"}
    else:
        del running_processes[experiment_dir]
        return {"status": "already_stopped", "message": "训练进程已经停止"}

@app.get("/experiments")
async def list_running_experiments():
    """列出正在运行的实验"""
    active_experiments = []
    
    # 清理已结束的进程
    to_remove = []
    for exp_dir, process in running_processes.items():
        if process.poll() is None:
            active_experiments.append({
                "experiment_dir": exp_dir,
                "process_id": process.pid,
                "status": "running"
            })
        else:
            to_remove.append(exp_dir)
    
    for exp_dir in to_remove:
        del running_processes[exp_dir]
    
    return {
        "active_experiments": active_experiments,
        "count": len(active_experiments)
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RVC Training API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"启动RVC训练API服务器: {args.host}:{args.port}")
    
    uvicorn.run(
        "rvc_training_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()