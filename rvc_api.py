#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC Training Pipeline FastAPI 封装

这个API服务封装了RVC数据集处理管道，提供REST接口来执行：
1. 数据预处理 (音频切片、重采样)
2. F0特征提取 (RMVPE)
3. HuBERT特征提取
4. 训练文件列表生成
5. 配置文件生成
"""

import os
import sys
import json
import asyncio
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 导入原有的处理模块
now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc_dataset_pipeline import (
    Logger,
    ConfigManager,
    AudioPreprocessor,
    FeatureInput,
    HubertFeatureExtractor,
    FilelistGenerator,
    RVCTrainingPipeline
)

# 创建FastAPI应用
app = FastAPI(
    title="RVC Training Pipeline API",
    description="RVC数据集处理管道的REST API接口",
    version="1.0.0"
)

# 全局任务状态存储
task_status = {}


class PipelineConfig(BaseModel):
    """管道配置模型"""
    exp_dir: str = Field(..., description="实验目录路径")
    inp_root: Optional[str] = Field(None, description="输入音频目录路径")
    sample_rate: int = Field(40000, description="采样率配置")
    n_processes: int = Field(4, description="并行进程数")
    per: float = Field(3.7, description="音频切片长度(秒)")
    
    # 预处理参数
    skip_preprocess: bool = Field(False, description="跳过预处理步骤")
    use_parallel: bool = Field(True, description="使用多进程处理")
    
    # 特征提取参数
    version: str = Field("v2", description="模型版本")
    n_part: int = Field(1, description="总分片数")
    i_part: int = Field(0, description="当前分片索引")
    device: str = Field("cuda", description="计算设备")
    is_half: bool = Field(False, description="使用半精度浮点数")
    
    # 训练参数
    if_f0: bool = Field(True, description="是否使用F0特征")
    spk_id: int = Field(0, description="说话人ID")
    f0_method: str = Field("rmvpe", description="F0提取方法")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        """验证采样率"""
        valid_rates = [32000, 40000, 48000]
        if v not in valid_rates:
            raise ValueError(f'采样率必须是以下值之一: {valid_rates}')
        return v
    
    @validator('exp_dir')
    def validate_exp_dir(cls, v):
        """验证实验目录"""
        if not v or v.strip() == '':
            raise ValueError('实验目录不能为空')
        return v.strip()


class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    """管道响应模型"""
    success: bool
    task_id: str
    message: str
    data: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RVC Training Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "POST /pipeline/run": "运行完整的训练管道",
            "POST /pipeline/preprocess": "仅运行数据预处理",
            "POST /pipeline/extract_f0": "仅运行F0特征提取",
            "POST /pipeline/extract_hubert": "仅运行HuBERT特征提取",
            "POST /pipeline/generate_filelist": "仅生成训练文件列表",
            "GET /tasks/{task_id}": "查询任务状态",
            "GET /tasks": "查询所有任务状态"
        }
    }


@app.post("/pipeline/run", response_model=PipelineResponse)
async def run_full_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    """运行完整的训练管道"""
    task_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # 初始化任务状态
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        message="任务已创建，等待执行",
        start_time=datetime.now()
    )
    
    # 添加后台任务
    background_tasks.add_task(run_pipeline_task, task_id, config)
    
    return PipelineResponse(
        success=True,
        task_id=task_id,
        message="管道任务已启动",
        data={"config": config.dict()}
    )


@app.post("/pipeline/preprocess", response_model=PipelineResponse)
async def run_preprocess_only(config: PipelineConfig, background_tasks: BackgroundTasks):
    """仅运行数据预处理"""
    task_id = f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        message="预处理任务已创建",
        start_time=datetime.now()
    )
    
    background_tasks.add_task(run_preprocess_task, task_id, config)
    
    return PipelineResponse(
        success=True,
        task_id=task_id,
        message="预处理任务已启动"
    )


@app.post("/pipeline/extract_f0", response_model=PipelineResponse)
async def run_f0_extraction(config: PipelineConfig, background_tasks: BackgroundTasks):
    """仅运行F0特征提取"""
    task_id = f"f0_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        message="F0特征提取任务已创建",
        start_time=datetime.now()
    )
    
    background_tasks.add_task(run_f0_task, task_id, config)
    
    return PipelineResponse(
        success=True,
        task_id=task_id,
        message="F0特征提取任务已启动"
    )


@app.post("/pipeline/extract_hubert", response_model=PipelineResponse)
async def run_hubert_extraction(config: PipelineConfig, background_tasks: BackgroundTasks):
    """仅运行HuBERT特征提取"""
    task_id = f"hubert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        message="HuBERT特征提取任务已创建",
        start_time=datetime.now()
    )
    
    background_tasks.add_task(run_hubert_task, task_id, config)
    
    return PipelineResponse(
        success=True,
        task_id=task_id,
        message="HuBERT特征提取任务已启动"
    )


@app.post("/pipeline/generate_filelist", response_model=PipelineResponse)
async def run_filelist_generation(config: PipelineConfig, background_tasks: BackgroundTasks):
    """仅生成训练文件列表"""
    task_id = f"filelist_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        message="文件列表生成任务已创建",
        start_time=datetime.now()
    )
    
    background_tasks.add_task(run_filelist_task, task_id, config)
    
    return PipelineResponse(
        success=True,
        task_id=task_id,
        message="文件列表生成任务已启动"
    )


@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查询特定任务状态"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return task_status[task_id]


@app.get("/tasks", response_model=List[TaskStatus])
async def get_all_tasks():
    """查询所有任务状态"""
    return list(task_status.values())


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务记录"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    del task_status[task_id]
    return {"message": f"任务 {task_id} 已删除"}


@app.get("/logs/{task_id}")
async def get_task_logs(task_id: str):
    """获取任务日志"""
    log_file = f"runlogs/{task_id}.log"
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    return FileResponse(log_file, media_type="text/plain", filename=f"{task_id}.log")


# 后台任务函数
async def run_pipeline_task(task_id: str, config: PipelineConfig):
    """运行完整管道的后台任务"""
    try:
        # 更新任务状态
        task_status[task_id].status = "running"
        task_status[task_id].message = "正在运行完整管道"
        
        # 创建模拟的args对象
        class Args:
            def __init__(self, config: PipelineConfig):
                self.exp_dir = config.exp_dir
                self.inp_root = config.inp_root
                # self.sr = config.sample_rate
                self.sample_rate = config.sample_rate  # 添加sample_rate属性
                self.n_processes = config.n_processes
                self.per = config.per
                self.skip_preprocess = config.skip_preprocess
                self.use_parallel = config.use_parallel
                self.no_parallel = not config.use_parallel  # 添加no_parallel属性
                self.version = config.version
                self.n_part = config.n_part
                self.i_part = config.i_part
                self.device = config.device
                self.is_half = config.is_half
                self.if_f0 = config.if_f0
                self.spk_id = config.spk_id
                self.f0_method = config.f0_method
                self.i_gpu = 0  # 默认值
        
        args = Args(config)
        
        # 运行管道
        pipeline = RVCTrainingPipeline(args)
        success = pipeline.run()
        
        # 更新任务状态
        if success:
            task_status[task_id].status = "completed"
            task_status[task_id].message = "管道执行完成"
            task_status[task_id].progress = 100.0
            task_status[task_id].result = {
                "exp_dir": config.exp_dir,
                "files_generated": {
                    "config": os.path.join(config.exp_dir, "config.json"),
                    "filelist": os.path.join(config.exp_dir, "filelist.txt")
                }
            }
        else:
            task_status[task_id].status = "failed"
            task_status[task_id].message = "管道执行失败"
            
    except Exception as e:
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"执行出错: {str(e)}"
        task_status[task_id].error = traceback.format_exc()
    
    finally:
        task_status[task_id].end_time = datetime.now()


async def run_preprocess_task(task_id: str, config: PipelineConfig):
    """运行预处理的后台任务"""
    try:
        task_status[task_id].status = "running"
        task_status[task_id].message = "正在进行音频预处理"
        
        # 创建日志记录器
        log_file = f"runlogs/{task_id}.log"
        os.makedirs("runlogs", exist_ok=True)
        
        with Logger(log_file) as logger:
            # 解析采样率
            # sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
            # sr = sr_map.get(config.sr, 40000)
            
            # 创建预处理器
            preprocessor = AudioPreprocessor(
                sr=config.sample_rate,
                exp_dir=config.exp_dir,
                logger=logger,
                per=config.per
            )
            
            # 运行预处理
            success = preprocessor.process_directory(
                inp_root=config.inp_root,
                n_processes=config.n_processes,
                use_parallel=config.use_parallel
            )
            
            if success:
                task_status[task_id].status = "completed"
                task_status[task_id].message = "预处理完成"
                task_status[task_id].progress = 100.0
            else:
                task_status[task_id].status = "failed"
                task_status[task_id].message = "预处理失败"
                
    except Exception as e:
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"预处理出错: {str(e)}"
        task_status[task_id].error = traceback.format_exc()
    
    finally:
        task_status[task_id].end_time = datetime.now()


async def run_f0_task(task_id: str, config: PipelineConfig):
    """运行F0特征提取的后台任务"""
    try:
        task_status[task_id].status = "running"
        task_status[task_id].message = "正在提取F0特征"
        
        log_file = f"runlogs/{task_id}.log"
        os.makedirs("runlogs", exist_ok=True)
        
        with Logger(log_file) as logger:
            # 创建F0特征提取器
            f0_extractor = FeatureInput(is_half=config.is_half)
            
            # 准备文件路径
            wavs16k_dir = os.path.join(config.exp_dir, "1_16k_wavs")
            f0_dir = os.path.join(config.exp_dir, "2a_f0")
            f0nsf_dir = os.path.join(config.exp_dir, "2b-f0nsf")
            
            os.makedirs(f0_dir, exist_ok=True)
            os.makedirs(f0nsf_dir, exist_ok=True)
            
            # 获取文件列表
            if os.path.exists(wavs16k_dir):
                wav_files = [f for f in os.listdir(wavs16k_dir) if f.endswith(".wav")]
                paths = []
                
                for wav_file in wav_files:
                    wav_path = os.path.join(wavs16k_dir, wav_file)
                    base_name = wav_file.replace(".wav", "")
                    f0_path = os.path.join(f0_dir, base_name)
                    f0nsf_path = os.path.join(f0nsf_dir, base_name)
                    paths.append([wav_path, f0_path, f0nsf_path])
                
                # 提取F0特征
                f0_extractor.extract_f0_features(paths, logger, config.f0_method)
                
                task_status[task_id].status = "completed"
                task_status[task_id].message = "F0特征提取完成"
                task_status[task_id].progress = 100.0
            else:
                task_status[task_id].status = "failed"
                task_status[task_id].message = f"16k音频目录不存在: {wavs16k_dir}"
                
    except Exception as e:
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"F0特征提取出错: {str(e)}"
        task_status[task_id].error = traceback.format_exc()
    
    finally:
        task_status[task_id].end_time = datetime.now()


async def run_hubert_task(task_id: str, config: PipelineConfig):
    """运行HuBERT特征提取的后台任务"""
    try:
        task_status[task_id].status = "running"
        task_status[task_id].message = "正在提取HuBERT特征"
        
        log_file = f"runlogs/{task_id}.log"
        os.makedirs("runlogs", exist_ok=True)
        
        with Logger(log_file) as logger:
            # 创建HuBERT特征提取器
            hubert_extractor = HubertFeatureExtractor(
                device=config.device,
                exp_dir=config.exp_dir,
                logger=logger,
                version=config.version,
                is_half=config.is_half
            )
            
            # 提取特征
            success = hubert_extractor.extract_features_batch(
                n_part=config.n_part,
                i_part=config.i_part
            )
            
            if success:
                task_status[task_id].status = "completed"
                task_status[task_id].message = "HuBERT特征提取完成"
                task_status[task_id].progress = 100.0
            else:
                task_status[task_id].status = "failed"
                task_status[task_id].message = "HuBERT特征提取失败"
                
    except Exception as e:
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"HuBERT特征提取出错: {str(e)}"
        task_status[task_id].error = traceback.format_exc()
    
    finally:
        task_status[task_id].end_time = datetime.now()


async def run_filelist_task(task_id: str, config: PipelineConfig):
    """运行文件列表生成的后台任务"""
    try:
        task_status[task_id].status = "running"
        task_status[task_id].message = "正在生成训练文件列表"
        
        log_file = f"runlogs/{task_id}.log"
        os.makedirs("runlogs", exist_ok=True)
        
        with Logger(log_file) as logger:
            # 创建配置管理器
            config_manager = ConfigManager(logger)
            config_manager.load_configs()
            
            # 生成配置文件
            config_manager.generate_config(
                exp_dir=config.exp_dir,
                version=config.version,
                sr_key=config.sample_rate
            )
            
            # 创建文件列表生成器
            filelist_generator = FilelistGenerator(logger)
            
            # 生成文件列表
            success = filelist_generator.generate_filelist(
                exp_dir=config.exp_dir,
                version=config.version,
                if_f0=config.if_f0,
                spk_id=config.spk_id,
                sr=config.sample_rate
            )
            
            if success:
                task_status[task_id].status = "completed"
                task_status[task_id].message = "文件列表生成完成"
                task_status[task_id].progress = 100.0
            else:
                task_status[task_id].status = "failed"
                task_status[task_id].message = "文件列表生成失败"
                
    except Exception as e:
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"文件列表生成出错: {str(e)}"
        task_status[task_id].error = traceback.format_exc()
    
    finally:
        task_status[task_id].end_time = datetime.now()


if __name__ == "__main__":
    uvicorn.run(
        "rvc_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )