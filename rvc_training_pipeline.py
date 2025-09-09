#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC训练Pipeline - 整合模型训练和索引构建
整合了train.py和train_index.py的功能
"""

import os
import sys
import json
import argparse
import logging
import traceback
import platform
from typing import Optional, Dict, Any
from pathlib import Path

# 添加项目路径
now_dir = os.getcwd()
sys.path.append(now_dir)

import numpy as np
import torch
import faiss
from sklearn.cluster import MiniBatchKMeans
from dotenv import load_dotenv
from multiprocessing import cpu_count

# 导入训练相关模块
from infer.lib.train import utils
from infer.lib.train.utils import HParams

load_dotenv()

class Logger:
    """统一的日志管理器"""
    
    def __init__(self, name: str = "RVCTrainingPipeline", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)

class TrainingConfig:
    """训练配置管理器"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.experiment_dir = os.path.join("./logs", args.experiment_name)
        self.config_path = os.path.join(self.experiment_dir, "config.json")
        self.logger = Logger("TrainingConfig")
        
        # 验证配置
        self._validate_config()
        
        # 加载训练配置
        self.hparams = self._load_hparams()
    
    def _validate_config(self):
        """验证配置参数"""
        if not os.path.exists(self.experiment_dir):
            raise ValueError(f"实验目录不存在: {self.experiment_dir}")
        
        if not os.path.exists(self.config_path):
            raise ValueError(f"配置文件不存在: {self.config_path}")
        
        # 验证预训练模型路径
        if self.args.pretrain_g and not os.path.exists(self.args.pretrain_g):
            raise ValueError(f"预训练生成器模型不存在: {self.args.pretrain_g}")
        
        if self.args.pretrain_d and not os.path.exists(self.args.pretrain_d):
            raise ValueError(f"预训练判别器模型不存在: {self.args.pretrain_d}")
    
    def _load_hparams(self) -> HParams:
        """加载超参数配置"""
        with open(self.config_path, "r") as f:
            config = json.load(f)
        
        hparams = HParams(**config)
        
        # 设置实验相关参数
        hparams.model_dir = hparams.experiment_dir = self.experiment_dir
        hparams.name = self.args.experiment_name
        hparams.save_every_epoch = self.args.save_every_epoch
        hparams.total_epoch = self.args.total_epoch
        hparams.pretrainG = self.args.pretrain_g or ""
        hparams.pretrainD = self.args.pretrain_d or ""
        hparams.version = self.args.version
        hparams.gpus = self.args.gpus
        hparams.train.batch_size = self.args.batch_size
        hparams.sample_rate = self.args.sample_rate
        hparams.if_f0 = self.args.if_f0
        hparams.if_latest = self.args.if_latest
        hparams.save_every_weights = self.args.save_every_weights
        hparams.if_cache_data_in_gpu = self.args.if_cache_data_in_gpu
        hparams.data.training_files = f"{self.experiment_dir}/filelist.txt"
        
        return hparams
    
    def get_hparams(self) -> HParams:
        """获取超参数配置"""
        return self.hparams

class IndexTrainer:
    """特征索引训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.hparams = config.get_hparams()
        self.logger = Logger("IndexTrainer")
        self.n_cpu = self._get_n_cpu()
    
    def _get_n_cpu(self, n_cpu: Optional[int] = None) -> int:
        """获取CPU核心数"""
        if n_cpu is None or n_cpu == 0:
            return cpu_count()
        return n_cpu
    
    def train_index(self, outside_index_root: Optional[str] = None) -> bool:
        """训练特征索引"""
        try:
            exp_dir = self.hparams.experiment_dir
            version = self.hparams.version
            exp_name = self.hparams.name
            
            self.logger.info(f"开始训练索引，实验: {exp_name}, 版本: {version}")
            self.logger.info(f"使用CPU核心数: {self.n_cpu}")
            
            # 确定特征目录
            feature_dir = (
                f"{exp_dir}/3_feature256" if version == "v1" 
                else f"{exp_dir}/3_feature768"
            )
            
            # 检查特征目录
            if not os.path.exists(feature_dir):
                self.logger.error("错误: 请先进行特征提取!")
                return False
            
            listdir_res = list(os.listdir(feature_dir))
            if len(listdir_res) == 0:
                self.logger.error("错误: 请先进行特征提取！")
                return False
            
            self.logger.info(f"找到 {len(listdir_res)} 个特征文件")
            
            # 加载所有特征文件
            npys = []
            for name in sorted(listdir_res):
                feature_path = f"{feature_dir}/{name}"
                self.logger.info(f"加载特征文件: {feature_path}")
                phone = np.load(feature_path)
                npys.append(phone)
            
            # 合并所有特征
            big_npy = np.concatenate(npys, 0)
            big_npy_idx = np.arange(big_npy.shape[0])
            np.random.shuffle(big_npy_idx)
            big_npy = big_npy[big_npy_idx]
            
            self.logger.info(f"特征矩阵形状: {big_npy.shape}")
            
            # 如果特征数量过多，使用K-means聚类降维
            if big_npy.shape[0] > 2e5:
                self.logger.info(f"特征数量过多 ({big_npy.shape[0]})，使用K-means聚类到10k个中心点")
                try:
                    big_npy = (
                        MiniBatchKMeans(
                            n_clusters=10000,
                            verbose=True,
                            batch_size=256 * self.n_cpu,
                            compute_labels=False,
                            init="random",
                        )
                        .fit(big_npy)
                        .cluster_centers_
                    )
                    self.logger.info(f"K-means聚类完成，新的特征矩阵形状: {big_npy.shape}")
                except Exception as e:
                    self.logger.error(f"K-means聚类失败: {e}")
                    self.logger.error(traceback.format_exc())
            
            # 保存总特征文件
            total_fea_path = f"{exp_dir}/total_fea.npy"
            np.save(total_fea_path, big_npy)
            self.logger.info(f"保存总特征文件: {total_fea_path}")
            
            # 计算IVF参数
            n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
            feature_dim = 256 if version == "v1" else 768
            self.logger.info(f"特征维度: {feature_dim}, IVF聚类数: {n_ivf}")
            
            # 创建FAISS索引
            index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
            
            self.logger.info("开始训练索引...")
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.nprobe = 1
            index.train(big_npy)
            
            # 保存训练后的索引
            trained_index_path = (
                f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
            )
            faiss.write_index(index, trained_index_path)
            self.logger.info(f"保存训练索引: {trained_index_path}")
            
            self.logger.info("添加特征到索引...")
            batch_size_add = 8192
            for i in range(0, big_npy.shape[0], batch_size_add):
                index.add(big_npy[i : i + batch_size_add])
                if i % (batch_size_add * 10) == 0:
                    self.logger.info(f"已添加 {i + batch_size_add}/{big_npy.shape[0]} 个特征")
            
            # 保存最终索引
            final_index_path = (
                f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
            )
            faiss.write_index(index, final_index_path)
            self.logger.info(f"成功构建索引: {final_index_path}")
            
            # 尝试链接到外部索引目录
            if outside_index_root and os.path.exists(outside_index_root):
                try:
                    link = os.link if platform.system() == "Windows" else os.symlink
                    external_index_path = (
                        f"{outside_index_root}/{exp_name}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
                    )
                    
                    # 如果目标文件已存在，先删除
                    if os.path.exists(external_index_path):
                        os.remove(external_index_path)
                    
                    link(final_index_path, external_index_path)
                    self.logger.info(f"成功链接索引到外部目录: {external_index_path}")
                except Exception as e:
                    self.logger.error(f"链接索引到外部目录失败: {e}")
            
            self.logger.info("索引训练完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"索引训练失败: {e}")
            self.logger.error(traceback.format_exc())
            return False

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.hparams = config.get_hparams()
        self.logger = Logger("ModelTrainer")
    
    def train_model(self) -> bool:
        """训练模型"""
        try:
            self.logger.info("开始模型训练...")
            
            # 设置CUDA环境
            os.environ["CUDA_VISIBLE_DEVICES"] = self.hparams.gpus.replace("-", ",")
            n_gpus = len(self.hparams.gpus.split("-"))
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                self.logger.info(f"使用GPU: {self.hparams.gpus}, 总数: {n_gpus}")
            elif torch.backends.mps.is_available():
                n_gpus = 1
                self.logger.info("使用MPS设备")
            else:
                n_gpus = 1
                self.logger.warning("未检测到GPU，使用CPU训练 - 这可能需要很长时间")
            
            # 设置多进程环境
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(20000 + hash(self.hparams.name) % 30000)
            
            # 启动训练进程
            import torch.multiprocessing as mp
            
            children = []
            train_logger = utils.get_logger(self.hparams.model_dir)
            
            for i in range(n_gpus):
                subproc = mp.Process(
                    target=self._run_training,
                    args=(i, n_gpus, self.hparams, train_logger),
                )
                children.append(subproc)
                subproc.start()
            
            for i in range(n_gpus):
                children[i].join()
            
            self.logger.info("模型训练完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_training(self, rank: int, n_gpus: int, hps: HParams, logger: logging.Logger):
        """运行训练进程"""
        # 导入训练模块
        from infer.modules.train.train import run
        
        # 调用原始训练函数
        run(rank, n_gpus, hps, logger)

class RVCTrainingPipeline:
    """RVC训练Pipeline主类"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = Logger("RVCTrainingPipeline")
        
        try:
            self.config = TrainingConfig(args)
            self.model_trainer = ModelTrainer(self.config)
            self.index_trainer = IndexTrainer(self.config)
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def run_full_pipeline(self) -> bool:
        """运行完整的训练pipeline"""
        self.logger.info("=" * 50)
        self.logger.info("开始RVC训练Pipeline")
        self.logger.info(f"实验名称: {self.args.experiment_name}")
        self.logger.info(f"模型版本: {self.args.version}")
        self.logger.info(f"采样率: {self.args.sample_rate}")
        self.logger.info(f"是否使用F0: {self.args.if_f0}")
        self.logger.info("=" * 50)
        
        success = True
        
        # 步骤1: 模型训练
        if not self.args.skip_training:
            self.logger.info("\n步骤1: 开始模型训练")
            if not self.model_trainer.train_model():
                self.logger.error("模型训练失败")
                success = False
                if not self.args.continue_on_error:
                    return False
        else:
            self.logger.info("\n步骤1: 跳过模型训练")
        
        # 步骤2: 索引训练
        if not self.args.skip_index:
            self.logger.info("\n步骤2: 开始索引训练")
            outside_index_root = self.args.outside_index_root or os.getenv("outside_index_root")
            if not self.index_trainer.train_index(outside_index_root):
                self.logger.error("索引训练失败")
                success = False
                if not self.args.continue_on_error:
                    return False
        else:
            self.logger.info("\n步骤2: 跳过索引训练")
        
        if success:
            self.logger.info("\n=" * 50)
            self.logger.info("RVC训练Pipeline完成！")
            self.logger.info("=" * 50)
        else:
            self.logger.error("\n=" * 50)
            self.logger.error("RVC训练Pipeline部分失败！")
            self.logger.error("=" * 50)
        
        return success
    
    def run_training_only(self) -> bool:
        """仅运行模型训练"""
        self.logger.info("运行模型训练...")
        return self.model_trainer.train_model()
    
    def run_index_only(self) -> bool:
        """仅运行索引训练"""
        self.logger.info("运行索引训练...")
        outside_index_root = self.args.outside_index_root or os.getenv("outside_index_root")
        return self.index_trainer.train_index(outside_index_root)

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RVC训练Pipeline - 整合模型训练和索引构建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整训练pipeline
  python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -bs 32 -te 300 -se 100
  
  # 仅训练模型
  python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -bs 32 -te 300 -se 100 --skip-index
  
  # 仅训练索引
  python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 --skip-training
        """
    )
    
    # 必需参数
    parser.add_argument(
        "-e", "--experiment_name", 
        type=str, required=True,
        help="实验名称（对应logs目录下的文件夹名）"
    )
    parser.add_argument(
        "-sr", "--sample_rate", 
        type=str, required=True, choices=["32k", "40k", "48k"],
        help="采样率"
    )
    parser.add_argument(
        "-f0", "--if_f0", 
        type=int, required=True, choices=[0, 1],
        help="是否使用F0作为模型输入之一"
    )
    
    # 训练参数
    parser.add_argument(
        "-bs", "--batch_size", 
        type=int, default=4,
        help="批次大小 (默认: 4)"
    )
    parser.add_argument(
        "-te", "--total_epoch", 
        type=int, default=300,
        help="总训练轮数 (默认: 300)"
    )
    parser.add_argument(
        "-se", "--save_every_epoch", 
        type=int, default=50,
        help="保存检查点的频率（轮数） (默认: 50)"
    )
    parser.add_argument(
        "-g", "--gpus", 
        type=str, default="0",
        help="GPU设备号，用-分隔 (默认: 0)"
    )
    parser.add_argument(
        "-v", "--version", 
        type=str, default="v2", choices=["v1", "v2"],
        help="模型版本 (默认: v2)"
    )
    
    # 预训练模型
    parser.add_argument(
        "-pg", "--pretrain_g", 
        type=str, default="",
        help="预训练生成器模型路径"
    )
    parser.add_argument(
        "-pd", "--pretrain_d", 
        type=str, default="",
        help="预训练判别器模型路径"
    )
    
    # 其他选项
    parser.add_argument(
        "-l", "--if_latest", 
        type=int, default=0, choices=[0, 1],
        help="是否只保存最新的G/D模型文件 (默认: 0)"
    )
    parser.add_argument(
        "-c", "--if_cache_data_in_gpu", 
        type=int, default=0, choices=[0, 1],
        help="是否将数据集缓存到GPU内存 (默认: 0)"
    )
    parser.add_argument(
        "-sw", "--save_every_weights", 
        type=str, default="0",
        help="保存检查点时是否在weights目录保存提取的模型 (默认: 0)"
    )
    
    # 索引相关
    parser.add_argument(
        "-o", "--outside_index_root", 
        type=str, default="",
        help="外部索引根目录"
    )
    
    # 流程控制
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="跳过模型训练，仅进行索引训练"
    )
    parser.add_argument(
        "--skip-index", 
        action="store_true",
        help="跳过索引训练，仅进行模型训练"
    )
    parser.add_argument(
        "--continue-on-error", 
        action="store_true",
        help="遇到错误时继续执行后续步骤"
    )
    
    # 日志级别
    parser.add_argument(
        "--log-level", 
        type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    try:
        # 创建训练pipeline
        pipeline = RVCTrainingPipeline(args)
        
        # 根据参数选择运行模式
        if args.skip_training and args.skip_index:
            print("错误: 不能同时跳过模型训练和索引训练")
            sys.exit(1)
        elif args.skip_training:
            success = pipeline.run_index_only()
        elif args.skip_index:
            success = pipeline.run_training_only()
        else:
            success = pipeline.run_full_pipeline()
        
        if success:
            print("\n训练Pipeline执行成功！")
            sys.exit(0)
        else:
            print("\n训练Pipeline执行失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练Pipeline执行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 设置多进程启动方法
    torch.multiprocessing.set_start_method("spawn")
    main()