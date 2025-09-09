#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC Training Pipeline - 整合数据预处理和特征提取

这个脚本整合了以下功能：
1. 数据预处理 (音频切片、重采样)
2. F0特征提取 (RMVPE)
3. HuBERT特征提取
4. 训练文件列表生成
5. 配置文件生成
"""

import os
import sys
import json
import shutil
import argparse
import traceback
import multiprocessing
from typing import Optional, List, Tuple
from random import shuffle
from pathlib import Path

# 音频处理相关
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile

# PyTorch相关
import torch
import torch.nn.functional as F
import fairseq

# RVC相关模块
now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer
from multiprocessing import cpu_count

# 日志设置
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)

# 版本配置列表
VERSION_CONFIG_LIST = [
    "v1/32k.json",
    "v1/40k.json", 
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


class Logger:
    """统一的日志管理器"""
    
    def __init__(self, log_file_path: str):
        self.log_file = open(log_file_path, "a+", encoding="utf-8")
    
    def log(self, message: str):
        """打印并记录日志"""
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.configs = {}
    
    def load_configs(self) -> bool:
        """加载所有配置文件"""
        try:
            for config_file in VERSION_CONFIG_LIST:
                config_source_path = os.path.join(now_dir, "configs", config_file)
                config_inuse_path = os.path.join(now_dir, "configs", "inuse", config_file)
                
                # 确保 inuse 目录存在
                inuse_dir = os.path.dirname(config_inuse_path)
                os.makedirs(inuse_dir, exist_ok=True)
                
                # 如果 inuse 中不存在配置文件，从原始位置复制
                if not os.path.exists(config_inuse_path):
                    if os.path.exists(config_source_path):
                        shutil.copy(config_source_path, config_inuse_path)
                    else:
                        self.logger.log(f"⚠️ 警告: 配置文件 {config_source_path} 不存在")
                        continue
                
                # 读取配置文件
                with open(config_inuse_path, "r", encoding="utf-8") as f:
                    self.configs[config_file] = json.load(f)
            
            self.logger.log(f"✅ 成功加载 {len(self.configs)} 个配置文件")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ 加载配置文件失败: {e}")
            return False
    
    def generate_config(self, exp_dir: str, version: str = "v2", sr_key: str = "40k") -> bool:
        """生成训练配置文件"""
        try:
            # 确定配置文件路径
            if version == "v1" or sr_key == "40k":
                config_path = f"v1/{sr_key}.json"
            else:
                config_path = f"v2/{sr_key}.json"
            
            # 检查配置是否存在
            if config_path not in self.configs:
                self.logger.log(f"❌ 错误: 配置文件 {config_path} 不存在")
                return False
            
            # 配置文件保存路径
            config_save_path = os.path.join(exp_dir, "config.json")
            
            # 检查配置文件是否已存在
            if not Path(config_save_path).exists():
                # 创建目录（如果不存在）
                os.makedirs(exp_dir, exist_ok=True)
                
                # 写入配置文件
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self.configs[config_path],
                        f,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    )
                    f.write("\n")
                
                self.logger.log(f"✅ 成功生成配置文件: {config_save_path}")
            else:
                self.logger.log(f"ℹ️ 配置文件已存在: {config_save_path}")
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ 生成配置文件失败: {e}")
            return False


class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(self, sr: int, exp_dir: str, logger: Logger, per: float = 3.7):
        self.sr = sr
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.logger = logger
        
        # 初始化音频切片器
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        
        # 初始化高通滤波器
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        
        # 创建输出目录
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
    
    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int):
        """标准化并写入音频文件"""
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            self.logger.log(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        
        # 标准化音频
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        
        # 写入原始采样率音频
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        
        # 重采样到16kHz并写入
        tmp_audio_16k = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),
            16000,
            tmp_audio_16k.astype(np.float32),
        )
    
    def process_single_file(self, path: str, idx0: int):
        """处理单个音频文件"""
        try:
            # 加载音频
            audio = load_audio(path, self.sr)
            
            # 应用高通滤波器
            audio = signal.lfilter(self.bh, self.ah, audio)
            
            idx1 = 0
            for audio_slice in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio_slice[start:]) > self.tail * self.sr:
                        tmp_audio = audio_slice[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_slice[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            
            self.logger.log(f"{path} -> Success")
            
        except Exception as e:
            self.logger.log(f"{path} -> {traceback.format_exc()}")
    
    def process_directory(self, inp_root: str, n_processes: int = None, use_parallel: bool = True):
        """处理整个目录"""
        try:
            if n_processes is None:
                n_processes = cpu_count()
            
            # 获取所有音频文件
            audio_files = sorted([f for f in os.listdir(inp_root) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))])
            
            if not audio_files:
                self.logger.log("❌ 未找到音频文件")
                return False
            
            self.logger.log(f"📁 找到 {len(audio_files)} 个音频文件")
            
            # 创建文件信息列表
            infos = [
                (os.path.join(inp_root, name), idx)
                for idx, name in enumerate(audio_files)
            ]
            
            if not use_parallel or n_processes == 1:
                # 单进程处理
                self.logger.log("🔄 开始单进程处理...")
                for path, idx in infos:
                    self.process_single_file(path, idx)
            else:
                # 多进程处理
                self.logger.log(f"🔄 开始多进程处理 (进程数: {n_processes})...")
                processes = []
                for i in range(n_processes):
                    process_infos = infos[i::n_processes]
                    if process_infos:  # 确保有文件要处理
                        p = multiprocessing.Process(
                            target=self._process_batch,
                            args=(process_infos,)
                        )
                        processes.append(p)
                        p.start()
                
                # 等待所有进程完成
                for p in processes:
                    p.join()
            
            self.logger.log("✅ 音频预处理完成")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ 预处理失败: {e}")
            self.logger.log(traceback.format_exc())
            return False
    
    def _process_batch(self, infos: List[Tuple[str, int]]):
        """批量处理文件（用于多进程）"""
        for path, idx in infos:
            self.process_single_file(path, idx)


class FeatureInput:
    """F0特征提取器"""
    
    def __init__(self, is_half: bool = True, samplerate: int = 16000, hop_size: int = 160):
        self.fs = samplerate
        self.hop = hop_size
        self.is_half = is_half
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.model_rmvpe = None
    
    def compute_f0(self, path: str, f0_method: str = "rmvpe") -> np.ndarray:
        """计算F0特征"""
        x = load_audio(path, self.fs)
        
        if f0_method == "rmvpe":
            if self.model_rmvpe is None:
                from infer.lib.rmvpe import RMVPE
                print("Loading RMVPE model...")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", 
                    is_half=self.is_half, 
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        
        return f0
    
    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """将F0转换为粗糙F0"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse
    
    def extract_f0_features(self, paths: List[List[str]], logger: Logger, f0_method: str = "rmvpe"):
        """批量提取F0特征"""
        if len(paths) == 0:
            logger.log("no-f0-todo")
            return
        
        logger.log(f"todo-f0-{len(paths)}")
        n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
        
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    logger.log(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                
                # 检查文件是否已存在
                if (
                    os.path.exists(opt_path1 + ".npy")
                    and os.path.exists(opt_path2 + ".npy")
                ):
                    continue
                
                # 计算F0特征
                featur_pit = self.compute_f0(inp_path, f0_method)
                
                # 保存NSF F0
                np.save(opt_path2, featur_pit, allow_pickle=False)
                
                # 保存粗糙F0
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)
                
            except Exception as e:
                logger.log(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


class HubertFeatureExtractor:
    """HuBERT特征提取器"""
    
    def __init__(self, device: str, exp_dir: str, logger: Logger, 
                 version: str = "v2", is_half: bool = False,
                 model_path: str = "assets/hubert/hubert_base.pt"):
        self.exp_dir = exp_dir
        self.version = version
        self.is_half = is_half
        self.model_path = model_path
        self.logger = logger
        
        # 设置环境变量
        self._setup_environment()
        
        # 初始化设备
        self.device = self._setup_device(device)
        
        # 设置路径
        self.wav_path = os.path.join(exp_dir, "1_16k_wavs")
        self.out_path = os.path.join(
            exp_dir,
            "3_feature256" if version == "v1" else "3_feature768"
        )
        os.makedirs(self.out_path, exist_ok=True)
        
        # 加载模型
        self.model, self.saved_cfg = self._load_model()
    
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if "privateuseone" not in device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            try:
                import torch_directml
                device = torch_directml.device(torch_directml.default_device())
                
                # 设置DirectML前向传播
                def forward_dml(ctx, x, scale):
                    ctx.scale = scale
                    res = x.clone().detach()
                    return res
                fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
            except ImportError:
                self.logger.log("⚠️ torch_directml not available, falling back to CPU")
                device = "cpu"
        
        return device
    
    def _load_model(self) -> Tuple[torch.nn.Module, object]:
        """加载HuBERT模型"""
        self.logger.log(f"🔄 加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            error_msg = (
                f"❌ 错误: 模型文件 {self.model_path} 不存在，"
                "请从 https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main 下载"
            )
            self.logger.log(error_msg)
            raise FileNotFoundError(error_msg)
        
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.model_path], suffix=""
        )
        
        model = models[0]
        model = model.to(self.device)
        self.logger.log(f"📱 模型已移动到: {self.device}")
        
        if self.is_half and self.device not in ["mps", "cpu"]:
            model = model.half()
        
        model.eval()
        return model, saved_cfg
    
    def readwave(self, wav_path: str, normalize: bool = False) -> torch.Tensor:
        """读取并预处理音频文件"""
        wav, sr = sf.read(wav_path)
        assert sr == 16000, f"采样率必须为16000，当前为{sr}"
        
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # 双声道转单声道
            feats = feats.mean(-1)
        assert feats.dim() == 1, f"期望1D张量，得到{feats.dim()}D"
        
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        
        feats = feats.view(1, -1)
        return feats
    
    def extract_features_from_audio(self, wav_path: str) -> Optional[np.ndarray]:
        """从单个音频文件提取特征"""
        try:
            feats = self.readwave(wav_path, normalize=self.saved_cfg.task.normalize)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            
            inputs = {
                "source": (
                    feats.half().to(self.device)
                    if self.is_half and self.device not in ["mps", "cpu"]
                    else feats.to(self.device)
                ),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9 if self.version == "v1" else 12,
            }
            
            with torch.no_grad():
                logits = self.model.extract_features(**inputs)
                feats = (
                    self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
                )
            
            feats = feats.squeeze(0).float().cpu().numpy()
            
            if np.isnan(feats).sum() > 0:
                self.logger.log(f"⚠️ 警告: {os.path.basename(wav_path)} 包含NaN值")
                return None
            
            return feats
            
        except Exception as e:
            self.logger.log(f"❌ 处理 {wav_path} 时出错: {str(e)}")
            return None
    
    def extract_features_batch(self, n_part: int = 1, i_part: int = 0):
        """批量提取特征"""
        if not os.path.exists(self.wav_path):
            self.logger.log(f"❌ 错误: 输入目录 {self.wav_path} 不存在")
            return False
        
        # 获取待处理文件列表
        all_files = sorted([f for f in os.listdir(self.wav_path) if f.endswith(".wav")])
        todo = all_files[i_part::n_part]
        
        if len(todo) == 0:
            self.logger.log("no-feature-todo")
            return True
        
        self.logger.log(f"all-feature-{len(todo)}")
        n_log = max(1, len(todo) // 10)  # 最多打印十条进度
        
        for idx, file in enumerate(todo):
            wav_path = os.path.join(self.wav_path, file)
            out_path = os.path.join(self.out_path, file.replace(".wav", ".npy"))
            
            # 跳过已存在的文件
            if os.path.exists(out_path):
                continue
            
            # 提取特征
            feats = self.extract_features_from_audio(wav_path)
            if feats is not None:
                np.save(out_path, feats, allow_pickle=False)
            
            # 打印进度
            if idx % n_log == 0:
                shape_str = str(feats.shape) if feats is not None else "None"
                self.logger.log(f"now-{idx},all-{len(todo)},{file},{shape_str}")
        
        self.logger.log("✅ HuBERT特征提取完成")
        return True


class FilelistGenerator:
    """训练文件列表生成器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def generate_filelist(self, exp_dir: str, version: str = "v2", 
                         if_f0: bool = True, spk_id: int = 0, sr: str = "40k") -> bool:
        """生成训练用的filelist.txt文件"""
        try:
            # 获取当前工作目录的绝对路径
            current_dir = os.path.abspath(os.getcwd())
            
            # 智能处理实验目录路径
            if not os.path.isabs(exp_dir):
                if exp_dir.startswith("logs/"):
                    exp_dir = os.path.abspath(exp_dir)
                else:
                    exp_dir = os.path.abspath(os.path.join(current_dir, "logs", exp_dir))
            
            # 确保实验目录存在
            if not os.path.exists(exp_dir):
                self.logger.log(f"❌ 错误: 实验目录 {exp_dir} 不存在")
                return False
            
            # 定义各个数据目录
            gt_wavs_dir = os.path.abspath(os.path.join(exp_dir, "0_gt_wavs"))
            feature_dir = os.path.abspath(os.path.join(
                exp_dir, "3_feature256" if version == "v1" else "3_feature768"
            ))
            
            # 检查必要目录是否存在
            if not os.path.exists(gt_wavs_dir):
                self.logger.log(f"❌ 错误: 原始音频目录 {gt_wavs_dir} 不存在")
                return False
            
            if not os.path.exists(feature_dir):
                self.logger.log(f"❌ 错误: 特征目录 {feature_dir} 不存在")
                return False
            
            # 获取文件名集合
            gt_names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir) if name.endswith(".wav")])
            feature_names = set([name.split(".")[0] for name in os.listdir(feature_dir) if name.endswith(".npy")])
            
            if if_f0:
                f0_dir = os.path.abspath(os.path.join(exp_dir, "2a_f0"))
                f0nsf_dir = os.path.abspath(os.path.join(exp_dir, "2b-f0nsf"))
                
                if not os.path.exists(f0_dir):
                    self.logger.log(f"❌ 错误: F0目录 {f0_dir} 不存在")
                    return False
                
                if not os.path.exists(f0nsf_dir):
                    self.logger.log(f"❌ 错误: F0NSF目录 {f0nsf_dir} 不存在")
                    return False
                
                f0_names = set([name.split(".")[0] for name in os.listdir(f0_dir) if name.endswith(".npy")])
                f0nsf_names = set([name.split(".")[0] for name in os.listdir(f0nsf_dir) if name.endswith(".npy")])
                
                # 取所有目录的交集
                names = gt_names & feature_names & f0_names & f0nsf_names
            else:
                # 不使用F0时，只需要原始音频和特征的交集
                names = gt_names & feature_names
            
            if len(names) == 0:
                self.logger.log("❌ 错误: 没有找到匹配的文件")
                return False
            
            self.logger.log(f"📋 找到 {len(names)} 个匹配的文件")
            
            # 构建文件路径列表
            opt = []
            for name in names:
                if if_f0:
                    line = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
                else:
                    line = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}"
                opt.append(line)
            
            # 添加静音数据
            fea_dim = 256 if version == "v1" else 768
            mute_base_dir = os.path.abspath(os.path.join(current_dir, "logs", "mute"))
            
            if if_f0:
                for _ in range(2):
                    mute_line = f"{mute_base_dir}/0_gt_wavs/mute{sr}.wav|{mute_base_dir}/3_feature{fea_dim}/mute.npy|{mute_base_dir}/2a_f0/mute.wav.npy|{mute_base_dir}/2b-f0nsf/mute.wav.npy|{spk_id}"
                    opt.append(mute_line)
            else:
                for _ in range(2):
                    mute_line = f"{mute_base_dir}/0_gt_wavs/mute{sr}.wav|{mute_base_dir}/3_feature{fea_dim}/mute.npy|{spk_id}"
                    opt.append(mute_line)
            
            # 随机打乱数据
            shuffle(opt)
            
            # 写入filelist.txt
            filelist_path = os.path.abspath(os.path.join(exp_dir, "filelist.txt"))
            with open(filelist_path, "w", encoding="utf-8") as f:
                f.write("\n".join(opt))
            
            self.logger.log(f"✅ 成功生成 {filelist_path}")
            self.logger.log(f"📊 总共 {len(opt)} 行数据 (包含 {len(names)} 个训练样本 + 2个静音样本)")
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ 生成文件列表失败: {e}")
            self.logger.log(traceback.format_exc())
            return False


class RVCTrainingPipeline:
    """RVC训练数据准备Pipeline"""
    
    def __init__(self, args):
        self.args = args
        
        # 创建实验目录
        self.exp_dir = os.path.join(now_dir, "logs", args.exp_dir)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 初始化日志
        log_path = os.path.join(self.exp_dir, "pipeline.log")
        self.logger = Logger(log_path)
        
        # 初始化各个组件
        self.config_manager = ConfigManager(self.logger)
        self.filelist_generator = FilelistGenerator(self.logger)
        
        # 设置GPU环境
        if hasattr(args, 'i_gpu'):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.i_gpu)
    
    def run(self) -> bool:
        """运行完整的训练数据准备流程"""
        try:
            self.logger.log("🚀 开始RVC训练数据准备Pipeline")
            self.logger.log(f"📂 实验目录: {self.exp_dir}")
            self.logger.log(f"⚙️ 参数: {' '.join(sys.argv)}")
            
            # 1. 加载配置文件
            self.logger.log("\n📋 步骤1: 加载配置文件")
            if not self.config_manager.load_configs():
                return False
            
            # 2. 音频预处理
            if self.args.skip_preprocess:
                self.logger.log("\n⏭️ 跳过音频预处理步骤")
            else:
                self.logger.log("\n🎵 步骤2: 音频预处理")
                preprocessor = AudioPreprocessor(
                    sr=self.args.sample_rate,
                    exp_dir=self.exp_dir,
                    logger=self.logger,
                    per=self.args.per
                )
                
                if not preprocessor.process_directory(
                    inp_root=self.args.inp_root,
                    n_processes=self.args.n_processes,
                    use_parallel=not self.args.no_parallel
                ):
                    return False
            
            # 3. F0特征提取
            if self.args.if_f0:
                self.logger.log("\n🎼 步骤3: F0特征提取")
                if not self._extract_f0_features():
                    return False
            else:
                self.logger.log("\n⏭️ 跳过F0特征提取")
            
            # 4. HuBERT特征提取
            self.logger.log("\n🧠 步骤4: HuBERT特征提取")
            if not self._extract_hubert_features():
                return False
            
            # 5. 生成配置文件
            self.logger.log("\n⚙️ 步骤5: 生成配置文件")
            sr_key = f"{self.args.sample_rate // 1000}k"
            if not self.config_manager.generate_config(
                exp_dir=self.exp_dir,
                version=self.args.version,
                sr_key=sr_key
            ):
                return False
            
            # 6. 生成训练文件列表
            self.logger.log("\n📋 步骤6: 生成训练文件列表")
            if not self.filelist_generator.generate_filelist(
                exp_dir=self.exp_dir,
                version=self.args.version,
                if_f0=self.args.if_f0,
                spk_id=self.args.spk_id,
                sr=sr_key
            ):
                return False
            
            self.logger.log("\n🎉 RVC训练数据准备Pipeline完成！")
            self.logger.log(f"📁 所有文件已保存到: {self.exp_dir}")
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Pipeline执行失败: {e}")
            self.logger.log(traceback.format_exc())
            return False
        
        finally:
            self.logger.close()
    
    def _extract_f0_features(self) -> bool:
        """提取F0特征"""
        try:
            # 创建F0输出目录
            f0_coarse_dir = os.path.join(self.exp_dir, "2a_f0")
            f0_nsf_dir = os.path.join(self.exp_dir, "2b-f0nsf")
            os.makedirs(f0_coarse_dir, exist_ok=True)
            os.makedirs(f0_nsf_dir, exist_ok=True)
            
            # 获取输入文件
            wav_dir = os.path.join(self.exp_dir, "1_16k_wavs")
            if not os.path.exists(wav_dir):
                self.logger.log(f"❌ 错误: 16k音频目录 {wav_dir} 不存在")
                return False
            
            # 构建文件路径列表
            paths = []
            for name in sorted(os.listdir(wav_dir)):
                if not name.endswith('.wav') or 'spec' in name:
                    continue
                
                inp_path = os.path.join(wav_dir, name)
                opt_path1 = os.path.join(f0_coarse_dir, name)
                opt_path2 = os.path.join(f0_nsf_dir, name)
                paths.append([inp_path, opt_path1, opt_path2])
            
            if not paths:
                self.logger.log("❌ 未找到待处理的音频文件")
                return False
            
            # 分片处理
            paths_part = paths[self.args.i_part::self.args.n_part]
            
            # 初始化F0提取器
            feature_input = FeatureInput(is_half=self.args.is_half)
            
            # 提取F0特征
            feature_input.extract_f0_features(
                paths=paths_part,
                logger=self.logger,
                f0_method="rmvpe"
            )
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ F0特征提取失败: {e}")
            self.logger.log(traceback.format_exc())
            return False
    
    def _extract_hubert_features(self) -> bool:
        """提取HuBERT特征"""
        try:
            # 初始化HuBERT特征提取器
            extractor = HubertFeatureExtractor(
                device=self.args.device,
                exp_dir=self.exp_dir,
                logger=self.logger,
                version=self.args.version,
                is_half=self.args.is_half
            )
            
            # 提取特征
            return extractor.extract_features_batch(
                n_part=self.args.n_part,
                i_part=self.args.i_part
            )
            
        except Exception as e:
            self.logger.log(f"❌ HuBERT特征提取失败: {e}")
            self.logger.log(traceback.format_exc())
            return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RVC训练数据准备Pipeline - 整合音频预处理和特征提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整流程
  python rvc_training_pipeline.py -i /path/to/audio -e my_model
  
  # 跳过预处理，只做特征提取
  python rvc_training_pipeline.py -e my_model --skip-preprocess
  
  # 多GPU并行处理
  python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 0 -g 0
  python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 1 -g 1
  python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 2 -g 2
  python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 3 -g 3
        """
    )
    
    # 基本参数
    parser.add_argument("-i", "--inp_root", type=str, 
                       help="输入音频目录路径 (预处理时必需)")
    parser.add_argument("-e", "--exp_dir", type=str, required=True,
                       help="实验目录名称")
    
    # 音频预处理参数
    parser.add_argument("-sr", "--sample_rate", type=int, default=40000,
                       choices=[32000, 40000, 48000],
                       help="目标采样率 (默认: 40000)")
    parser.add_argument("--per", type=float, default=3.7,
                       help="音频切片长度(秒) (默认: 3.7)")
    parser.add_argument("--n_processes", type=int, default=None,
                       help="预处理进程数 (默认: CPU核心数)")
    parser.add_argument("--no_parallel", action="store_true",
                       help="禁用多进程处理")
    parser.add_argument("--skip_preprocess", action="store_true",
                       help="跳过音频预处理步骤")
    
    # 特征提取参数
    parser.add_argument("-v", "--version", type=str, default="v2",
                       choices=["v1", "v2"], help="模型版本 (默认: v2)")
    parser.add_argument("-n", "--n_part", type=int, default=1,
                       help="总分片数 (用于多GPU并行) (默认: 1)")
    parser.add_argument("-p", "--i_part", type=int, default=0,
                       help="当前分片索引 (默认: 0)")
    parser.add_argument("-g", "--i_gpu", type=int, default=0,
                       help="GPU设备ID (默认: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cpu", "cuda", "mps", "privateuseone"],
                       help="计算设备 (默认: cuda)")
    parser.add_argument("--is_half", action="store_true",
                       help="使用半精度浮点数")
    
    # 训练参数
    parser.add_argument("-f0", "--if_f0", action="store_true", default=True,
                       help="是否使用F0特征 (默认: True)")
    parser.add_argument("--no_f0", action="store_true",
                       help="不使用F0特征")
    parser.add_argument("-s", "--spk_id", type=int, default=0,
                       help="说话人ID (默认: 0)")
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if args.no_f0:
        args.if_f0 = False
    
    # 验证参数
    if not args.skip_preprocess and not args.inp_root:
        parser.error("当不跳过预处理时，必须指定 --inp_root")
    
    if args.n_part <= 0 or args.i_part < 0 or args.i_part >= args.n_part:
        parser.error(f"无效的分片配置: i_part={args.i_part}, n_part={args.n_part}")
    
    return args


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 创建并运行Pipeline
        pipeline = RVCTrainingPipeline(args)
        success = pipeline.run()
        
        # 返回适当的退出码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()