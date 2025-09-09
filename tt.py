import os
import sys
import traceback
import argparse
from typing import Optional, Tuple
now_dir = os.getcwd()
sys.path.append(now_dir)
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


class FeatureExtractor:
    """HuBERT特征提取器"""
    
    def __init__(self, device: str, exp_dir: str, version: str = "v2", is_half: bool = False, 
                 model_path: str = "assets/hubert/hubert_base.pt"):
        """
        初始化特征提取器
        
        Args:
            device: 设备类型 (cpu/cuda/mps/privateuseone)
            exp_dir: 实验目录路径
            version: 模型版本 (v1/v2)
            is_half: 是否使用半精度
            model_path: HuBERT模型路径
        """
        self.exp_dir = exp_dir
        self.version = version
        self.is_half = is_half
        self.model_path = model_path
        
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
        
        # 初始化日志
        self.log_file = open(os.path.join(exp_dir, "extract_f0_feature.log"), "a+")
        
        # 加载模型
        self.model, self.saved_cfg = self._load_model()
    
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if "privateuseone" not in device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            import torch_directml
            device = torch_directml.device(torch_directml.default_device())
            
            # 设置DirectML前向传播
            def forward_dml(ctx, x, scale):
                ctx.scale = scale
                res = x.clone().detach()
                return res
            fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
        
        return device
    
    def _load_model(self) -> Tuple[torch.nn.Module, object]:
        """加载HuBERT模型"""
        self.printt(f"load model(s) from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            error_msg = (
                f"Error: Extracting is shut down because {self.model_path} does not exist, "
                "you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )
            self.printt(error_msg)
            raise FileNotFoundError(error_msg)
        
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.model_path], suffix=""
        )
        
        model = models[0]
        model = model.to(self.device)
        self.printt(f"move model to {self.device}")
        
        if self.is_half and self.device not in ["mps", "cpu"]:
            model = model.half()
        
        model.eval()
        return model, saved_cfg
    
    def printt(self, message: str):
        """打印并记录日志"""
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
    
    def readwave(self, wav_path: str, normalize: bool = False) -> torch.Tensor:
        """读取并预处理音频文件"""
        wav, sr = sf.read(wav_path)
        assert sr == 16000, f"Sample rate must be 16000, got {sr}"
        
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # 双声道转单声道
            feats = feats.mean(-1)
        assert feats.dim() == 1, f"Expected 1D tensor, got {feats.dim()}D"
        
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
                self.printt(f"Warning: {os.path.basename(wav_path)} contains NaN values")
                return None
            
            return feats
            
        except Exception as e:
            self.printt(f"Error processing {wav_path}: {str(e)}")
            self.printt(traceback.format_exc())
            return None
    
    def extract_features_batch(self, n_part: int = 1, i_part: int = 0):
        """批量提取特征"""
        if not os.path.exists(self.wav_path):
            self.printt(f"Error: Input directory {self.wav_path} does not exist")
            return
        
        # 获取待处理文件列表
        all_files = sorted([f for f in os.listdir(self.wav_path) if f.endswith(".wav")])
        todo = all_files[i_part::n_part]
        
        if len(todo) == 0:
            self.printt("no-feature-todo")
            return
        
        self.printt(f"all-feature-{len(todo)}")
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
                self.printt(f"now-{idx},all-{len(todo)},{file},{shape_str}")
        
        self.printt("all-feature-done")
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Extract features using HuBERT model")
    parser.add_argument("--device", type=str,default="cuda", help="Device to use (cpu/cuda/mps/privateuseone)")
    parser.add_argument("-n", "--n_part", type=int, default=1, help="Total number of parts")
    parser.add_argument("-i", "--i_part", type=int, default=0, help="Current part index")
    parser.add_argument("-g", "--i_gpu", type=int, default=0, help="GPU index (optional)")
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("-v", "--version", type=str, default="v2", choices=["v1", "v2"], help="Model version")
    parser.add_argument("--is_half", action="store_true", help="Use half precision")
    parser.add_argument("--model_path", type=str, default="assets/hubert/hubert_base.pt", help="HuBERT model path")
    
    return parser.parse_args()


def main():
    """主函数 - 兼容原有的命令行调用方式"""
    # 兼容原有的sys.argv调用方式
    if len(sys.argv) >= 4 and not sys.argv[1].startswith('-'):
        # 原有调用方式
        device = sys.argv[1]
        n_part = int(sys.argv[2])
        i_part = int(sys.argv[3])
        
        if len(sys.argv) == 7:
            exp_dir_name = sys.argv[4]
            version = sys.argv[5]
            is_half = sys.argv[6].lower() == "true"
            i_gpu = None
        else:
            i_gpu = int(sys.argv[4])
            exp_dir_name = sys.argv[5]
            version = sys.argv[6]
            is_half = sys.argv[7].lower() == "true"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    else:
        # 新的argparse方式
        args = parse_args()
        device = args.device
        n_part = args.n_part
        i_part = args.i_part
        exp_dir_name = args.exp_dir
        version = args.version
        is_half = args.is_half
        i_gpu = args.i_gpu
        exp_dir = "%s/logs/%s" % (now_dir, exp_dir_name)
        if i_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    
    # 使用特征提取器
    with FeatureExtractor(
        device=device,
        exp_dir=exp_dir,
        version=version,
        is_half=is_half
    ) as extractor:
        extractor.printt(" ".join(sys.argv))
        extractor.extract_features_batch(n_part=n_part, i_part=i_part)


if __name__ == "__main__":
    main()