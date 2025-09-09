from email.policy import default
import os
import sys
import traceback
import argparse
import parselmouth
now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import fairseq
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from infer.lib.audio import load_audio
logging.getLogger("numba").setLevel(logging.WARNING)

# n_part = int(sys.argv[1])
# i_part = int(sys.argv[2])
# i_gpu = sys.argv[3]
# os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
# exp_dir = sys.argv[4]
# is_half = sys.argv[5]
# f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr,f):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Extract F0 features for RVC training")
    parser.add_argument("--device", type=str,default="cuda", help="Device to use (cpu/cuda/mps/privateuseone)")
    parser.add_argument("-v", "--version", type=str, default="v2", choices=["v1", "v2"], help="Model version")
    parser.add_argument("-n", "--n_part", type=int, default=1, help="Total number of parts")
    parser.add_argument("-i", "--i_part", type=int, default=0, help="Current part index")
    parser.add_argument("-g", "--i_gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--is_half", type=str, default="True", help="Use half precision (True/False)")
    return parser.parse_args()


class FeatureInput(object):
    def __init__(self, is_half=True, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.is_half = is_half
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        # p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=self.is_half, device="cuda"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method,f):
        if len(paths) == 0:
            printt("no-f0-todo",f)
        else:
            printt("todo-f0-%s" % len(paths),f)
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path),f)
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()),f)


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


def validate_args(args):
    """验证命令行参数"""
    if not os.path.exists(f"{os.getcwd()}/logs/{args.exp_dir}"):
        raise ValueError(f"Experiment directory does not exist: {args.exp_dir}")
    
    if args.n_part <= 0 or args.i_part < 0 or args.i_part >= args.n_part:
        raise ValueError(f"Invalid part configuration: i_part={args.i_part}, n_part={args.n_part}")
    
    if args.is_half.lower() not in ['true', 'false']:
        raise ValueError(f"Invalid is_half value: {args.is_half}. Must be 'true' or 'false'")

def setup_directories(exp_dir):
    """设置并创建必要的目录"""
    directories = {
        'input': os.path.join(exp_dir, "1_16k_wavs"),
        'f0_coarse': os.path.join(exp_dir, "2a_f0"),
        'f0_nsf': os.path.join(exp_dir, "2b-f0nsf")
    }
    
    # 验证输入目录存在
    if not os.path.exists(directories['input']):
        raise FileNotFoundError(f"Input directory not found: {directories['input']}")
    
    # 创建输出目录
    for dir_path in [directories['f0_coarse'], directories['f0_nsf']]:
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def build_file_paths(directories):
    """构建文件路径列表"""
    paths = []
    input_dir = directories['input']
    
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith('.wav') or 'spec' in name:
            continue
            
        inp_path = os.path.join(input_dir, name)
        opt_path1 = os.path.join(directories['f0_coarse'], name)
        opt_path2 = os.path.join(directories['f0_nsf'], name)
        paths.append([inp_path, opt_path1, opt_path2])
    
    return paths

def extract_f0_features(paths, i_part, n_part, is_half, f0_method="rmvpe", log_file=None):
    """提取F0特征"""
    try:
        feature_input = FeatureInput(is_half=is_half)
        feature_input.go(paths[i_part::n_part], f0_method, log_file)
        return True
    except Exception as e:
        if log_file:
            printt(f"f0_extraction_failed: {str(e)}", log_file)
            printt(f"traceback: {traceback.format_exc()}", log_file)
        return False

def extract_hubert_features(device, exp_dir, version, is_half, n_part, i_part, log_file=None):
    """提取HuBERT特征"""
    try:
        with FeatureExtractor(
            device=device,
            exp_dir=exp_dir,
            version=version,
            is_half=is_half
        ) as extractor:
            if log_file:
                extractor.printt(" ".join(sys.argv))
            extractor.extract_features_batch(n_part=n_part, i_part=i_part)
        return True
    except Exception as e:
        if log_file:
            printt(f"hubert_extraction_failed: {str(e)}", log_file)
            printt(f"traceback: {traceback.format_exc()}", log_file)
        return False

def main():
    """主函数 - 优化版本"""
    log_file = None
    
    try:
        # 解析和验证参数
        args = parse_args()
        validate_args(args)
        
        # 设置基本变量
        device = args.device
        version = args.version
        n_part = args.n_part
        i_part = args.i_part
        i_gpu = args.i_gpu
        exp_dir_name = args.exp_dir
        is_half = args.is_half.lower() == 'true'
        exp_dir = os.path.join(now_dir, "logs", exp_dir_name)
        
        # 设置GPU环境
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
        
        # 初始化日志
        log_file = open(os.path.join(exp_dir, "extract_f0_feature.log"), "a+")
        printt(" ".join(sys.argv), log_file)
        printt(f"Starting feature extraction for experiment: {exp_dir_name}", log_file)
        
        # 设置目录结构
        directories = setup_directories(exp_dir)
        printt(f"Directories setup completed", log_file)
        
        # 构建文件路径
        paths = build_file_paths(directories)
        printt(f"Found {len(paths)} audio files to process", log_file)
        
        if len(paths) == 0:
            printt("Warning: No audio files found for processing", log_file)
            return
        
        # Step 1: F0特征提取
        printt("Step 1: Starting F0 feature extraction", log_file)
        f0_success = extract_f0_features(
            paths=paths,
            i_part=i_part,
            n_part=n_part,
            is_half=is_half,
            f0_method="rmvpe",
            log_file=log_file
        )
        
        if f0_success:
            printt("Step 1: F0 feature extraction completed successfully", log_file)
        else:
            printt("Step 1: F0 feature extraction failed", log_file)
            # 可以选择继续或退出
        
        # Step 2: HuBERT特征提取
        printt("Step 2: Starting HuBERT feature extraction", log_file)
        hubert_success = extract_hubert_features(
            device=device,
            exp_dir=exp_dir,
            version=version,
            is_half=is_half,
            n_part=n_part,
            i_part=i_part,
            log_file=log_file
        )
        
        if hubert_success:
            printt("Step 2: HuBERT feature extraction completed successfully", log_file)
        else:
            printt("Step 2: HuBERT feature extraction failed", log_file)
        
        # 总结
        if f0_success and hubert_success:
            printt("All feature extraction tasks completed successfully", log_file)
        else:
            printt("Some feature extraction tasks failed. Check logs for details.", log_file)
            
    except Exception as e:
        error_msg = f"Fatal error in main: {str(e)}"
        print(error_msg)
        if log_file:
            printt(error_msg, log_file)
            printt(f"Traceback: {traceback.format_exc()}", log_file)
        sys.exit(1)
        
    finally:
        # 确保日志文件被正确关闭
        if log_file:
            printt("Feature extraction process finished", log_file)
            log_file.close()

if __name__ == "__main__":
    main()