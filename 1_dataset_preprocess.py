import multiprocessing
import os
import sys
import argparse
import traceback
from scipy import signal
import librosa
import numpy as np
from scipy.io import wavfile
from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer
from multiprocessing import cpu_count
import json  # 添加 json 导入用于配置生成
import pathlib  # 添加 pathlib 导入用于路径检查
import shutil  # 添加 shutil 导入
# 版本配置列表
version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]
now_dir = os.getcwd()
sys.path.append(now_dir)

def println(strr, f):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

def get_n_cpu(n_cpu=None):
    if n_cpu is None or n_cpu == 0:
        return cpu_count()
    return n_cpu

def load_config_json(ff=None):
    """
    加载所有配置文件到字典中
    """
    d = {}
    for config_file in version_config_list:
        # 构建配置文件路径
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
                println(f"警告: 配置文件 {config_source_path} 不存在",ff)
                continue
        
        # 读取配置文件
        try:
            with open(config_inuse_path, "r", encoding="utf-8") as f:
                d[config_file] = json.load(f)
        except Exception as e:
            println(f"错误: 无法读取配置文件 {config_inuse_path}: {e}",ff)
            continue

    return d

def generate_config(exp_dir, version="v2", sr_key="40k",ff=None):
    """
    生成训练配置文件 config.json
    
    Args:
        exp_dir: 实验目录路径
        version: 模型版本 ("v1" 或 "v2")
        sr_key: 采样率 ("32k", "40k", "48k")
    """
    # 加载配置文件
    json_config = load_config_json(ff)
    
    # 确定配置文件路径
    if version == "v1" or sr_key == "40k":
        config_path = f"v1/{sr_key}.json"
    else:
        config_path = f"v2/{sr_key}.json"
    
    # 检查配置是否存在
    if config_path not in json_config:
        println(f"❌ 错误: 配置文件 {config_path} 不存在",ff)
        return False
    
    # 确保实验目录存在
    if not os.path.isabs(exp_dir):
        if exp_dir.startswith("logs/"):
            exp_dir = os.path.abspath(exp_dir)
        else:
            exp_dir = os.path.abspath(os.path.join(os.getcwd(), "logs", exp_dir))
    
    # 配置文件保存路径
    config_save_path = os.path.join(exp_dir, "config.json")
    
    # 检查配置文件是否已存在
    if not pathlib.Path(config_save_path).exists():
        try:
            # 创建目录（如果不存在）
            os.makedirs(exp_dir, exist_ok=True)
            
            # 写入配置文件
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(
                    json_config[config_path],
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                f.write("\n")
            
            println(f"✅ 成功生成配置文件: {config_save_path}",ff)
            return True
            
        except Exception as e:
            println(f"❌ 生成配置文件失败: {e}",ff)
            return False
    else:
        println(f"ℹ️  配置文件已存在: {config_save_path}",ff)
        return True

class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0, f):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println("%s\t-> Success" % path, f)
        except:
            println("%s\t-> %s" % (path, traceback.format_exc()), f)

    def pipeline_mp(self, infos, f, noparallel):
        for path, idx0 in infos:
            self.pipeline(path, idx0, f)

    def pipeline_mp_inp_dir(self, inp_root, n_p, f, noparallel):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p], f, noparallel)
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p], f, noparallel)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc(), f)


def preprocess_trainset():
    # 在函数内部解析参数
    parser = argparse.ArgumentParser(description="Dataset preprocessing for RVC training")
    parser.add_argument("-i", "--inp_root", dest="inp_root", type=str, required=True, help="Input root directory")
    parser.add_argument("-sr","--sample_rate", type=int, default=40000, help="Sample rate")
    parser.add_argument("--n_p", type=int, default=0, help="Number of processes")
    parser.add_argument("-e","--exp_dir", type=str, required=True, help="Experiment directory name")
    parser.add_argument("--noparallel", type=str, default="False", help="Whether to disable parallel processing (True/False)")
    parser.add_argument("--per", type=float, default=3.7, help="Segment length in seconds")
    parser.add_argument("-v","--version", type=str, default="v2", help="version (v1 or v2)")
    args = parser.parse_args()
    
    # 提取参数
    inp_root = args.inp_root
    sr = args.sample_rate
    sr_key = f"{sr // 1000}k"
    n_p = get_n_cpu(args.n_p)
    exp_dir_name = args.exp_dir
    noparallel = args.noparallel == "True"
    per = args.per
    version = args.version
    # print("所有参数:")
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
    # 创建实验目录
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Created experiment directory: {exp_dir}")
    else:
        print(f"Experiment directory already exists: {exp_dir}")
    
    # 打开日志文件
    f = open("%s/preprocess.log" % exp_dir, "a+")
    
    try:
        pp = PreProcess(sr, exp_dir, per)
        println("start preprocess", f)
        pp.pipeline_mp_inp_dir(inp_root, n_p, f, noparallel)
        println("end preprocess", f)
        generate_config(exp_dir,version=version, sr_key=sr_key,ff=f)
        println("end write confing", f)
    finally:
        f.close()


if __name__ == "__main__":
    preprocess_trainset()
    # d = load_config_json()
    # print(d)