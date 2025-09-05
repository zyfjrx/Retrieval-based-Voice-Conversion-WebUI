import os
import argparse
from random import shuffle

def generate_filelist(exp_dir, version="v2", if_f0=True, spk_id=0, sr="40k"):
    """
    生成训练用的filelist.txt文件 (使用绝对路径)
    
    Args:
        exp_dir: 实验目录路径 (例如: "logs/leijun1")
        version: 模型版本 ("v1" 或 "v2")
        if_f0: 是否使用F0 (True/False)
        spk_id: 说话人ID (默认0)
        sr: 采样率 ("32k", "40k", "48k")
    """
    
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath(os.getcwd())
    
    # 转换为绝对路径
    if not os.path.isabs(exp_dir):
        exp_dir = os.path.abspath(exp_dir)
    
    # 确保实验目录存在
    if not os.path.exists(exp_dir):
        print(f"错误: 实验目录 {exp_dir} 不存在")
        return False
    
    # 定义各个数据目录 (绝对路径)
    gt_wavs_dir = os.path.abspath(os.path.join(exp_dir, "0_gt_wavs"))
    feature_dir = os.path.abspath(os.path.join(exp_dir, "3_feature256" if version == "v1" else "3_feature768"))
    
    # 检查必要目录是否存在
    if not os.path.exists(gt_wavs_dir):
        print(f"错误: 原始音频目录 {gt_wavs_dir} 不存在")
        return False
    
    if not os.path.exists(feature_dir):
        print(f"错误: 特征目录 {feature_dir} 不存在")
        return False
    
    # 获取文件名集合
    gt_names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir) if name.endswith(".wav")])
    feature_names = set([name.split(".")[0] for name in os.listdir(feature_dir) if name.endswith(".npy")])
    
    if if_f0:
        f0_dir = os.path.abspath(os.path.join(exp_dir, "2a_f0"))
        f0nsf_dir = os.path.abspath(os.path.join(exp_dir, "2b-f0nsf"))
        
        if not os.path.exists(f0_dir):
            print(f"错误: F0目录 {f0_dir} 不存在")
            return False
        
        if not os.path.exists(f0nsf_dir):
            print(f"错误: F0NSF目录 {f0nsf_dir} 不存在")
            return False
        
        f0_names = set([name.split(".")[0] for name in os.listdir(f0_dir) if name.endswith(".npy")])
        f0nsf_names = set([name.split(".")[0] for name in os.listdir(f0nsf_dir) if name.endswith(".npy")])
        
        # 取所有目录的交集
        names = gt_names & feature_names & f0_names & f0nsf_names
    else:
        # 不使用F0时，只需要原始音频和特征的交集
        names = gt_names & feature_names
    
    if len(names) == 0:
        print("错误: 没有找到匹配的文件")
        return False
    
    print(f"找到 {len(names)} 个匹配的文件")
    
    # 构建文件路径列表 (使用绝对路径)
    opt = []
    for name in names:
        if if_f0:
            # 使用F0的格式: 原始音频|特征|F0|F0NSF|说话人ID (绝对路径)
            line = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
        else:
            # 不使用F0的格式: 原始音频|特征|说话人ID (绝对路径)
            line = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}"
        
        opt.append(line)
    
    # 添加静音数据 (mute data) - 使用绝对路径
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
    
    # 写入filelist.txt (绝对路径)
    filelist_path = os.path.abspath(os.path.join(exp_dir, "filelist.txt"))
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    
    print(f"成功生成 {filelist_path}")
    print(f"总共 {len(opt)} 行数据 (包含 {len(names)} 个训练样本 + 2个静音样本)")
    
    # 显示前几行示例
    print("\n生成的文件路径示例:")
    for i, line in enumerate(opt[:3]):
        print(f"  {i+1}: {line}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="生成RVC训练用的filelist.txt文件 (绝对路径版本)")
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="实验目录路径 (例如: logs/leijun1)")
    parser.add_argument("-v", "--version", type=str, default="v2", choices=["v1", "v2"], help="模型版本 (默认: v2)")
    parser.add_argument("-f0", "--if_f0", type=int, default=1, choices=[0, 1], help="是否使用F0 (1=是, 0=否, 默认: 1)")
    parser.add_argument("-s", "--spk_id", type=int, default=0, help="说话人ID (默认: 0)")
    parser.add_argument("-sr", "--sample_rate", type=str, default="40k", choices=["32k", "40k", "48k"], help="采样率 (默认: 40k)")
    
    args = parser.parse_args()
    
    print(f"当前工作目录: {os.path.abspath(os.getcwd())}")
    print(f"实验目录: {os.path.abspath(args.exp_dir)}")
    print(f"参数: version={args.version}, f0={bool(args.if_f0)}, spk_id={args.spk_id}, sr={args.sample_rate}")
    print("-" * 60)
    
    success = generate_filelist(
        exp_dir=args.exp_dir,
        version=args.version,
        if_f0=bool(args.if_f0),
        spk_id=args.spk_id,
        sr=args.sample_rate
    )
    
    if success:
        print("\n✅ filelist.txt 生成完成! (使用绝对路径)")
    else:
        print("\n❌ filelist.txt 生成失败!")
        exit(1)

if __name__ == "__main__":
    main()