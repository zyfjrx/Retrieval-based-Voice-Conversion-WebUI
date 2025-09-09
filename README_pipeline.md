# RVC训练数据准备Pipeline

这个脚本整合了RVC训练数据准备的完整流程，包括音频预处理和特征提取。

## 功能特性

✅ **音频预处理**
- 音频切片和标准化
- 高通滤波处理
- 多采样率支持 (32k/40k/48k)
- 多进程并行处理

✅ **F0特征提取**
- 使用RMVPE算法提取F0特征
- 生成粗糙F0和精细F0特征
- 支持GPU加速

✅ **HuBERT特征提取**
- 支持v1/v2模型版本
- 自动设备选择 (CPU/CUDA/MPS/DirectML)
- 半精度浮点数支持

✅ **配置管理**
- 自动生成训练配置文件
- 智能配置文件管理
- 支持多版本配置

✅ **文件列表生成**
- 自动生成训练用filelist.txt
- 支持F0和非F0模式
- 包含静音数据处理

## 使用方法

### 1. 完整流程 (推荐)

```bash
# 基本用法
python rvc_training_pipeline.py -i /path/to/audio -e my_model

# 指定采样率和版本
python rvc_training_pipeline.py -i /path/to/audio -e my_model -sr 48000 -v v2

# 使用半精度和指定GPU
python rvc_training_pipeline.py -i /path/to/audio -e my_model --is_half -g 0
```

### 2. 跳过预处理 (已有预处理数据)

```bash
# 只做特征提取
python rvc_training_pipeline.py -e my_model --skip_preprocess

# 不使用F0特征
python rvc_training_pipeline.py -e my_model --skip_preprocess --no_f0
```

### 3. 多GPU并行处理

```bash
# GPU 0
python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 0 -g 0

# GPU 1
python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 1 -g 1

# GPU 2
python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 2 -g 2

# GPU 3
python rvc_training_pipeline.py -i /path/to/audio -e my_model -n 4 -p 3 -g 3
```

## 参数说明

### 基本参数
- `-i, --inp_root`: 输入音频目录路径 (预处理时必需)
- `-e, --exp_dir`: 实验目录名称 (必需)

### 音频预处理参数
- `-sr, --sample_rate`: 目标采样率 (32000/40000/48000, 默认: 40000)
- `--per`: 音频切片长度(秒) (默认: 3.7)
- `--n_processes`: 预处理进程数 (默认: CPU核心数)
- `--no_parallel`: 禁用多进程处理
- `--skip_preprocess`: 跳过音频预处理步骤

### 特征提取参数
- `-v, --version`: 模型版本 (v1/v2, 默认: v2)
- `-n, --n_part`: 总分片数 (用于多GPU并行, 默认: 1)
- `-p, --i_part`: 当前分片索引 (默认: 0)
- `-g, --i_gpu`: GPU设备ID (默认: 0)
- `--device`: 计算设备 (cpu/cuda/mps/privateuseone, 默认: cuda)
- `--is_half`: 使用半精度浮点数

### 训练参数
- `-f0, --if_f0`: 是否使用F0特征 (默认: True)
- `--no_f0`: 不使用F0特征
- `-s, --spk_id`: 说话人ID (默认: 0)

## 输出目录结构

```
logs/your_model_name/
├── 0_gt_wavs/          # 原始采样率音频文件
├── 1_16k_wavs/         # 16kHz重采样音频文件
├── 2a_f0/              # 粗糙F0特征 (如果使用F0)
├── 2b-f0nsf/           # 精细F0特征 (如果使用F0)
├── 3_feature768/       # HuBERT特征 (v2) 或 3_feature256/ (v1)
├── config.json         # 训练配置文件
├── filelist.txt        # 训练文件列表
└── pipeline.log        # 处理日志
```

## 注意事项

1. **依赖文件**: 确保以下文件存在
   - `assets/hubert/hubert_base.pt` (HuBERT模型)
   - `assets/rmvpe/rmvpe.pt` (RMVPE模型)
   - `logs/mute/` (静音数据目录)

2. **内存使用**: HuBERT特征提取需要较大内存，建议:
   - 使用`--is_half`减少内存使用
   - 适当调整分片数量

3. **多GPU处理**: 
   - 确保每个GPU有足够显存
   - 分片索引从0开始，小于总分片数

4. **音频格式**: 支持常见音频格式 (.wav, .mp3, .flac, .m4a)

## 故障排除

### 常见错误

1. **模型文件不存在**
   ```
   错误: 模型文件 assets/hubert/hubert_base.pt 不存在
   ```
   解决: 从 [HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 下载模型文件

2. **CUDA内存不足**
   ```
   CUDA out of memory
   ```
   解决: 使用`--is_half`或增加分片数量

3. **音频文件格式错误**
   ```
   采样率必须为16000，当前为44100
   ```
   解决: 检查音频文件格式，确保预处理正确

### 日志查看

所有处理日志保存在 `logs/your_model_name/pipeline.log`，可以查看详细的处理过程和错误信息。

## 与原始脚本的对比

| 功能 | 原始方式 | Pipeline方式 |
|------|----------|-------------|
| 数据预处理 | `1_dataset_preprocess.py` | 集成在pipeline中 |
| 特征提取 | `2_extract_f0_feature.py` | 集成在pipeline中 |
| 配置管理 | 手动复制配置文件 | 自动生成和管理 |
| 错误处理 | 分散在各脚本 | 统一错误处理 |
| 进度跟踪 | 基本日志 | 详细进度和状态 |
| 参数管理 | 多个脚本不同参数 | 统一参数接口 |

## 性能优化建议

1. **使用SSD存储**: 提高I/O性能
2. **合理设置进程数**: 通常设为CPU核心数
3. **使用半精度**: 在支持的GPU上使用`--is_half`
4. **批量处理**: 对于大量数据，使用多GPU并行
5. **内存监控**: 监控内存使用，避免OOM错误