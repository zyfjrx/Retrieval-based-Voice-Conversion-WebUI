# RVC 训练 Pipeline

这是一个整合了模型训练和索引构建的统一训练pipeline，将原本的 `train.py` 和 `train_index.py` 合并为一个完整的训练流程。

## 功能特性

### 核心功能
- **模型训练**: 基于原始 `train.py` 的完整训练流程
- **索引构建**: 自动进行 FAISS 索引训练和保存
- **统一配置**: 集中管理所有训练参数
- **流程控制**: 支持跳过特定步骤或仅执行部分流程

### 改进功能
- **配置验证**: 自动验证训练参数的有效性
- **错误处理**: 完善的错误处理和恢复机制
- **进度跟踪**: 详细的训练进度和状态信息
- **日志管理**: 分级日志输出和文件记录
- **资源管理**: 智能的GPU和内存使用管理

## 使用方法

### 基本用法

```bash
# 完整训练pipeline（模型训练 + 索引构建）
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -bs 32 -te 300 -se 100
```

### 高级用法

```bash
# 仅进行模型训练
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -bs 32 -te 300 -se 100 --skip-index

# 仅进行索引训练
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 --skip-training

# 使用预训练模型
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -pg pretrain/G_40k.pth -pd pretrain/D_40k.pth

# 多GPU训练
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -g 0-1-2-3 -bs 64

# 启用错误容忍模式
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 --continue-on-error
```

## 参数说明

### 必需参数
- `-e, --exp_dir`: 实验目录名（logs目录下的文件夹名）
- `-sr, --sample_rate`: 采样率 (32k/40k/48k)
- `-f0, --if_f0`: 是否使用F0作为模型输入 (0/1)

### 训练参数
- `-bs, --batch_size`: 批次大小 (默认: 4)
- `-te, --total_epoch`: 总训练轮数 (默认: 300)
- `-se, --save_every_epoch`: 保存检查点的频率 (默认: 50)
- `-g, --gpus`: GPU设备号，用-分隔 (默认: 0)
- `-v, --version`: 模型版本 (v1/v2，默认: v2)

### 预训练模型
- `-pg, --pretrain_g`: 预训练生成器模型路径
- `-pd, --pretrain_d`: 预训练判别器模型路径

### 流程控制
- `--skip-training`: 跳过模型训练，仅进行索引训练
- `--skip-index`: 跳过索引训练，仅进行模型训练
- `--continue-on-error`: 遇到错误时继续执行后续步骤

### 其他选项
- `-l, --if_latest`: 是否只保存最新的G/D模型文件 (默认: 0)
- `-c, --if_cache_data_in_gpu`: 是否将数据集缓存到GPU内存 (默认: 0)
- `-sw, --save_every_weights`: 保存检查点时是否在weights目录保存提取的模型 (默认: 0)
- `-o, --outside_index_root`: 外部索引根目录
- `--log-level`: 日志级别 (DEBUG/INFO/WARNING/ERROR，默认: INFO)

## 输出目录结构

训练完成后，会在 `logs/{exp_dir}` 目录下生成以下文件：

```
logs/{exp_dir}/
├── config.json              # 训练配置文件
├── G_*.pth                   # 生成器检查点
├── D_*.pth                   # 判别器检查点
├── added_*.index             # FAISS索引文件
├── total_fea.npy            # 特征文件
└── weights/                  # 提取的模型权重（如果启用）
    ├── {exp_dir}.pth
    └── {exp_dir}.index
```

## 与原始脚本的对比

### 优势
1. **统一接口**: 一个命令完成完整训练流程
2. **参数验证**: 自动检查配置参数的有效性
3. **错误处理**: 更好的错误恢复和继续机制
4. **进度跟踪**: 实时显示训练进度和状态
5. **资源优化**: 更智能的GPU和内存管理
6. **日志管理**: 结构化的日志输出和文件记录

### 兼容性
- 完全兼容原始 `train.py` 和 `train_index.py` 的所有功能
- 支持所有原有的命令行参数
- 生成的模型文件格式完全一致

## 注意事项

1. **数据准备**: 确保已完成数据预处理和特征提取
2. **GPU内存**: 根据GPU内存调整批次大小
3. **磁盘空间**: 确保有足够的磁盘空间保存检查点
4. **配置文件**: 确保对应版本的配置文件存在于 `configs/` 目录

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 -bs 16
   ```

2. **配置文件不存在**
   ```bash
   # 检查configs目录下是否有对应版本和采样率的配置文件
   ls configs/v2/40k.json
   ```

3. **预训练模型路径错误**
   ```bash
   # 确保预训练模型文件存在
   ls pretrain/G_40k.pth pretrain/D_40k.pth
   ```

### 调试模式

```bash
# 启用详细日志
python rvc_training_pipeline.py -e my_model -sr 40k -f0 1 --log-level DEBUG
```

## 性能优化建议

1. **多GPU训练**: 使用多个GPU可以显著提升训练速度
2. **数据缓存**: 如果GPU内存充足，启用数据缓存可以减少I/O开销
3. **批次大小**: 在GPU内存允许的情况下，增大批次大小可以提升训练效率
4. **检查点频率**: 适当调整保存频率，平衡训练速度和数据安全

---

这个训练pipeline简化了RVC模型的训练流程，提供了更好的用户体验和错误处理能力。如有问题，请检查日志输出或使用调试模式获取更多信息。