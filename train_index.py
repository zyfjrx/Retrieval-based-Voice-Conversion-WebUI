#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC训练特征索引独立脚本
从infer-web.py中提取的train_index函数逻辑
"""

import os
import sys
import argparse
import platform
import traceback
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from dotenv import load_dotenv
from multiprocessing import cpu_count

# 设置工作目录和路径
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

def get_n_cpu(n_cpu=None):
    """
    获取CPU核心数
    
    Args:
        n_cpu: 指定的CPU核心数，如果为None或0则自动检测
    
    Returns:
        int: CPU核心数
    """
    if n_cpu is None or n_cpu == 0:
        return cpu_count()
    return n_cpu

def train_index(exp_dir1, version19="v2", outside_index_root=None, n_cpu=None):
    """
    训练特征索引
    
    Args:
        exp_dir1: 实验目录名称
        version19: 模型版本 ("v1" 或 "v2")
        outside_index_root: 外部索引根目录
        n_cpu: CPU核心数，如果为None则自动检测
    """
    
    # 获取CPU核心数
    actual_n_cpu = get_n_cpu(n_cpu)
    print(f"使用CPU核心数: {actual_n_cpu}")
    
    # 设置实验目录
    exp_dir = "logs/%s" % exp_dir1
    os.makedirs(exp_dir, exist_ok=True)
    
    # 确定特征目录
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    
    # 检查特征目录是否存在
    if not os.path.exists(feature_dir):
        print("错误: 请先进行特征提取!")
        return False
        
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        print("错误: 请先进行特征提取！")
        return False
    
    print(f"开始训练索引，找到 {len(listdir_res)} 个特征文件")
    
    # 加载所有特征文件
    npys = []
    for name in sorted(listdir_res):
        feature_path = "%s/%s" % (feature_dir, name)
        print(f"加载特征文件: {feature_path}")
        phone = np.load(feature_path)
        npys.append(phone)
    
    # 合并所有特征
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    print(f"特征矩阵形状: {big_npy.shape}")
    
    # 如果特征数量过多，使用K-means聚类降维
    if big_npy.shape[0] > 2e5:
        print(f"特征数量过多 ({big_npy.shape[0]})，使用K-means聚类到10k个中心点")
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * actual_n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
            print(f"K-means聚类完成，新的特征矩阵形状: {big_npy.shape}")
        except Exception as e:
            print(f"K-means聚类失败: {e}")
            print(traceback.format_exc())
    
    # 保存总特征文件
    total_fea_path = "%s/total_fea.npy" % exp_dir
    np.save(total_fea_path, big_npy)
    print(f"保存总特征文件: {total_fea_path}")
    
    # 计算IVF参数
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print(f"特征维度: {256 if version19 == 'v1' else 768}, IVF聚类数: {n_ivf}")
    
    # 创建FAISS索引
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    
    print("开始训练索引...")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    
    # 保存训练后的索引
    trained_index_path = (
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    faiss.write_index(index, trained_index_path)
    print(f"保存训练索引: {trained_index_path}")
    
    print("添加特征到索引...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
        if i % (batch_size_add * 10) == 0:
            print(f"已添加 {i + batch_size_add}/{big_npy.shape[0]} 个特征")
    
    # 保存最终索引
    final_index_path = (
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    faiss.write_index(index, final_index_path)
    print(f"成功构建索引: {final_index_path}")
    
    # 尝试链接到外部索引目录
    # if outside_index_root and os.path.exists(outside_index_root):
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        external_index_path = (
            "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (
                outside_index_root,
                exp_dir1,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            )
            )
            # 如果目标文件已存在，先删除
        if os.path.exists(external_index_path):
            os.remove(external_index_path)
            
        link(final_index_path, external_index_path)
        print(f"成功链接索引到外部目录: {external_index_path}")
    except Exception as e:
        print(f"链接索引到外部目录失败: {e}")
    
    print("索引训练完成！")
    return True

def main():
    parser = argparse.ArgumentParser(description="RVC训练特征索引")
    parser.add_argument("-e", "--exp_dir", required=True, help="实验目录名称")
    parser.add_argument("-v", "--version", default="v2", choices=["v1", "v2"], help="模型版本")
    parser.add_argument("-o", "--outside_index_root", help="外部索引根目录")
    parser.add_argument("-c", "--n_cpu", type=int, help="CPU核心数，如果不指定则自动检测")
    
    args = parser.parse_args()
    
    # 如果没有指定外部索引目录，尝试从环境变量获取
    outside_index_root = args.outside_index_root or os.getenv("outside_index_root")
    
    print(f"实验目录: {args.exp_dir}")
    print(f"模型版本: {args.version}")
    print(f"外部索引目录: {outside_index_root or '未设置'}")
    if args.n_cpu:
        print(f"指定CPU核心数: {args.n_cpu}")
    else:
        print("CPU核心数: 自动检测")
    
    success = train_index(args.exp_dir, args.version, outside_index_root, args.n_cpu)
    
    if success:
        print("\n索引训练成功完成！")
        sys.exit(0)
    else:
        print("\n索引训练失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()