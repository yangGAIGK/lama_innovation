# -*- coding: utf-8 -*-
"""
壁画级联预处理净化脚本
四阶段流水线：白斑剔除 → 织补 → 裂纹淡化 → 残差保真融合
用于 LaMa 图像修复模型前的图像预处理。
"""

import cv2
import numpy as np


def guided_filter(I, p, r, eps):
    """
    导向滤波（Guided Filter）实现。
    参考论文: He et al., "Guided Image Filtering", ECCV 2010.

    参数:
        I: 引导图 (H, W), float32, 范围 0~1
        p: 输入图 (H, W), float32, 范围 0~1（此处与 I 相同则为自导向滤波）
        r: 滤波半径
        eps: 正则化参数，防止 a_k 过大
    返回:
        q: 滤波输出 (H, W), float32, 范围 0~1
    """
    # 盒式滤波核尺寸
    d = 2 * r + 1

    # 均值: mean_I, mean_p, mean_Ip, mean_II
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (d, d))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (d, d))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (d, d))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (d, d))

    # 局部方差 cov_Ip = mean_Ip - mean_I * mean_p
    # 局部方差 var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # a = cov_Ip / (var_I + eps), b = mean_p - a * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 对 a, b 做盒式平滑
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (d, d))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (d, d))

    # q = mean_a * I + mean_b
    q = mean_a * I + mean_b
    return q


def preprocess_mural(image_path, top_hat_thresh=20, inpaint_radius=3,
                     guided_r=3, guided_eps=0.02, fusion_alpha=0.4):
    """
    壁画四阶段级联预处理净化流水线。

    参数:
        image_path:    输入图像路径
        top_hat_thresh: 白顶帽变换二值化阈值 (默认 20, 范围 10~30)
        inpaint_radius: 快速行进法修复半径 (默认 3)
        guided_r:       导向滤波半径 (默认 3)
        guided_eps:     导向滤波正则化参数 ε (默认 0.02, 图像归一化后)
        fusion_alpha:   净化图在残差融合中的权重 (默认 0.4)

    返回:
        output: 最终净化后的 RGB 图像 (uint8, H×W×3)
    """
    # 读取原始图像 (中文路径用 imdecode 兼容)
    src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # ==========================================================================
    # 阶段一：空间域白斑检测（形态学白顶帽变换）
    # 原理: f_tophat = f - (f ∘ b), 即原图减去开运算结果,
    #       开运算会"打开"亮的小区域, 相减后得到比邻域亮且小于结构元的斑点
    # ==========================================================================
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 半径为 4 的椭圆/圆形结构元 (直径 = 2*4+1 = 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # 白顶帽变换: 提取孤立亮斑
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # 二值化: 阈值截断得到白斑掩码 (阈值越高越严格, 只保留最亮的斑点)
    _, speckle_mask = cv2.threshold(tophat, top_hat_thresh, 255, cv2.THRESH_BINARY)

    # ==========================================================================
    # 阶段二：局部快速"织补"（快速行进法 FMM 修复）
    # TELEA (Fast Marching Method): 从掩码边缘向内推进,
    # 用邻域已知像素的加权平均填充未知区域, 修复半径 3 像素
    # ==========================================================================
    speckle_mask_u8 = speckle_mask.astype(np.uint8)
    inpainted = cv2.inpaint(src, speckle_mask_u8, inpaint_radius, cv2.INPAINT_TELEA)

    # ==========================================================================
    # 阶段三：通道剥离与导向滤波（LAB 空间裂纹淡化）
    # LAB 空间将亮度 L 与色彩 A/B 解耦, 仅对 L 通道做边缘保留滤波,
    # 既平滑裂纹又不污染色彩信息
    # ==========================================================================
    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    L, A, B_ch = cv2.split(lab)

    # 导向滤波要求输入在 0~1 范围, 滤波后还原到 0~255
    L_norm = L.astype(np.float32) / 255.0
    L_filtered_norm = guided_filter(L_norm, L_norm, guided_r, guided_eps)
    L_filtered = np.clip(L_filtered_norm * 255.0, 0, 255).astype(np.uint8)

    # 合并新 L 通道与原 A、B 通道, 转回 BGR
    lab_clean = cv2.merge([L_filtered, A, B_ch])
    clean_rgb = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)

    # ==========================================================================
    # 阶段四：残差保真融合（质感保留）
    # 完全净化图可能过度平滑 ("塑料磨皮感"),
    # 通过加权融合保留部分原始纹理: output = α * clean + (1-α) * original
    # α=0.4 意味着偏向保留原图质感, 同时引入净化信息
    # ==========================================================================
    output = cv2.addWeighted(clean_rgb, fusion_alpha, src, 1.0 - fusion_alpha, 0)

    return output


def preprocess_mural_file(image_path, output_path=None):
    """
    读取壁画图像, 执行预处理, 保存结果。

    参数:
        image_path: 输入图像路径
        output_path: 输出路径, 默认为输入路径同目录下加 _preprocessed 后缀
    """
    if output_path is None:
        import os
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_preprocessed{ext}"

    result = preprocess_mural(image_path)

    # cv2.imwrite 不支持非 ASCII 路径, 使用 imencode 写入
    ext = output_path.rsplit('.', 1)[-1]
    ok, buf = cv2.imencode(f".{ext}", result)
    if ok:
        buf.tofile(output_path)
        print(f"预处理完成: {output_path}")
    else:
        print(f"[错误] 图像编码失败: {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="壁画级联预处理净化脚本")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("-o", "--output", default=None, help="输出图像路径")
    parser.add_argument("--thresh", type=int, default=20,
                        help="白顶帽二值化阈值 (默认 20, 范围 10~30)")
    parser.add_argument("--inpaint_radius", type=int, default=3,
                        help="FMM 修复半径 (默认 3)")
    parser.add_argument("--guided_r", type=int, default=3,
                        help="导向滤波半径 (默认 3)")
    parser.add_argument("--guided_eps", type=float, default=0.02,
                        help="导向滤波正则化 ε (默认 0.02)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="残差融合中净化图权重 (默认 0.4)")

    args = parser.parse_args()
    preprocess_mural_file(
        args.image,
        output_path=args.output,
    )
