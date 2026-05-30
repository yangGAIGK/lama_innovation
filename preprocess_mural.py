# -*- coding: utf-8 -*-
"""
壁画级联预处理净化脚本
四阶段流水线：白斑剔除 → 织补 → 裂纹淡化 → 残差保真融合
输出：5 图拼接对比（原图 + 四阶段各阶段效果）
用于 LaMa 图像修复模型前的图像预处理。
"""

import cv2
import numpy as np


def guided_filter(I, p, r, eps):
    """
    高度健壮的导向滤波实现（规避低半径下的数值饱和与边界蒙雾）。
    """
    d = 2 * r + 1

    # 【优化 1】显式指定边界处理 BORDER_REFLECT_101（对称反射），防止边缘均值污染
    mean_I  = cv2.boxFilter(I, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)
    mean_p  = cv2.boxFilter(p, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # 【优化 2】增加一个极其微小的 1e-6 防止 var_I + eps 极度接近 0 时 a 值产生数值暴增
    a = cov_Ip / (var_I + eps + 1e-6)
    b = mean_p - a * mean_I

    # 【优化 3】限制 a 的合理范围，防止局部线性系数过载，杜绝“蒙雾”和“塑料感”
    a = np.clip(a, -5.0, 5.0)

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (d, d), borderType=cv2.BORDER_REFLECT_101)

    q = mean_a * I + mean_b
    
    # 【优化 4】严格限制输出范围在 0~1，防止浮点数计算后发生越界
    return np.clip(q, 0.0, 1.0)


def stitch_horizontal(images, labels=None, gap=4, label_height=32):
    """
    将多张 BGR 图像水平拼接为一张大图，并在底部加标签。

    参数:
        images:         BGR 图像列表 (numpy arrays, H×W×3)
        labels:         每张图的标签文字列表 (None 则不显示)
        gap:            图间间隔 (像素)
        label_height:  标签栏高度 (像素)
    返回:
        stitched: 拼接后的 BGR 图像
    """
    n = len(images)
    h = max(img.shape[0] for img in images)
    total_h = h + (label_height if labels else 0)
    w = sum(img.shape[1] for img in images) + gap * (n - 1)

    canvas = np.full((total_h, w, 3), 255, dtype=np.uint8)

    x_offset = 0
    for i, img in enumerate(images):
        ih, iw = img.shape[:2]
        y_off = (h - ih) // 2
        canvas[y_off:y_off + ih, x_offset:x_offset + iw] = img

        if labels and i < len(labels):
            cv2.putText(canvas, labels[i],
                        (x_offset + 4, h + label_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                        cv2.LINE_AA)

        x_offset += iw + gap

    return canvas


def preprocess_mural(image_path, top_hat_thresh=80, inpaint_radius=12,
                     guided_r=100, guided_eps=0.001, fusion_alpha=0.3):
    """
    壁画四阶段级联预处理净化流水线。

    参数:
        image_path:      输入图像路径
        top_hat_thresh:  白顶帽变换二值化阈值 (默认 20, 范围 10~30)
        inpaint_radius:  快速行进法修复半径 (默认 3)
        guided_r:        导向滤波半径 (默认 3)
        guided_eps:      导向滤波正则化参数 ε (默认 0.02, 图像归一化后)
        fusion_alpha:    净化图在残差融合中的权重 (默认 0.4)

    返回:
        pipeline_images: 字典, 包含各阶段效果图 (BGR uint8)
            'original'      原图
            'mask'          白斑掩码 (单通道转 BGR)
            'stage2_inpaint' FMM 织补后
            'stage3_clean'   导向滤波后
            'stage4_final'   最终融合输出
    """
    src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # ==========================================================================
    # 阶段一：空间域白斑检测（形态学白顶帽变换）
    # ==========================================================================
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 半径为 4 的椭圆/圆形结构元 (直径 = 2*4+1 = 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # 白顶帽变换: 提取孤立亮斑
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # 二值化: 阈值截断得到白斑掩码
    _, speckle_mask = cv2.threshold(tophat, top_hat_thresh, 255, cv2.THRESH_BINARY)

    # 掩码可视化 (单通道转 BGR)
    mask_vis = cv2.cvtColor(speckle_mask, cv2.COLOR_GRAY2BGR)

    # ==========================================================================
    # 阶段二：局部快速"织补"（快速行进法 FMM 修复）
    # ==========================================================================
    speckle_mask_u8 = speckle_mask.astype(np.uint8)
    inpainted = cv2.inpaint(src, speckle_mask_u8, inpaint_radius, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    L, A, B_ch = cv2.split(lab)

    # 1. 导向滤波淡化裂纹
    L_norm = L.astype(np.float32) / 255.0
    L_filtered_norm = guided_filter(L_norm, L_norm, guided_r, guided_eps)
    L_filtered = np.clip(L_filtered_norm * 255.0, 0, 255).astype(np.uint8)
    
    # 2. 局部对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    L_enhanced = clahe.apply(L_filtered)

    # ------------------ 【新增：噪声防火墙】 ------------------
    # 使用双边滤波平滑细微颗粒，但保留结构线条
    # d=5: 像素邻域直径; sigmaColor=50: 色彩空间标准差; sigmaSpace=5: 坐标空间标准差
    L_smooth_for_features = cv2.bilateralFilter(L_enhanced, d=8, sigmaColor=50, sigmaSpace=5)
    # ------------------------------------------------------------

    # ------------------ 【核心大招：黑顶帽暗线加深】 ------------------
    # 注意：我们现在从 L_smooth_for_features 中提取暗线，而不是充满噪声的 L_enhanced
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(L_smooth_for_features, cv2.MORPH_BLACKHAT, kernel_line)
    
    blackhat_weighted = cv2.multiply(blackhat, 1.0) # 保持你调好的强度 1.0
    
    # 依然把提取到的干净暗线，加深回原始的 L_enhanced 中
    L_darkened = cv2.subtract(L_enhanced, blackhat_weighted)

    # ------------------ 【正确的带阈值 USM 锐化】 ------------------
    blur = cv2.GaussianBlur(L_darkened, (0, 0), 3.0)
    
    L_16s = L_darkened.astype(np.int16)
    blur_16s = blur.astype(np.int16)
    
    high_freq = L_16s - blur_16s
    
    # 【微调阈值】：将阈值从 10 稍微提高到 12 或 15，进一步防止平坦区域的微小波动被锐化
    mask = np.abs(high_freq) > 15  
    
    sharp_16s = L_16s.copy()
    # 锐化强度也可以稍微降低，比如从 1.5 降到 1.2
    sharp_16s[mask] = L_16s[mask] + high_freq[mask] * 1.2 
    
    L_sharp = np.clip(sharp_16s, 0, 255).astype(np.uint8)
    # ------------------------------------------------------------

    # 合并通道
    lab_clean = cv2.merge([L_sharp, A, B_ch])
    clean_rgb = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)
    # ==========================================================================
    # 阶段四：残差保真融合（质感保留）
    # ==========================================================================
    output = cv2.addWeighted(clean_rgb, fusion_alpha, inpainted, 1.0 - fusion_alpha, 0)

    return {
        'original':       src,
        'mask':           mask_vis,
        'stage2_inpaint': inpainted,
        'stage3_clean':   clean_rgb,
        'stage4_final':   output,
    }


def preprocess_mural_file(image_path, output_path=None, auto_name=False):
    """
    读取壁画图像, 执行预处理, 保存拼接后的五图对比结果。

    参数:
        image_path: 输入图像路径
        output_path: 输出路径, 若为 None 且 auto_name=False 则存到原图同目录加 _pipeline 后缀
        auto_name:   True 时自动在 output 目录下递增编号, 避免覆盖之前的输出
    """
    import os
    import glob

    if output_path is None or auto_name:
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1] or ".jpg"

        if output_path is None:
            out_dir = os.path.dirname(image_path) or "."
        else:
            out_dir = os.path.dirname(output_path) or "."

        if auto_name:
            # 扫描已有编号, 自动递增
            existing = glob.glob(os.path.join(out_dir, f"{img_stem}_pipeline_*.png"))
            max_n = 0
            for f in existing:
                name = os.path.splitext(os.path.basename(f))[0]  # e.g. 000009_pipeline_7
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_n = max(max_n, int(parts[1]))
            output_path = os.path.join(out_dir, f"{img_stem}_pipeline_{max_n + 1}.png")
        else:
            output_path = os.path.join(out_dir, f"{img_stem}_pipeline{ext}")
    else:
        # 补后缀
        base, curr_ext = os.path.splitext(output_path)
        if not curr_ext:
            _, src_ext = os.path.splitext(image_path)
            curr_ext = src_ext if src_ext else ".jpg"
            output_path = output_path + curr_ext

    # 自动创建输出目录
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stages = preprocess_mural(image_path)

    labels = [
        "Original",
        "Stage1: Speckle Mask",
        "Stage2: FMM Inpaint",
        "Stage3: Guided Filter",
        "Stage4: Blend(Stage3+Stage2)",
    ]

    stitched = stitch_horizontal(
        [stages['original'],
         stages['mask'],
         stages['stage2_inpaint'],
         stages['stage3_clean'],
         stages['stage4_final']],
        labels=labels,
        gap=4,
        label_height=36,
    )

    ext = output_path.rsplit('.', 1)[-1]
    ok, buf = cv2.imencode(f".{ext}", stitched)
    if ok:
        buf.tofile(output_path)
        print(f"预处理完成: {output_path}")
    else:
        print(f"[错误] 图像编码失败: {output_path}")

    return stitched


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="壁画级联预处理净化脚本（五图拼接输出）")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("-o", "--output", default=None, help="输出图像路径")
    parser.add_argument("--thresh", type=int, default=20,
                        help="白顶帽二值化阈值 (默认 20, 范围 10~30)")
    parser.add_argument("--inpaint_radius", type=int, default=8,
                        help="FMM 修复半径 (默认 3)")
    parser.add_argument("--guided_r", type=int, default=20,
                        help="导向滤波半径 (默认 3)")
    parser.add_argument("--guided_eps", type=float, default=0.02,
                        help="导向滤波正则化 ε (默认 0.02)")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="残差融合中净化图权重 (默认 0.4)")
    parser.add_argument("--auto_name", action="store_true", default=False,
                        help="自动递增编号命名输出文件, 避免覆盖")

    args = parser.parse_args()
    preprocess_mural_file(
        args.image,
        output_path=args.output,
        auto_name=args.auto_name,
    )
