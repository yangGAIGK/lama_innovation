# -*- coding: utf-8 -*-
"""
批量壁画预处理脚本 — 对 images 目录下所有图片执行 preprocess_mural 流水线。
"""

import os
import sys
import time
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocess_mural import preprocess_mural_file

IMAGES_DIR = Path(r"D:\Study\大三下\science\tasl\Datasets\Datasets\patches_v2\images")
OUTPUT_DIR = Path(r"D:\Study\大三下\science\tasl\lama\output\batch")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    img_paths = sorted(
        f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in IMG_EXTS
    )
    total = len(img_paths)
    print(f"共找到 {total} 张图片，开始批量预处理...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    start_time = time.time()

    for i, img_path in enumerate(img_paths, 1):
        out_path = OUTPUT_DIR / f"{img_path.stem}_pipeline.png"
        try:
            preprocess_mural_file(str(img_path), output_path=str(out_path))
            success += 1
        except Exception as e:
            failed += 1
            print(f"  [失败] {img_path.name}: {e}")

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i)
            print(f"进度: {i}/{total} | 成功: {success} | 失败: {failed} | "
                  f"已用: {elapsed:.0f}s | 剩余: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n批量处理完成! {elapsed:.0f}s | 成功: {success} | 失败: {failed}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
