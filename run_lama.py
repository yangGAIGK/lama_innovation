# -*- coding: utf-8 -*-
"""
LaMa 图像修复脚本（直接 import 模式，绕过 Hydra CLI 中文路径问题）
"""
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

# ─────────────────────── 路径配置 ───────────────────────
IMAGES_DIR = Path(r"D:\Study\大三下\science\tasl\Datasets\Datasets\patches_v2\images")
MASKS_DIR  = Path(r"D:\Study\大三下\science\tasl\Datasets\Datasets\patches_v2\masks")
OUTPUT_DIR = Path(r"D:\Study\大三下\science\tasl\lama\output")
LAMA_DIR   = Path(r"D:\Study\大三下\science\tasl\lama")
TMP_DIR    = Path(r"D:\Study\大三下\science\tasl\lama_tmp_input")
# ────────────────────────────────────────────────────────


def setup_lama_path():
    lama_str = str(LAMA_DIR)
    if lama_str not in sys.path:
        sys.path.insert(0, lama_str)


def prepare_input(images_dir: Path, masks_dir: Path, tmp_dir: Path) -> int:
    """把 images/ + masks/ 按 LaMa 格式合并到 tmp_dir"""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in images_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print(f"找到 {len(img_files)} 张图片，准备输入目录...")
    matched = 0

    for img_path in img_files:
        stem = img_path.stem
        mask_path = next(
            (masks_dir / (stem + ext) for ext in [".png", ".jpg", ".jpeg"]
             if (masks_dir / (stem + ext)).exists()),
            None
        )
        if mask_path is None:
            print(f"  [跳过] {img_path.name}: 无对应 mask")
            continue

        img  = cv2.imdecode(np.fromfile(str(img_path),  dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  [跳过] {img_path.name}: 读取失败")
            continue

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imencode(".png", img)[1].tofile(str(tmp_dir / f"{stem}.png"))
        cv2.imencode(".png", mask_bin)[1].tofile(str(tmp_dir / f"{stem}_mask.png"))
        matched += 1

    print(f"准备完成: {matched} 对图片-mask 写入 {tmp_dir}")
    return matched


def run_lama_inpainting(indir: Path, outdir: Path):
    """直接调用 LaMa 内部逻辑，不经过 Hydra CLI"""
    setup_lama_path()

    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.training.data.datasets import make_default_val_dataset
    from saicinpainting.training.trainers import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = LAMA_DIR / "big-lama"
    checkpoint_path = model_path / "models" / "best.ckpt"

    # 读取训练配置
    train_config_path = model_path / "config.yaml"
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    # 加载模型
    print("加载 LaMa 模型...")
    model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location="cpu")
    model.freeze()
    model.to(device)

    # 构建预测配置（不走 Hydra CLI）
    predict_config = OmegaConf.create({
        "indir": str(indir) + os.sep,
        "outdir": str(outdir),
        "dataset": {
            "kind": "default",
            "img_suffix": ".png",
            "pad_out_to_modulo": 8,
        },
        "out_key": "inpainted",
        "out_ext": ".png",
        "refine": False,
    })

    outdir.mkdir(parents=True, exist_ok=True)

    dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
    print(f"数据集大小: {len(dataset)}")

    import tqdm
    for img_i in tqdm.trange(len(dataset), desc="LaMa 修复中"):
        mask_fname = dataset.mask_filenames[img_i]

        indir_norm = os.path.normpath(predict_config.indir) + os.sep
        mask_norm  = os.path.normpath(mask_fname)
        rel_path   = (mask_norm[len(indir_norm):]
                      if mask_norm.startswith(indir_norm)
                      else os.path.basename(mask_fname))

        cur_out_fname = outdir / (os.path.splitext(rel_path)[0] + predict_config.out_ext)
        cur_out_fname.parent.mkdir(parents=True, exist_ok=True)

        batch = default_collate([dataset[img_i]])
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch["mask"] = (batch["mask"] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad = batch.get("unpad_to_size", None)
            if unpad is not None:
                cur_res = cur_res[:unpad[0], :unpad[1]]

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        ext = os.path.splitext(str(cur_out_fname))[1]
        ok, buf = cv2.imencode(ext, cur_res)
        if ok:
            buf.tofile(str(cur_out_fname))
        else:
            print(f"  [警告] 编码失败: {cur_out_fname}")

    print(f"\n完成！结果保存在: {outdir}")


def cleanup(tmp_dir: Path):
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"已清理临时目录: {tmp_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=str(IMAGES_DIR))
    parser.add_argument("--masks_dir",  default=str(MASKS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no_cleanup", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir  = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    matched = prepare_input(images_dir, masks_dir, TMP_DIR)
    if matched == 0:
        print("没有匹配的图片-mask 对，退出")
        sys.exit(1)

    run_lama_inpainting(TMP_DIR, output_dir)

    if not args.no_cleanup:
        cleanup(TMP_DIR)
