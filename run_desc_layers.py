"""
批量运行 extract_and_save：针对 desc_layer 在给定区间内循环执行。

用法示例（在项目根目录运行）：
  python run_desc_layers.py --start 0 --end 23 \
         --cfg config/base_dinov3_extract_D2S.yml \
         --model_name dinov3_vit7b16 --gpu 3

说明：
- 会依次使用 --desc_layer=<start..end> 调用：python -m extract_and_save ...
- start 和 end 为闭区间（包含两端）。
- 其他未列出的参数可直接跟在命令后面，会原样传递给 extract_and_save。
"""

import argparse
import os
import sys
import subprocess
from typing import List


def _inclusive_range(start: int, end: int, step: int = 1):
    if step == 0:
        raise ValueError("step 不能为 0")
    if start <= end and step < 0:
        step = -step  # 保证正向遍历时 step 为正
    if start >= end and step > 0:
        step = -step  # 保证反向遍历时 step 为负
    cur = start
    if step > 0:
        while cur <= end:
            yield cur
            cur += step
    else:
        while cur >= end:
            yield cur
            cur += step


def resolve_cfg_path(cfg_arg: str, root_dir: str) -> str:
    # 如果传入的是绝对路径，直接返回
    if os.path.isabs(cfg_arg):
        return cfg_arg
    # 优先按相对项目根目录解析
    candidate = os.path.join(root_dir, cfg_arg)
    if os.path.exists(candidate):
        return candidate
    # 兼容可能的 "configs/" 与 "config/" 路径差异
    alt_dirs = ["config", "configs"]
    for d in alt_dirs:
        candidate = os.path.join(root_dir, d, os.path.basename(cfg_arg))
        if os.path.exists(candidate):
            return candidate
    # 若仍不存在，返回原始值（让下游报错以便用户修正）
    return cfg_arg


def main(argv: List[str] = None):
    root_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="循环执行 extract_and_save，针对 desc_layer 指定区间")
    parser.add_argument("--start", type=int, required=True, help="起始层（包含）")
    parser.add_argument("--end", type=int, required=True, help="结束层（包含）")
    parser.add_argument("--step", type=int, default=1, help="步长（默认 1）")

    # 与原命令一致的关键参数
    parser.add_argument("--cfg", type=str, default=os.path.join(root_dir, "config", "base_dinov3_extract_D2S.yml"),
                        help="配置文件路径，默认使用项目内 config/base_dinov3_extract_D2S.yml")
    parser.add_argument("--model_name", type=str, default="dinov3_vit7b16", help="DINOv3 模型名称")
    parser.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES 值（字符串）")
    parser.add_argument("--dry_run", action="store_true", help="仅打印将要执行的命令，不实际运行")
    parser.add_argument("--stop_on_error", action="store_true", help="当某次运行失败时立即停止")

    # 接收其余未定义的参数并原样传递
    args, passthrough = parser.parse_known_args(argv)

    cfg_path = resolve_cfg_path(args.cfg, root_dir)

    print(f"项目根目录: {root_dir}")
    print(f"使用配置: {cfg_path}")
    print(f"模型: {args.model_name}, GPU: {args.gpu}")
    print(f"层范围: {args.start}..{args.end} (step={args.step})")
    if passthrough:
        print(f"透传额外参数: {' '.join(passthrough)}")

    py_exe = sys.executable  # 使用当前 Python 解释器

    for layer in _inclusive_range(args.start, args.end, args.step):
        cmd = [
            #py_exe, "-m", "extract_and_save",
            py_exe, "-m", "extract_and_save_vit7b",
            cfg_path,
            "--model_name", args.model_name,
            "--gpu", str(args.gpu),
            "--desc_layer", str(layer),
            "--save_desc_layer", str(layer),
        ] + passthrough

        print("\n===========================================")
        print(f"运行层: {layer}")
        print("命令: ", " ".join(cmd))
        print("===========================================\n")

        if args.dry_run:
            continue

        # 在项目根目录执行，确保 python -m extract_and_save 能找到模块
        proc = subprocess.run(cmd, cwd=root_dir)
        if proc.returncode != 0:
            print(f"[错误] desc_layer={layer} 运行失败，返回码 {proc.returncode}")
            if args.stop_on_error:
                print("已根据 --stop_on_error 选项停止后续运行。")
                sys.exit(proc.returncode)

    print("\n全部运行结束。")


if __name__ == "__main__":
    main()