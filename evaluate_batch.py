"""
evaluate_batch.py

批量评估脚本：
- 递归遍历指定 feats 根目录（或模型目录）下的所有特征子目录，
  对每个子目录（包含 sat_* 与 dro_* 文件）计算指标：top-1、top-5、top-10、top-1%、AP；
- 按模型名称分组输出 CSV 文件，文件名自动识别为 “<模型名称>.csv”，表头样式与示例一致。

用法示例：
  # 遍历整个 feats 根目录
  python -m evaluate_batch --save_dir data/jzl/0-pipei-dinov3/feats

  # 只遍历某个模型目录
  python -m evaluate_batch --save_dir data/jzl/0-pipei-dinov3/feats/dinov3_vits16

  # 指定输出目录、S2D 模式、以百分比写入 CSV
  python -m evaluate_batch --save_dir data/jzl/0-pipei-dinov3/feats \
      --out_dir outputs/eval_csv --S2D --percent

说明：
- 路径既可为绝对路径，也可为相对 HOME 的路径（与原 evaluate.py 风格兼容）。
- 不写死目录结构：只要某个目录下存在 sat_feat/sat_id/sat_name 和 dro_feat/dro_id/dro_name 即认定为特征目录。
"""

import argparse
import csv
import os
import os.path as osp
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


HOME = osp.expanduser('~')


def resolve_path(p: str) -> str:
    """将路径解析为绝对路径：绝对路径原样返回；相对路径按 HOME 解析。"""
    if not p:
        raise ValueError("路径不能为空")
    return p if osp.isabs(p) else osp.join(HOME, p)


def is_feature_dir(dirpath: str) -> bool:
    """判断目录是否包含保存的特征文件。"""
    return (
        osp.exists(osp.join(dirpath, 'sat_feat')) and
        osp.exists(osp.join(dirpath, 'sat_id')) and
        osp.exists(osp.join(dirpath, 'sat_name')) and
        osp.exists(osp.join(dirpath, 'dro_feat')) and
        osp.exists(osp.join(dirpath, 'dro_id')) and
        osp.exists(osp.join(dirpath, 'dro_name'))
    )


def find_feature_dirs(root: str) -> List[str]:
    """在 root 下递归查找所有包含特征文件的目录。"""
    feature_dirs = []
    if is_feature_dir(root):
        feature_dirs.append(root)
    for cur, _dirs, _files in os.walk(root):
        if is_feature_dir(cur):
            feature_dirs.append(cur)
    return sorted(set(feature_dirs))


def detect_model_and_layer(dirpath: str) -> Tuple[str, Optional[int]]:
    """
    根据目录推断 model 名与层编号：通常保存为 .../<model>/<layer>/。
    若最后一级不是纯数字，则提取其中的第一个整数；失败则返回 None。
    """
    layer_str = osp.basename(dirpath)
    model_name = osp.basename(osp.dirname(dirpath)) or "unknown_model"
    m = re.search(r'(\d+)', layer_str)
    layer = int(m.group(1)) if m else None
    return model_name, layer


def load_feat(savedir: str, view: str):
    feat = torch.load(osp.join(savedir, f'{view}_feat'), map_location='cpu').to(torch.float32)
    gid = torch.load(osp.join(savedir, f'{view}_id'), map_location='cpu')
    name = torch.load(osp.join(savedir, f'{view}_name'))
    return feat, gid, name


def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def eval_query(qf, ql, gf, gl):
    # 内积作为相似度（已归一化向量）
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()

    index = np.argsort(score)[::-1]

    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)

    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc


def cmc_at(CMC: torch.Tensor, k: int) -> float:
    """安全获取 top-k（k>=1），返回比例值。"""
    if k <= 0:
        k = 1
    idx = min(max(k - 1, 0), len(CMC) - 1)
    return float(CMC[idx].item())


def evaluate_dir(savedir: str, S2D: bool = False) -> Dict[str, float]:
    """评估单个目录，返回指标字典。"""
    gall_view = 'sat' if not S2D else 'dro'
    query_view = 'dro' if not S2D else 'sat'

    gall_feat, gid, _gname = load_feat(savedir, gall_view)
    query_feat, qid, _qname = load_feat(savedir, query_view)

    gl = gid.cpu().numpy()
    ql = qid.cpu().numpy()

    CMC = torch.IntTensor(len(gid)).zero_()
    ap_sum = 0.0
    for i in range(len(qid)):
        ap_tmp, CMC_tmp = eval_query(query_feat[i], ql[i], gall_feat, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap_sum += ap_tmp

    AP = ap_sum / max(len(qid), 1)
    CMC = (CMC.float() / max(len(qid), 1))

    top1 = cmc_at(CMC, 1)
    top5 = cmc_at(CMC, 5)
    top10 = cmc_at(CMC, 10)
    k1p = max(1, len(gid) // 100)
    top1p = cmc_at(CMC, k1p)

    return {
        'dim': int(query_feat.shape[-1]),
        'top1': float(top1),
        'top5': float(top5),
        'top10': float(top10),
        'top1p': float(top1p),
        'AP': float(AP),
    }


def write_csv_per_model(results: Dict[str, List[Tuple[Optional[int], Dict[str, float], str]]], out_dir: str, percent: bool = False):
    """为每个模型写一个 CSV 文件。文件名自动识别为 <模型名称>.csv；percent=True 时按百分比写入。"""
    os.makedirs(out_dir, exist_ok=True)
    for model, rows in results.items():
        rows_sorted = sorted(rows, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0, x[2]))
        # 清理模型名，避免不合法文件名字符
        safe_model = re.sub(r'[^\w\.-]+', '_', model)
        csv_path = osp.join(out_dir, f'{safe_model}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '层数(depth)', 'top-1', 'top-5', 'top-10', 'top-1%', 'AP'])
            for idx, (layer, metrics, d) in enumerate(rows_sorted, start=1):
                def fmt(x: float) -> str:
                    return f"{x*100:.2f}%" if percent else f"{x:.4f}"
                writer.writerow([
                    idx,
                    '' if layer is None else layer,
                    fmt(metrics['top1']),
                    fmt(metrics['top5']),
                    fmt(metrics['top10']),
                    fmt(metrics['top1p']),
                    fmt(metrics['AP']),
                ])
        print(f"[保存] 模型 {model} 的评估结果已写入: {csv_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='批量遍历/评估聚合特征并保存 CSV')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='目标目录：可以是 feats 根目录，也可以是某个模型目录，或具体层目录。支持绝对路径或相对 HOME。')
    parser.add_argument('--S2D', action='store_true', help='satellite -> drone 模式（默认 dro->sat）')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='CSV 输出目录；默认在目标目录下创建 eval_csv/')
    parser.add_argument('--percent', action='store_true', help='将指标按百分比格式写入 CSV（例如 12.34%）')

    args = parser.parse_args(argv)
    target = resolve_path(args.save_dir)

    if not osp.exists(target):
        print(f"[错误] 路径不存在: {target}")
        sys.exit(1)

    feature_dirs = find_feature_dirs(target)
    if not feature_dirs:
        print(f"[提示] 未找到特征文件，请确认目录: {target}")
        sys.exit(1)

    out_dir = args.out_dir or osp.join(target if osp.isdir(target) else osp.dirname(target), 'eval_csv')
    os.makedirs(out_dir, exist_ok=True)

    # 收集结果并按模型分组
    results_by_model: Dict[str, List[Tuple[Optional[int], Dict[str, float], str]]] = defaultdict(list)
    print(f"批量评估，共发现 {len(feature_dirs)} 个特征目录。")
    for d in feature_dirs:
        try:
            metrics = evaluate_dir(d, S2D=args.S2D)
            model, layer = detect_model_and_layer(d)
            results_by_model[model].append((layer, metrics, d))
            print(f"[完成] {model} / {osp.basename(d)} -> top-1:{metrics['top1']:.2%}, AP:{metrics['AP']:.2%}")
        except Exception as e:
            print(f"[跳过] 目录 {d} 评估失败：{e}")

    # 写 CSV
    write_csv_per_model(results_by_model, out_dir, percent=args.percent)
    print("全部评估完成。")


if __name__ == '__main__':
    main()