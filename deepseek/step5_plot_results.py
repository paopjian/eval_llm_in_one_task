#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5: 可视化 — 相似度分布图 + TPIR@FPIR 曲线 (中文字体)
输入: outputs/histograms_<tag>.npz (pos/neg直方图), tag默认multi优先
输出: outputs/similarity_distribution.png, outputs/tpir_fpir_curve.png
"""
import os
import json
import argparse
import numpy as np
import faireval_lib as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default=None,
                    help='直方图tag: single|multi (默认自动选择multi优先)')
    args = ap.parse_args()
    os.makedirs(fl.OUT_DIR, exist_ok=True)

    if args.tag:
        npz = os.path.join(fl.OUT_DIR, 'histograms_%s.npz' % args.tag)
        jfn = os.path.join(fl.OUT_DIR, 'metrics_%s.json' % args.tag)
    else:
        for t in ('multi', 'single'):
            p = os.path.join(fl.OUT_DIR, 'histograms_%s.npz' % t)
            if os.path.exists(p):
                npz, jfn, args.tag = p, os.path.join(fl.OUT_DIR,
                                                     'metrics_%s.json' % t), t
                break
        else:
            raise SystemExit('未找到 outputs/histograms_*.npz, 请先运行 step2/step3')
    data = np.load(npz)
    ph, nh = data['pos_hist'], data['neg_hist']
    nbins = int(data['nbins'])
    tpos, tneg = int(data['total_pos']), int(data['total_neg'])
    with open(jfn) as f:
        meta = json.load(f)

    print('=' * 78)
    print('Step5 可视化  (数据: %s, %s)' % (os.path.basename(npz), meta.get('method')))
    print('=' * 78)
    metrics = fl.compute_metrics(ph, nh, tpos, tneg,
                                 [1e-5, 1e-4, 1e-3, 1e-2], nbins)
    font = fl.setup_chinese_font()
    print('  中文字体: %s' % font)
    p1 = fl.plot_distributions(ph, nh, tpos, tneg, metrics, nbins,
                               os.path.join(fl.OUT_DIR, 'similarity_distribution.png'))
    p2 = fl.plot_curve(metrics, os.path.join(fl.OUT_DIR, 'tpir_fpir_curve.png'))
    print('  分布图 : %s' % p1)
    print('  曲线图 : %s' % p2)
    print('=' * 78)


if __name__ == '__main__':
    main()
