#!/usr/bin/env python3
"""大规模人脸特征相似度评估（分块、多 GPU、低内存）。"""
import argparse, csv, os, pickle, time
import numpy as np
import torch
import multiprocessing as mp

NBINS = 20000

def worker(gpu, start, end, path, block, out_dir, extract_threshold, max_extract):
    with open(path, 'rb') as f:
        feats, _, ids, files = pickle.load(f)
    feats = np.asarray(feats, dtype=np.float32)
    ids = np.asarray(ids)
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
    # Ampere及更新架构上的TF32可显著提升FP32矩阵乘法吞吐；对余弦相似度评估精度影响可忽略。
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    x = torch.from_numpy(feats).to(device)
    n = len(ids); pos_hist = np.zeros(NBINS, np.int64); neg_hist = np.zeros(NBINS, np.int64)
    pos_values = []; extracted = []
    for i0 in range(start, end, block):
        i1 = min(i0 + block, end)
        with torch.inference_mode():
            sims = (x[i0:i1] @ x.T).float().cpu().numpy()
        for local, i in enumerate(range(i0, i1)):
            j0 = i + 1
            if j0 >= n: continue
            vals = sims[local, j0:]
            same = ids[j0:] == ids[i]
            pv, nv = vals[same], vals[~same]
            if pv.size:
                pos_values.append(pv.astype(np.float32, copy=False))
                idx = np.clip(((pv + 1) * (NBINS / 2)).astype(np.int64), 0, NBINS-1)
                pos_hist += np.bincount(idx, minlength=NBINS)
            if nv.size:
                idx = np.clip(((nv + 1) * (NBINS / 2)).astype(np.int64), 0, NBINS-1)
                neg_hist += np.bincount(idx, minlength=NBINS)
                if extract_threshold is not None and len(extracted) < max_extract:
                    hit = np.flatnonzero(nv > extract_threshold)[:max_extract-len(extracted)]
                    neg_indices = np.flatnonzero(~same)[hit] + j0
                    extracted.extend((i, int(j), float(nv[k])) for k, j in zip(hit, neg_indices))
            if extract_threshold is not None and len(extracted) < max_extract and pv.size:
                hit = np.flatnonzero(pv < extract_threshold)[:max_extract-len(extracted)]
                pos_indices = np.flatnonzero(same)[hit] + j0
                extracted.extend((i, int(j), float(pv[k])) for k, j in zip(hit, pos_indices))
    pos = np.concatenate(pos_values) if pos_values else np.empty(0, np.float32)
    np.savez_compressed(os.path.join(out_dir, f'part_{gpu}.npz'), pos_values=pos, pos_hist=pos_hist, neg_hist=neg_hist)
    if extracted:
        with open(os.path.join(out_dir, f'extract_{gpu}.csv'), 'w', newline='') as f:
            w=csv.writer(f); w.writerow(['i','j','similarity','path_i','path_j','type'])
            for i,j,s in extracted:
                w.writerow([i,j,s,files[i],files[j], 'positive' if ids[i]==ids[j] else 'negative'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default='s4_0618_enhance.pkl'); ap.add_argument('--output-dir', default='evaluation_output')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6'); ap.add_argument('--block-size', type=int, default=512)
    ap.add_argument('--extract-threshold', type=float); ap.add_argument('--max-extract', type=int, default=10000)
    args=ap.parse_args(); t0=time.time(); os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input,'rb') as f: feats, _, ids, files = pickle.load(f)
    n=len(ids); uniq,cnt=np.unique(ids,return_counts=True); total=n*(n-1)//2; pos_total=int(np.sum(cnt*(cnt-1)//2)); neg_total=total-pos_total
    print(f'N={n:,}, identities={len(uniq):,}, pairs={total:,}, positive={pos_total:,}, negative={neg_total:,}')
    gpus=[int(x) for x in args.gpus.split(',') if x.strip()]; gpus=gpus or [0]
    ranges=[(n*k//len(gpus), n*(k+1)//len(gpus)) for k in range(len(gpus))]
    ctx=mp.get_context('spawn'); ps=[]
    for gpu,(s,e) in zip(gpus,ranges):
        p=ctx.Process(target=worker,args=(gpu,s,e,args.input,args.block_size,args.output_dir,args.extract_threshold,args.max_extract)); p.start(); ps.append(p)
    for p in ps: p.join()
    if any(p.exitcode for p in ps): raise RuntimeError('worker failed')
    ph=np.zeros(NBINS,np.int64); nh=np.zeros(NBINS,np.int64); pos=[]
    for gpu in gpus:
        z=np.load(os.path.join(args.output_dir,f'part_{gpu}.npz')); ph+=z['pos_hist']; nh+=z['neg_hist']; pos.append(z['pos_values'])
    pos=np.concatenate(pos) if pos else np.empty(0,np.float32); edges=np.linspace(-1,1,NBINS+1); centers=(edges[:-1]+edges[1:])/2
    targets=[1e-5,1e-4,1e-3,1e-2]; result=[]
    neg_cum=np.cumsum(nh[::-1]); pos_cum=np.cumsum(ph[::-1]);
    for fp in targets:
        rank=min(int(np.ceil(fp*neg_total)), neg_total); bi=int(np.searchsorted(neg_cum, max(rank,1))); threshold=centers[::-1][min(bi,NBINS-1)]
        idx=np.clip(((threshold+1)*NBINS/2).astype(int) if hasattr(threshold,'astype') else int((threshold+1)*NBINS/2),0,NBINS-1)
        tp=ph[idx:].sum()/max(pos_total,1); result.append((fp,threshold,tp))
        print(f'TPIR @ FPIR={fp:g}: {tp*100:.3f}% (threshold={threshold:.6f})')
    np.savez(os.path.join(args.output_dir,'histograms.npz'), centers=centers,pos_hist=ph,neg_hist=nh)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.figure(); plt.plot(centers, ph/max(pos_total,1), label='正样本'); plt.plot(centers, nh/max(neg_total,1), label='负样本'); plt.xlabel('余弦相似度'); plt.ylabel('比例'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,'similarity_distribution.png'),dpi=150); plt.close()
        fps=np.logspace(-5,-1,200); ths=[]; tps=[]
        for fp in fps:
            r=max(1,int(np.ceil(fp*neg_total))); bi=int(np.searchsorted(neg_cum,r)); th=centers[::-1][min(bi,NBINS-1)]; ths.append(th); tps.append(ph[int((th+1)*NBINS/2):].sum()/max(pos_total,1))
        plt.figure(); plt.semilogx(fps,tps); plt.xlabel('FPIR'); plt.ylabel('TPIR'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,'tpir_fpir_curve.png'),dpi=150); plt.close()
    except Exception as e: print('绘图失败:',e)
    elapsed=time.time()-t0; open(os.path.join(args.output_dir,'summary.md'),'w').write(f'# 评估总结\n\n- 样本数：{n:,}\n- 正/负样本对：{pos_total:,} / {neg_total:,}\n- GPU：{gpus}\n- 用时：{elapsed:.2f} 秒\n\n| FPIR | 阈值 | TPIR |\n|---:|---:|---:|\n' + ''.join(f'| {f:g} | {th:.6f} | {tp*100:.3f}% |\n' for f,th,tp in result))
    print(f'完成，用时 {elapsed:.2f} 秒；结果见 {args.output_dir}/summary.md')
if __name__=='__main__': main()
