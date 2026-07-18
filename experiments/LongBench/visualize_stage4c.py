#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

TASKS=['qasper','hotpotqa','passage_retrieval_en']
NAMES={'qasper':'Qasper','hotpotqa':'HotpotQA','passage_retrieval_en':'PassageRetrieval-en'}


def args():
    p=argparse.ArgumentParser(description='Visualize EntroKV layer-head budgets')
    p.add_argument('--input-root',default='outputs/longbench_stage4b')
    p.add_argument('--run-name',default='entrokv_r0.30_a0.50')
    p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--expected-samples',type=int,default=None)
    p.add_argument('--dpi',type=int,default=300)
    return p.parse_args()


def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                if r.get('status')=='ok': rows.append(r)
    rows.sort(key=lambda x:int(x['sample_index']))
    if not rows: raise RuntimeError(f'No successful records: {path}')
    if len({int(r['sample_index']) for r in rows})!=len(rows):
        raise AssertionError(f'Duplicate sample indices: {path}')
    return rows


def corr(x,y):
    x=np.asarray(x,float).ravel(); y=np.asarray(y,float).ravel()
    if x.size<2 or x.std()==0 or y.std()==0:return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def load_task(path,task,expected):
    rows=read_jsonl(path)
    if expected is not None and len(rows)!=expected:
        raise AssertionError(f'{task}: expected {expected}, got {len(rows)}')
    bs=[]; es=[]; ms=[]; uniforms=[]; truncated=[]; input_tokens=[]
    shape=None
    for r in rows:
        b=np.asarray(r.get('history_budgets'),float)
        e=np.asarray(r.get('entropy'),float)
        if b.ndim!=2 or e.shape!=b.shape:
            raise AssertionError(f'{task}/{r["sample_index"]}: invalid budget/entropy shape')
        shape=shape or b.shape
        if b.shape!=shape: raise AssertionError(f'{task}: inconsistent shapes')
        u=float(r['target_history_capacity_per_head'])
        if u<=0: raise AssertionError(f'{task}: invalid uniform history budget')
        expected_total=b.size*u
        if not math.isclose(float(b.sum()),expected_total,abs_tol=1e-6*b.size):
            raise AssertionError(f'{task}/{r["sample_index"]}: global budget not conserved')
        m=b/u
        if not math.isclose(float(m.mean()),1.0,abs_tol=1e-6):
            raise AssertionError(f'{task}/{r["sample_index"]}: multiplier mean != 1')
        bs.append(b); es.append(e); ms.append(m); uniforms.append(u)
        truncated.append(bool(r.get('truncated'))); input_tokens.append(int(r['input_tokens']))
    return {
        'task':task,'n':len(rows),'budget':np.stack(bs),'entropy':np.stack(es),
        'mult':np.stack(ms),'uniform':np.asarray(uniforms),'truncated':np.asarray(truncated),
        'input_tokens':np.asarray(input_tokens)
    }


def save(fig,path,dpi):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=dpi,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def axes(ax,layers,heads):
    ax.set_xlabel('KV head'); ax.set_ylabel('Layer')
    ax.set_xticks(range(heads)); ax.set_xticklabels(range(heads))
    ticks=np.arange(0,layers,2 if layers<=40 else max(1,layers//16))
    ax.set_yticks(ticks); ax.set_yticklabels(ticks)


def span(mats,center):
    d=np.concatenate([np.abs(x.ravel()-center) for x in mats])
    return max(float(np.quantile(d,.99)),float(d.max())*.7,.05)


def heatmaps(data,out,dpi):
    means=[d['mult'].mean(0) for d in data]
    s=span(means,1.0); norm=TwoSlopeNorm(vmin=1-s,vcenter=1,vmax=1+s)
    fig,axs=plt.subplots(1,len(data),figsize=(4.4*len(data),9),sharey=True,constrained_layout=True)
    axs=np.atleast_1d(axs)
    for ax,d,m in zip(axs,data,means):
        im=ax.imshow(m,aspect='auto',origin='upper',cmap='RdBu_r',norm=norm,interpolation='nearest')
        ax.set_title(f'{NAMES.get(d["task"],d["task"])}\nN={d["n"]}')
        axes(ax,*m.shape)
    cb=fig.colorbar(im,ax=list(axs),shrink=.78,pad=.02)
    cb.set_label('Mean allocation multiplier vs. uniform SnapKV')
    fig.suptitle('EntroKV dynamic history-budget allocation')
    save(fig,out/'01_task_budget_multiplier_heatmaps',dpi)

    for d,m in zip(data,means):
        fig,ax=plt.subplots(figsize=(6,9),constrained_layout=True)
        im=ax.imshow(m,aspect='auto',origin='upper',cmap='RdBu_r',norm=norm,interpolation='nearest')
        axes(ax,*m.shape); ax.set_title(f'{NAMES.get(d["task"],d["task"])} mean allocation multiplier')
        fig.colorbar(im,ax=ax,shrink=.78,pad=.03).set_label('Multiplier vs. uniform history budget')
        save(fig,out/'individual'/f'{d["task"]}_budget_multiplier_heatmap',dpi)


def differences(data,out,dpi):
    pairs=list(combinations(data,2)); mats=[]
    for a,b in pairs:mats.append(100*(a['mult'].mean(0)-b['mult'].mean(0)))
    s=span(mats,0); norm=TwoSlopeNorm(vmin=-s,vcenter=0,vmax=s)
    fig,axs=plt.subplots(1,len(pairs),figsize=(4.7*len(pairs),9),sharey=True,constrained_layout=True)
    axs=np.atleast_1d(axs)
    for ax,(a,b),m in zip(axs,pairs,mats):
        im=ax.imshow(m,aspect='auto',origin='upper',cmap='RdBu_r',norm=norm,interpolation='nearest')
        ax.set_title(f'{NAMES.get(a["task"],a["task"])} − {NAMES.get(b["task"],b["task"])}')
        axes(ax,*m.shape)
    fig.colorbar(im,ax=list(axs),shrink=.78,pad=.02).set_label('Multiplier difference (percentage points)')
    fig.suptitle('Task-wise differences in EntroKV allocation')
    save(fig,out/'02_pairwise_task_difference_heatmaps',dpi)


def profiles(data,out,dpi):
    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    for d in data:ax.plot(d['mult'].mean((0,2)),marker='o',ms=3,label=NAMES.get(d['task'],d['task']))
    ax.axhline(1,ls='--',lw=1,label='Uniform SnapKV'); ax.set(xlabel='Layer',ylabel='Mean allocation multiplier',title='Layer-wise EntroKV budget profile')
    ax.grid(axis='y',alpha=.25); ax.legend(ncol=2); save(fig,out/'03_layerwise_budget_profile',dpi)

    fig,ax=plt.subplots(figsize=(8.2,4.8),constrained_layout=True); x=np.arange(data[0]['mult'].shape[2]); w=.8/len(data)
    for i,d in enumerate(data):ax.bar(x+(i-(len(data)-1)/2)*w,d['mult'].mean((0,1)),width=w,label=NAMES.get(d['task'],d['task']))
    ax.axhline(1,ls='--',lw=1,label='Uniform SnapKV'); ax.set(xlabel='KV head',ylabel='Mean allocation multiplier',title='Head-wise EntroKV budget profile')
    ax.set_xticks(x); ax.grid(axis='y',alpha=.25); ax.legend(ncol=2); save(fig,out/'04_headwise_budget_profile',dpi)

    fig,ax=plt.subplots(figsize=(7.4,5.4),constrained_layout=True)
    for d in data:
        x=d['entropy'].mean(0).ravel(); y=d['mult'].mean(0).ravel(); ax.scatter(x,y,s=18,alpha=.55,label=NAMES.get(d['task'],d['task']))
        if x.std()>0:
            p=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),100); ax.plot(xx,p[0]*xx+p[1],lw=1.5)
    ax.axhline(1,ls='--',lw=1); ax.set(xlabel='Mean normalized attention entropy',ylabel='Mean allocation multiplier',title='Entropy signal and allocated history budget')
    ax.grid(alpha=.25); ax.legend(); save(fig,out/'05_entropy_budget_relationship',dpi)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def tables(data,out):
    task_rows=[]; cell_rows=[]; rank_rows=[]; sim_rows=[]; pair_rows=[]
    for d in data:
        m=d['mult']; mm=m.mean(0); mb=d['budget'].mean(0); me=d['entropy'].mean(0); sm=m.std(0)
        sample_corr=[corr(d['entropy'][i],m[i]) for i in range(d['n'])]
        finite=[x for x in sample_corr if np.isfinite(x)]
        task_rows.append({'task':d['task'],'samples':d['n'],'layers':m.shape[1],'heads':m.shape[2],
          'mean_input_tokens':d['input_tokens'].mean(),'truncation_rate':d['truncated'].mean(),
          'mean_uniform_history_budget':d['uniform'].mean(),'mean_abs_multiplier_deviation':np.abs(m-1).mean(),
          'mean_sample_multiplier_std':m.reshape(d['n'],-1).std(1).mean(),'min_mean_multiplier':mm.min(),'max_mean_multiplier':mm.max(),
          'mean_sample_entropy_budget_corr':np.mean(finite) if finite else float('nan')})
        for l in range(mm.shape[0]):
            for h in range(mm.shape[1]):cell_rows.append({'task':d['task'],'layer':l,'kv_head':h,'mean_history_budget_tokens':mb[l,h],
              'mean_allocation_multiplier':mm[l,h],'std_allocation_multiplier':sm[l,h],'mean_entropy':me[l,h]})
        order=np.argsort(mm.ravel())
        for direction,ids in [('lowest',order[:12]),('highest',order[-12:][::-1])]:
            for rank,idx in enumerate(ids,1):
                l,h=np.unravel_index(idx,mm.shape); rank_rows.append({'task':d['task'],'direction':direction,'rank':rank,'layer':l,'kv_head':h,
                  'mean_allocation_multiplier':mm[l,h],'mean_history_budget_tokens':mb[l,h],'std_allocation_multiplier':sm[l,h],'mean_entropy':me[l,h]})
    for a in data:
        for b in data:sim_rows.append({'task_a':a['task'],'task_b':b['task'],'pearson_correlation':corr(a['mult'].mean(0),b['mult'].mean(0))})
    for a,b in combinations(data,2):
        diff=a['mult'].mean(0)-b['mult'].mean(0); pair_rows.append({'task_a':a['task'],'task_b':b['task'],
          'mean_absolute_multiplier_difference':np.abs(diff).mean(),'max_absolute_multiplier_difference':np.abs(diff).max(),
          'pearson_correlation':corr(a['mult'].mean(0),b['mult'].mean(0))})
    write_csv(out/'tables'/'task_summary.csv',task_rows); write_csv(out/'tables'/'layer_head_summary.csv',cell_rows)
    write_csv(out/'tables'/'top_bottom_layer_heads.csv',rank_rows); write_csv(out/'tables'/'task_allocation_similarity.csv',sim_rows)
    write_csv(out/'tables'/'pairwise_task_difference_summary.csv',pair_rows)
    return {'task_summary':task_rows,'top_bottom_layer_heads':rank_rows,'pairwise_task_differences':pair_rows}


def analysis(data,summary,out,input_root,run_name):
    lines=['# Stage 4C：EntroKV 动态预算可视化分析','',f'- 输入：`{input_root}`',f'- Run：`{run_name}`','',
      '主图使用 $M_{s,l,h}=B_{s,l,h}/U_s$。其中 $B$ 是 EntroKV history budget，$U_s$ 是同一样本固定 SnapKV 的均匀 history budget。',
      '因此 $M>1$ 表示获得更多预算，$M<1$ 表示更激进压缩；先归一化再跨样本平均可以消除输入长度差异。','','## 任务级统计','',
      '| Task | N | Truncation | Mean |M−1| | Min mean M | Max mean M | Entropy-budget r |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['task_summary']:lines.append(f"| {NAMES.get(r['task'],r['task'])} | {r['samples']} | {100*r['truncation_rate']:.1f}% | {r['mean_abs_multiplier_deviation']:.3f} | {r['min_mean_multiplier']:.3f} | {r['max_mean_multiplier']:.3f} | {r['mean_sample_entropy_budget_corr']:.3f} |")
    lines+=['','## Top/bottom layer-head','']
    for d in data:
        rows=[x for x in summary['top_bottom_layer_heads'] if x['task']==d['task']]
        hi=[x for x in rows if x['direction']=='highest'][:5]; lo=[x for x in rows if x['direction']=='lowest'][:5]
        lines+=['### '+NAMES.get(d['task'],d['task']),'','- 高预算：'+'；'.join(f"L{x['layer']}-H{x['kv_head']} ({x['mean_allocation_multiplier']:.2f}×)" for x in hi),
          '- 低预算：'+'；'.join(f"L{x['layer']}-H{x['kv_head']} ({x['mean_allocation_multiplier']:.2f}×)" for x in lo),'']
    lines+=['## 报告解释原则','','1. 先说明热力图是相对固定 SnapKV 的预算倍率，不是绝对 token 数。','2. 用共享色标比较任务，并用差值图讨论 task-wise heterogeneity。',
      '3. 结合 layer/head profile 判断差异来自层还是 head。','4. 熵与预算相关是算法预期，但相关性不等于语义重要性。','5. 不应仅凭单个高亮格声称某个 head 具有固定语义功能。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    a=args(); root=Path(a.input_root).resolve(); run=root/a.run_name
    out=Path(a.output_dir).resolve() if a.output_dir else root/'visualization'/a.run_name
    out.mkdir(parents=True,exist_ok=True)
    data=[load_task(run/f'{t}.jsonl',t,a.expected_samples) for t in a.tasks]
    if len({d['mult'].shape[1:] for d in data})!=1:raise AssertionError('Tasks have different layer-head shapes')
    heatmaps(data,out,a.dpi); differences(data,out,a.dpi); profiles(data,out,a.dpi); summary=tables(data,out)
    summary.update({'schema_version':1,'input_root':str(root),'run_name':a.run_name,'tasks':a.tasks,
      'normalization':'history_budgets / target_history_capacity_per_head'})
    (out/'visualization_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    analysis(data,summary,out,root,a.run_name)
    print('\nStage 4C dynamic-budget visualization')
    print('='*72)
    for r in summary['task_summary']:print(f"{r['task']:24s} n={r['samples']:3d} mean|M-1|={r['mean_abs_multiplier_deviation']:.4f} entropy-budget-r={r['mean_sample_entropy_budget_corr']:.4f}")
    print('[PASS] Per-sample global budget conservation verified.')
    print(f'[PASS] Figures and tables written to {out}')

if __name__=='__main__':main()
