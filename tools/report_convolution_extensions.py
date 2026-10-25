"""Render saved convolution experiments without rerunning fitting or hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf8'))


def report(root: Path, manifest: Path, output: Path) -> None:
    if output.exists():
        raise ValueError('Refusing overwrite')
    output.mkdir(parents=True)
    endpoints=read(root/'endpoints/summary.json')
    geometry=read(root/'geometry_causal/summary.json')
    dynamics=read(root/'dynamics_frozen/summary.json')
    selected=[r for r in dynamics['records'] if r['variant']=='main' and r['split']=='historical_only']
    if len(selected)!=2 or selected[0]['model_sha256']!=selected[1]['model_sha256']:
        raise ValueError('Expected two primary-bin evaluations with the same frozen historical model')
    times=np.arange(5,300,10)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for ax,r in zip(axes,selected):
        ax.plot(times,r['actual'],'o-',color='black',label='Measured (unaccepted geometry)',markersize=3)
        for name,color in [('hold','#999999'),('trend','#e69f00'),('first_order','#0072b2'),('fir','#009e73')]:
            ax.plot(times,r['predictions'][name],label=name,color=color)
        ax.set_title(r['uid'].split('->')[1]+' | historical-only retrospective forecast')
        ax.set_ylabel('Length / strip px');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('Seconds after post-stage observation start')
    fig.savefig(output/'predictions.png',dpi=160);plt.close(fig)
    rows=read(root/'geometry_causal/rows.json')
    manifest_stages={s['uid']:s for s in read(manifest)['stages']}
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    cases=[('F1',1,120),('F2',1,10),('F2',1,120),('F2',2,120)]
    for ax,(block,stage,second) in zip(axes.ravel(),cases):
        row=next(r for r in rows if (r['block'],r['stage'],r['second'])==(block,stage,second))
        image=cv2.imread(row['source']['image'],0);h,w=image.shape
        ax.imshow(image,cmap='gray',vmin=np.percentile(image,2),vmax=np.percentile(image,98))
        for wall in manifest_stages[f'20261023_{block}_{stage}']['walls']:
            ax.plot([wall['x1']*w,wall['x2']*w],[wall['y1']*h,wall['y2']*h],color='#ff9933',linewidth=1)
        ax.set_title(f'{block} stage {stage}, {second}s: '+row['result']['state'],fontsize=9)
        ax.set_axis_off()
    fig.suptitle('Registered seeds (orange), original frames; contrast enhanced for display only',fontsize=10)
    fig.savefig(output/'geometry_review.png',dpi=160);plt.close(fig)
    lines=['# 卷积三项离线工作结果','',
        '已完成端点候选、视野质量诊断、时间卷积预测的实现、原数据回放与自动化测试。没有连接设备，生产测量与控制流程未修改。',
        '', '## 1. 端点：强梯度方案不替换现有规则','',
        '比较 60 张已有标注图、159 个标注液滴，以及五日期 539 张可分析原图。标注图属于校准集。',
        '', '|方法|匹配数|未匹配候选|长度相对误差中位数|端点误差P90 px|',
        '|---|---:|---:|---:|---:|']
    for name,m in endpoints['metrics'].items():
        lines.append(f"|{name}|{m['matched']}/159|{m['unmatched']}|{100*m['length_relative_median']:.2f}%|{m['endpoint_p90']:.1f}|")
    lines += ['', '强梯度容易落在滴内部或间隙。增加候选支持闸门没有增加误检，但匹配从154降到126，也不采用。此前额外高斯平滑的轻微收益保留为候选，不能与本次梯度替代结果混淆。',
        '', '## 2. 视野：固定参考＋因果短窗口','',
        '每块只用前10秒参考，之后不自动更新；用过去5个1 Hz原始帧做中位图，每5秒评分。横向位置、壁距、倾角变化保留独立阈值，几何来自各块记录。',
        '', '|块|一致候选|壁不确定|其他状态|', '|---|---:|---:|---:|']
    for block,m in geometry['blocks'].items():
        s=m['states'];other=sum(s.values())-s.get('stable_candidate',0)-s.get('wall_unresolved',0)
        lines.append(f"|{block}|{s.get('stable_candidate',0)}|{s.get('wall_unresolved',0)}|{other}|")
    faults=[f for f in geometry['faults'] if f['original_state']=='stable_candidate']
    lines += ['', f"真实图像的 {len(faults)//4} 个原本一致样本，经位移、倾斜、擦壁和平坦图注入，共 {len(faults)} 次，{sum(not f['result']['eligible'] for f in faults)} 次不再允许候选通过。这是配对注入检查，不是现场灵敏度或误报率。",
        '', 'R1/R2的240个位置由现场缺壁事实标为几何未确认，不能计入算法检出成功。F2大量不确定提示表明该参考/特征组合仍不适合直接上线；不据此宣称F2没有液滴。',
        '', 'F2早段与后段亮暗结构明显不同；模板出现约2–4px偏移及能量变化。目前无法仅凭模板失配区分实体壁移动、流态变化和光照改变。',
        '', '离线质量状态支持重复帧、中断、持续3次无效后提示，以及恢复检查。尚未接入真实冻结或停泵生命周期。',
        '', '![几何复核](geometry_review.png)',
        '', '## 3. 时间卷积：存在局部预测收益','',
        '实现因果分段脉冲响应积分，允许按实际事件时间累加多次Q2变化；未来命令对过去预测严格无影响。多事件叠加仅通过合成测试，实际合格实验每块只有一次Q2阶跃。',
        '', '历史数据有3个满足现有筛选条件的孤立Q2正向变化块。开发比较按整块/整日期隔离，正则强度只在训练块内部选择；无足够组时固定默认值。加入同日期同起始状态停启曲线的结果单列，不将扣除视为无偏因果估计。',
        '', '下表只使用10/16一个历史正向块训练，同一个冻结参数包评分10/23 F1/F2；历史训练使用1 Hz口径，本次输出使用原5 Hz/10秒箱。数值为回顾性RMSE（条带px），不授予留出或MPC资格。',
        '', '|块|保持值|事前趋势|一阶模型|时间卷积|相对保持改善|相对趋势改善|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        m=r['score']['metrics'];f=m['fir']['rmse_px'];h=m['hold']['rmse_px'];t=m['trend']['rmse_px']
        lines.append(f"|{r['uid'].split('->')[1]}|{h:.3f}|{t:.3f}|{m['first_order']['rmse_px']:.3f}|{f:.3f}|{100*(1-f/h):.1f}%|{100*(1-f/t):.1f}%|")
    lines += ['', '![预测比较](predictions.png)',
        '', '±1px几何变体均分别保存；更大壁距、倾角和视野变化不能由此消除。只验证了探索性正向预测，没有合格反向证据。停启曲线加入后某些开发评分变差，因此不能把所有残差解释成停启。',
        '', '10/23使用日志中的命令回读相对观测起点时间作为代理，并保留请求时间；不把回读当成芯片真实流动起点。较早记录缺完整细分事件时采用观测零点，并明确记为未知起效时间。未重建记录开始前的未知输入。',
        '', f"冻结参数包 SHA256：`{selected[0]['model_sha256']}`。两个输出的预测哈希分别保存在 dynamics_frozen/summary.json 中，哈希仅证明离线可复现。",
        '', '## 实际推进决定','',
        '- 保留现有外缘测量；不采用强梯度替代或额外支持过滤。',
        '- 壁诊断作为离线复核提示保留，先解释F2参考失配，再考虑实时提示接入。',
        '- 保留时间卷积为局部Q2预测候选，与一阶模型并列；下一轮完整正反实验仍需独立评分。',
        '- 各块几何、尺度及芯片参数继续独立，不把图像参考一致误称实体几何确认。',
        '', '## 文件','',
        '`endpoints/`：校准集与539原图的逐候选坐标；`geometry/`：初版逐帧诊断；`geometry_causal/`：因果窗口、固定参考、逐图哈希、注入与质量状态；`dynamics_frozen/`：42条分口径、分几何、分拆分策略评分及冻结参数。原数据未修改。', '']
    log=root/'full_pytest.log'
    if log.exists():
        results=[line.strip() for line in log.read_text(encoding='utf8',errors='replace').splitlines()
                 if ' passed' in line and ' in ' in line]
        lines += ['## 软件验证','',*results,'','上述测试不连接设备；不证明真实异常准确率或闭环收益。','']
    (output/'完成报告.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();report(args.root,args.manifest,args.output)
