"""Prepare a local, anonymous repeated-image review; never fills human answers."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import shutil


def repeated_order(count: int, seed: int = 20261024) -> list[int]:
    if count < 12:
        raise ValueError('At least 12 unique images required for separated repeats')
    rng = random.Random(seed)
    first = list(range(count))
    rng.shuffle(first)
    for _ in range(10000):
        second = list(range(count))
        rng.shuffle(second)
        if all(count + second.index(i) - first.index(i) >= 10 for i in range(count)):
            return first + second
    raise ValueError('Could not construct separated repeats')


def build(sources: list[dict], output: Path) -> None:
    if output.exists():
        raise ValueError('Refusing to overwrite review')
    if len({s['sha256'] for s in sources}) != len(sources):
        raise ValueError('Distinct original images required')
    order = repeated_order(len(sources))
    for source in sources:
        if hashlib.sha256(Path(source['image']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Source image changed')
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    public, key = [], []
    for number, source_id in enumerate(order, 1):
        name = f'B{number:03d}'
        source = sources[source_id]
        shutil.copyfile(source['image'], output / 'images' / f'{name}.png')
        public.append(dict(id=name, file=f'images/{name}.png', upper_wall_visibility=None,
                           lower_wall_visibility=None, upper_wall_points=None, lower_wall_points=None,
                           objects=[], review_complete=False, note=''))
        key.append(dict(id=name, source_id=source_id, source=source))
    (output / 'key_keep_closed.json').write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding='utf-8')
    package_id = hashlib.sha256(''.join(sources[i]['sha256'] for i in order).encode()).hexdigest()
    blank = dict(schema='blind-boundary-review-v1', package_id=package_id, reviewer=None, coordinate_system='original_sensor_px',
                 human_review_complete=False, images=public)
    (output / 'blank_answers.json').write_text(json.dumps(blank, ensure_ascii=False, indent=2), encoding='utf-8')
    render_html(public, package_id, output)


def render_html(public: list[dict], package_id: str, output: Path) -> None:
    # Inline images keep display-only pixel transforms usable from file://;
    # browsers can otherwise taint a canvas drawn from another local file.
    images = {r['id']: 'data:image/png;base64,' + base64.b64encode((output / r['file']).read_bytes()).decode('ascii')
              for r in public}
    html = HTML.replace('__PUBLIC__', json.dumps(public, ensure_ascii=False)).replace('__PACKAGE_ID__', package_id)
    html = html.replace('__IMAGES__', json.dumps(images))
    (output / 'review.html').write_text(html, encoding='utf-8')


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>液滴与管壁盲复核</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#172535}main{max-width:1120px;margin:auto}button,select,input{font:inherit;padding:7px;margin:4px}canvas{width:720px;max-width:100%;background:#222;cursor:crosshair}aside{max-width:350px}section{display:flex;gap:16px;flex-wrap:wrap}.card{background:white;padding:16px;border-radius:10px}p{line-height:1.55}.warn{color:#92400e}textarea{width:95%;height:60px}</style>
<main><h1>液滴与管壁盲复核</h1><p>先不打开对应表。同一人两次标注仅评价重复性；有第二位人员时分别导出答案。按图中实际可见内容标注，不用亮边猜不可见下壁。</p>
<p class="warn">所有坐标均为原始传感器像素，不等同扶正后的条带像素。显示增亮不修改原图。此页不计算MPC或测量通过结论。</p>
<div class="card"><label>复核人 <input id="reviewer" placeholder="填写姓名或代号"></label><button id="export">导出答案 JSON</button><label>导入已保存答案 <input id="import" type="file" accept=".json"></label><p>本页不自动永久保存；关闭前请导出答案。导出的未复核项会保留空白。</p></div>
<p><button id="prev">上一张</button><strong id="position"></strong><button id="next">下一张</button><label>显示 <select id="display"><option value="raw">原始灰度</option><option value="bright">统一增亮（4×灰度−280）</option></select></label></p>
<section><div class="card"><canvas id="canvas" width="720" height="540"></canvas><p id="status"></p></div>
<aside class="card"><label>上壁 <select id="upper"><option value="">未复核</option><option value="visible">可见</option><option value="invisible">不可见</option><option value="uncertain">不确定</option></select></label><br>
<label>下壁 <select id="lower"><option value="">未复核</option><option value="visible">可见</option><option value="invisible">不可见</option><option value="uncertain">不确定</option></select></label>
<p>选择工具后点击两个端点。管壁选同一实体边界上两个相距较远的点；对象按从左到右顺序逐个标注可判断的候选。</p>
<select id="mode"><option value="object">对象左右端点</option><option value="upper_wall_points">上壁两点</option><option value="lower_wall_points">下壁两点</option></select>
<select id="kind"><option value="">先选择对象类别</option><option value="complete_droplet">完整液滴</option><option value="gap">间隙</option><option value="filament">液丝</option><option value="bubble">气泡</option><option value="truncated">截断对象</option><option value="uncertain">不确定</option></select>
<button id="cancel">取消当前点</button><button id="undo">删除最后对象</button><button id="clearwalls">清空管壁点</button>
<p id="count"></p><label><input id="done" type="checkbox">本图复核完成（不代表测量通过）</label><p>无可判定对象时可保持对象列表为空，并在备注说明；空列表不等于确认无液滴。</p><textarea id="note" placeholder="备注：不确定原因、可见性等"></textarea></aside></section></main>
<script>
const package_id='__PACKAGE_ID__',initial=__PUBLIC__,image_data=__IMAGES__;let rows=structuredClone(initial),index=0,pending=[],img=new Image();
const $=x=>document.getElementById(x), c=$('canvas'),ctx=c.getContext('2d');
function draw(){ctx.clearRect(0,0,720,540);if(!img.complete||!img.naturalWidth)return;ctx.drawImage(img,0,0);if($('display').value==='bright'){let p=ctx.getImageData(0,0,720,540);for(let k=0;k<p.data.length;k+=4){for(let j=0;j<3;j++)p.data[k+j]=Math.max(0,Math.min(255,p.data[k+j]*4-280));}ctx.putImageData(p,0,0);}let r=rows[index];function line(points,color){if(!points)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,k)=>{if(k)ctx.lineTo(...p);else ctx.moveTo(...p);});ctx.stroke();points.forEach(p=>{ctx.beginPath();ctx.arc(...p,3,0,2*Math.PI);ctx.fill();});}line(r.upper_wall_points,'#00ff88');line(r.lower_wall_points,'#00bbff');r.objects.forEach(o=>line(o.points,'#ffcf33'));line(pending,'#ff6688');$('count').textContent='已标对象：'+r.objects.length;$('status').textContent=pending.length?'已选第一点，请选第二点。':'原图 720×540；点击坐标按原图保存。';}
function show(){pending=[];let r=rows[index];$('position').textContent=r.id+' · '+(index+1)+' / '+rows.length;$('upper').value=r.upper_wall_visibility||'';$('lower').value=r.lower_wall_visibility||'';$('done').checked=r.review_complete;$('note').value=r.note;img.onload=draw;img.src=image_data[r.id];$('prev').disabled=index===0;$('next').disabled=index===rows.length-1;}
$('prev').onclick=()=>{if(index>0){index--;show();}};$('next').onclick=()=>{if(index<rows.length-1){index++;show();}};$('display').onchange=draw;$('mode').onchange=()=>{pending=[];draw();};$('kind').onchange=()=>{pending=[];draw();};
for(const side of ['upper','lower'])$(side).onchange=()=>{rows[index][side+'_wall_visibility']=$(side).value||null;if($(side).value!=='visible')rows[index][side+'_wall_points']=null;rows[index].review_complete=false;$('done').checked=false;pending=[];draw();};
c.onclick=e=>{const mode=$('mode').value,r=rows[index];if(mode==='object'&&!$('kind').value){alert('请先选择类别。');return;}if(mode!=='object'&&r[mode.replace('_points','_visibility')]!=='visible'){alert('只有实际可见的管壁才能标点。');return;}const b=c.getBoundingClientRect();pending.push([Math.max(0,Math.min(719,(e.clientX-b.left)*720/b.width)),Math.max(0,Math.min(539,(e.clientY-b.top)*540/b.height))]);if(pending.length===2){if(mode==='object')r.objects.push({kind:$('kind').value,points:pending});else r[mode]=pending;pending=[];r.review_complete=false;$('done').checked=false;}draw();};
$('cancel').onclick=()=>{pending=[];draw();};$('undo').onclick=()=>{rows[index].objects.pop();rows[index].review_complete=false;$('done').checked=false;draw();};$('clearwalls').onclick=()=>{rows[index].upper_wall_points=null;rows[index].lower_wall_points=null;rows[index].review_complete=false;$('done').checked=false;draw();};$('done').onchange=()=>rows[index].review_complete=$('done').checked;$('note').oninput=()=>rows[index].note=$('note').value;
$('export').onclick=()=>{let data={schema:'blind-boundary-review-v1',package_id,reviewer:$('reviewer').value||null,coordinate_system:'original_sensor_px',human_review_complete:rows.every(r=>r.review_complete),images:rows};const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='boundary_answers.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
$('import').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());if(d.schema!=='blind-boundary-review-v1'||d.package_id!==package_id||d.coordinate_system!=='original_sensor_px'||!Array.isArray(d.images)||d.images.length!==initial.length||d.images.some((r,k)=>r.id!==initial[k].id||!Array.isArray(r.objects)))throw Error('答案结构或样本编号不匹配');rows=d.images.map((r,k)=>({...r,file:initial[k].file}));$('reviewer').value=d.reviewer||'';index=0;show();}catch(x){alert('导入失败：'+x.message);}};show();
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.sources.read_text(encoding='utf-8')), args.output)
