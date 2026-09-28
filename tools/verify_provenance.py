"""离线核验运行追溯：源码快照/配置指纹、帧引用的四级校验、测量行→源帧。

只读写文件，**导入不会连接任何设备**（不 import 相机/泵适配器）。

**核验级别是显式声明的，总结果按请求级别判定**（各级结果另存，互不代替）：

| 请求 | 覆盖的级别 | 总结果要求 |
|---|---|---|
| 默认 | 引用结构 | 结构合法 |
| `--check-media` | + 媒体存在 + 解码帧索引在范围内 | 以上全部 |
| `--row N --decode` | + 真正解码该帧 + **内容哈希比对** | 以上全部，**且内容必须匹配** |

`--decode` 下「解码成功但内容不匹配」或「未记录预期哈希」都判失败——不许把
「帧可解码」写成「内容匹配」。

用法：
    # 快照 + 生效配置（含内容指纹比对）
    .venv\\Scripts\\python.exe tools\\verify_provenance.py --directory <prov_dir>

    # 帧引用：结构校验；加 --check-media 再核验媒体存在与索引范围
    .venv\\Scripts\\python.exe tools\\verify_provenance.py --rows <rows.ndjson> --session-root <dir> --check-media

    # 从测量行实际解码找回原帧并比对内容哈希
    .venv\\Scripts\\python.exe tools\\verify_provenance.py --rows <rows.ndjson> --row 12 --session-root <dir> --decode

    # 静帧核验：必须给出记录值才能判定匹配
    .venv\\Scripts\\python.exe tools\\verify_provenance.py --still <png> --still-sha256 <hex>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.provenance import (                                      # noqa: E402
    resolve_source_frame,
    verify_effective_config,
    verify_frame_references,
    verify_source_snapshot,
    verify_still,
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        help="含 source_manifest.json 与 effective_config.json 的目录")
    parser.add_argument("--rows", type=Path, help="逐帧/测量行 ndjson")
    parser.add_argument("--row", type=int, help="要回溯的测量行序号（0 起）")
    parser.add_argument("--session-root", type=Path,
                        help="引用相对路径的基准目录；缺省时用引用里记录的 session_root")
    parser.add_argument("--check-media", action="store_true",
                        help="核验媒体存在与解码帧索引范围（较慢）")
    parser.add_argument("--decode", action="store_true",
                        help="实际解码该帧并做内容哈希比对（只有它能让 matched 为真）")
    parser.add_argument("--still", type=Path, help="静帧路径")
    parser.add_argument("--still-sha256", help="该静帧的记录哈希；缺省即无法判定匹配")
    args = parser.parse_args()
    if args.directory is None and args.rows is None and args.still is None:
        parser.error("至少给出 --directory、--rows 或 --still")
    if args.decode and args.row is None:
        parser.error("--decode 需要同时给出 --row 才能找回具体帧")

    # 声明本次请求到哪一级。各级结果分开保留，**总结果按请求级别判定**——
    # 否则底层发现了内容不匹配、顶层仍报成功，自动验收会误收。
    requested_level = ("decode_and_content" if args.decode
                       else ("media_exists" if args.check_media else "structure"))

    report: dict = {"device_connections": 0, "note": "离线核验，未连接任何设备",
                    "requested_level": requested_level}
    ok = True
    if args.directory is not None:
        report["source_snapshot"] = verify_source_snapshot(args.directory)
        report["effective_config"] = verify_effective_config(args.directory)
        for key in ("source_snapshot", "effective_config"):
            if report[key].get("ok") is False:
                ok = False
    if args.still is not None:
        report["still"] = verify_still(args.still, args.still_sha256)
        if report["still"].get("ok") is False:
            ok = False
    if args.rows is not None:
        rows = load_rows(args.rows)
        report["frame_references"] = verify_frame_references(
            rows, session_root=args.session_root, check_media=args.check_media or args.decode)
        if report["frame_references"].get("ok") is False:
            ok = False
        if args.row is not None:
            if not 0 <= args.row < len(rows):
                parser.error(f"--row 越界：0..{len(rows) - 1}")
            outcome = resolve_source_frame(rows[args.row].get("source_frame_ref") or {},
                                          session_root=args.session_root,
                                          decode=args.decode)
            report["resolved_source_frame"] = outcome
            # 只把**检查过的**级别列出；某一级失败时它仍在列表里，但同级的 ok 为 False，
            # 总 ok 也随之为 False。字段名刻意不叫 achieved，以免读成「这些级别都通过了」。
            report["levels_checked"] = [level.get("level") for level in outcome.get("levels") or []]
            report["levels_passed"] = [level.get("level") for level in outcome.get("levels") or []
                                       if level.get("ok") is True]
            if args.decode:
                # --decode 声明的级别包含内容比对：解码失败、缺少预期哈希或哈希不匹配
                # 都必须让总结果为失败。
                if not outcome.get("resolved") or outcome.get("matched") is not True:
                    ok = False
            elif not outcome.get("resolved"):
                ok = False

    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
