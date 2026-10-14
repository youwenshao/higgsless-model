"""Summarise a ComfyUI UI-format template: nodes, widget values, links, models (incl. subgraphs).

python scripts/dev/template_summary.py <template.json> [--types SAM3_VideoTrack,KSampler]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(nodes: list[dict], links: list, prefix: str, only: set[str] | None) -> None:
    by_id = {n["id"]: n for n in nodes}
    link_src = {}
    for link in links or []:
        if isinstance(link, list):
            lid, src, slot = link[0], link[1], link[2]
        else:
            lid, src, slot = link["id"], link["origin_id"], link["origin_slot"]
        link_src[lid] = (src, slot)
    for n in nodes:
        ntype = n.get("type")
        if only and ntype not in only or ntype in ("MarkdownNote", "Note"):
            continue
        title = n.get("title") or ""
        mode = n.get("mode", 0)
        print(f"{prefix}[{n['id']}] {ntype} {('«' + title + '»') if title else ''}{' (bypassed)' if mode == 4 else ''}")
        ins = []
        for inp in n.get("inputs") or []:
            if inp.get("link") is not None and inp["link"] in link_src:
                src, slot = link_src[inp["link"]]
                ins.append(f"{inp['name']}<-{by_id.get(src, {}).get('type', src)}#{src}.{slot}")
        if ins:
            print(f"{prefix}    links: {', '.join(ins)}")
        if n.get("widgets_values") not in (None, [], {}):
            print(f"{prefix}    widgets: {json.dumps(n['widgets_values'], ensure_ascii=False)[:400]}")
        models = (n.get("properties") or {}).get("models")
        if models:
            for m in models:
                print(f"{prefix}    model: {m.get('name')} [{m.get('directory')}] {m.get('url', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--types")
    args = parser.parse_args()
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    only = set(args.types.split(",")) if args.types else None
    summarise(data.get("nodes", []), data.get("links", []), "", only)
    for sg in (data.get("definitions") or {}).get("subgraphs", []):
        print(f"== subgraph {sg.get('name')} ({sg.get('id')})")
        summarise(sg.get("nodes", []), sg.get("links", []), "  ", only)


if __name__ == "__main__":
    main()
