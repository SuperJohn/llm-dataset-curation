"""Eval report: HTML (self-contained, charts inlined) + Markdown (renders on GitHub)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from jinja2 import Environment  # noqa: E402

from curate.pipeline import RunResult  # noqa: E402

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_funnel(run: RunResult) -> bytes:
    stages = [f["stage"] for f in run.funnel[1:]]
    removed = [f["removed"] for f in run.funnel[1:]]
    remaining = [f["rows_out"] for f in run.funnel]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    bars = ax1.barh(stages, removed, color=PALETTE[2])
    ax1.invert_yaxis()
    ax1.bar_label(bars, padding=3, fontsize=8)
    ax1.set_xlabel("records removed")
    ax1.set_title("Removed per stage")
    ax2.plot([f["stage"] for f in run.funnel], remaining, marker="o", color=PALETTE[0])
    ax2.set_ylim(0, max(remaining) * 1.08)
    ax2.set_title(f"Records remaining ({remaining[0]:,} → {remaining[-1]:,})")
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_lengths(run: RunResult) -> bytes:
    before = run.input_df["text"].str.split(" ").list.len().to_numpy()
    after = run.curated["text"].str.split(" ").list.len().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    hi = max(int(max(before.max(initial=1), after.max(initial=1))), 10)
    bins = [0, *[int(1.25**k) for k in range(1, 60) if 1.25**k < hi * 1.25]]
    bins = sorted(set(bins))
    ax.hist(before, bins=bins, alpha=0.55, label=f"before (n={len(before)})", color=PALETTE[3])
    ax.hist(after, bins=bins, alpha=0.55, label=f"after (n={len(after)})", color=PALETTE[1])
    ax.set_xscale("symlog")
    ax.set_xlabel("words per record (log)")
    ax.set_ylabel("records")
    ax.set_title("Length distribution before vs after curation")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def chart_clusters(run: RunResult) -> bytes | None:
    if run.clusters is None or run.clusters.is_empty():
        return None
    sizes = run.clusters.unique("cluster_id").group_by("cluster_size").len().sort("cluster_size")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = [str(v) for v in sizes["cluster_size"].to_list()]
    ax.bar(range(len(labels)), sizes["len"].to_list(), color=PALETTE[4], tick_label=labels)
    ax.set_xlabel("cluster size (records)")
    ax.set_ylabel("clusters")
    ax.set_title("Near-duplicate cluster sizes")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def chart_pii(stats: dict) -> bytes | None:
    hits = stats.get("hits_by_type") or {}
    if not hits:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    keys = sorted(hits, key=hits.get, reverse=True)
    ax.barh(keys, [hits[k] for k in keys], color=PALETTE[5])
    ax.invert_yaxis()
    ax.set_xlabel("redactions")
    ax.set_title("PII redactions by type")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def _length_summary(run: RunResult) -> list[dict]:
    con = duckdb.connect()
    con.register("before_df", run.input_df.select("text").to_arrow())
    con.register("after_df", run.curated.select("text").to_arrow())
    q = """
        select '{label}' as dataset, count(*) as records,
               round(avg(len(string_split(text, ' '))), 1) as mean_words,
               round(quantile_cont(len(string_split(text, ' ')), 0.5), 1) as p50_words,
               round(quantile_cont(len(string_split(text, ' ')), 0.95), 1) as p95_words,
               max(len(string_split(text, ' '))) as max_words
        from {table}
    """
    rows = con.execute(
        q.format(label="before", table="before_df")
        + " union all "
        + q.format(label="after", table="after_df")
    ).pl()
    return rows.to_dicts()


def _removed_examples(run: RunResult, k: int) -> list[dict]:
    if run.removed.is_empty():
        return []
    out = []
    for stage in run.removed["_stage"].unique(maintain_order=True).to_list():
        sub = run.removed.filter(pl.col("_stage") == stage).head(k)
        out.append(
            {
                "stage": stage,
                "items": [
                    {"id": r["id"], "reason": r["_reason"], "text": r["text"][:400]}
                    for r in sub.select("id", "_reason", "text").to_dicts()
                ],
            }
        )
    return out


def _reason_table(run: RunResult) -> list[dict]:
    rows = []
    for r in run.stages:
        if r.name == "quality":
            rows += [
                {"stage": "quality", "rule": k, "removed": v} for k, v in r.stats["by_rule"].items()
            ]
        elif r.removed.height:
            rows.append({"stage": r.name, "rule": "—", "removed": r.removed.height})
    return rows


def build_context(run: RunResult) -> dict:
    stage_stats = {r.name: r.stats for r in run.stages}
    pii = stage_stats.get("pii", {})
    clusters = []
    if run.clusters is not None and not run.clusters.is_empty():
        top_ids = run.clusters.unique("cluster_id", maintain_order=True).head(5)["cluster_id"]
        for cid in top_ids:
            members = run.clusters.filter(pl.col("cluster_id") == cid).head(4).to_dicts()
            clusters.append({"id": cid, "members": members})
    return {
        "name": run.config.name,
        "n_in": run.input_df.height,
        "n_out": run.curated.height,
        "retention": run.curated.height / max(run.input_df.height, 1),
        "funnel": run.funnel,
        "reasons": _reason_table(run),
        "lengths": _length_summary(run),
        "pii": pii,
        "stage_stats": stage_stats,
        "clusters": clusters,
        "examples": _removed_examples(run, run.config.output.max_removed_examples),
    }


TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"


def _md_cell(s: object) -> str:
    s = str(s).replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")
    return s.replace("\n", " ⏎ ")


def render_markdown(c: dict, assets: dict[str, str]) -> str:
    lines = [
        f"# Curation report — {c['name']}",
        "",
        f"**{c['n_in']:,}** records in → **{c['n_out']:,}** kept "
        f"(**{c['retention'] * 100:.1f}%** retention). "
        f"PII redacted in {c['pii'].get('docs_with_pii', 0)} records.",
        "",
        "## Funnel",
        "",
        f"![funnel]({assets['funnel']})",
        "",
        "| stage | rows in | removed | rows out |",
        "|---|---:|---:|---:|",
        *[
            f"| {f['stage']} | {f['rows_in']} | {f['removed']} | {f['rows_out']} |"
            for f in c["funnel"]
        ],
        "",
        "### Removals by rule",
        "",
        "| stage | rule | removed |",
        "|---|---|---:|",
        *[f"| {r['stage']} | {r['rule']} | {r['removed']} |" for r in c["reasons"]],
        "",
        "## Length distribution",
        "",
        f"![lengths]({assets['lengths']})",
        "",
        "| " + " | ".join(c["lengths"][0].keys()) + " |",
        "|" + "---|" * len(c["lengths"][0]),
        *["| " + " | ".join(str(v) for v in r.values()) + " |" for r in c["lengths"]],
        "",
    ]
    if "clusters" in assets:
        lines += ["## Near-duplicate clusters", "", f"![clusters]({assets['clusters']})", ""]
        for cl in c["clusters"][:3]:
            lines += [
                f"**Cluster `{cl['id']}`**",
                "",
                "| id | rep | Ĵ | text |",
                "|---|---|---|---|",
            ]
            for m in cl["members"]:
                j = m["est_jaccard_to_rep"]
                lines.append(
                    f"| {m['id']} | {'✓' if m['is_representative'] else ''} | "
                    f"{'' if j is None else f'{j:.2f}'} | {_md_cell(m['text_preview'][:160])} |"
                )
            lines.append("")
    if "pii" in assets:
        lines += [
            "## PII",
            "",
            f"![pii]({assets['pii']})",
            "",
            f"Hit rate {(c['pii'].get('hit_rate') or 0) * 100:.2f}% of records; "
            f"action = `{c['pii'].get('action')}`.",
            "",
        ]
    lines += ["## Samples of removed records", ""]
    for ex in c["examples"]:
        lines += [f"### {ex['stage']}", "", "| id | reason | text (truncated) |", "|---|---|---|"]
        lines += [
            f"| {it['id']} | {_md_cell(it['reason'])} | {_md_cell(it['text'][:200])} |"
            for it in ex["items"]
        ]
        lines.append("")
    return "\n".join(lines)


def write_report(run: RunResult, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or run.out_dir or ".")
    assets_dir = out_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pii_stats = next((r.stats for r in run.stages if r.name == "pii"), {})
    pngs = {
        "funnel": chart_funnel(run),
        "lengths": chart_lengths(run),
        "clusters": chart_clusters(run),
        "pii": chart_pii(pii_stats),
    }
    pngs = {k: v for k, v in pngs.items() if v}
    rel = {}
    for k, v in pngs.items():
        (assets_dir / f"{k}.png").write_bytes(v)
        rel[k] = f"report_assets/{k}.png"
    ctx = build_context(run)
    env = Environment(autoescape=True)
    html_out = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8")).render(
        c=ctx, charts={k: base64.b64encode(v).decode() for k, v in pngs.items()}
    )
    paths = {"html": out_dir / "report.html", "markdown": out_dir / "report.md"}
    paths["html"].write_text(html_out, encoding="utf-8")
    paths["markdown"].write_text(render_markdown(ctx, rel), encoding="utf-8")
    return paths
