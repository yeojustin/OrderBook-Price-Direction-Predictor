from __future__ import annotations

import time
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

TITLE_ART = """\
██████╗ ██████╗ ██╗ ██████╗███████╗    ███╗   ███╗ ██████╗ ██╗   ██╗███████╗
██╔══██╗██╔══██╗██║██╔════╝██╔════╝    ████╗ ████║██╔═══██╗██║   ██║██╔════╝
██████╔╝██████╔╝██║██║     █████╗      ██╔████╔██║██║   ██║██║   ██║█████╗
██╔═══╝ ██╔══██╗██║██║     ██╔══╝      ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══╝
██║     ██║  ██║██║╚██████╗███████╗    ██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ███████╗
╚═╝     ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝

██████╗ ██████╗ ███████╗██████╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗
██████╔╝██████╔╝█████╗  ██║  ██║
██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║
██║     ██║  ██║███████╗██████╔╝
╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝
"""


def title_panel() -> Align:
    title = Text(TITLE_ART, style="bold white", no_wrap=True, justify="center")
    caption = Text.assemble(
        ("with ", "dim"),
        ("L2 order book ", "dim"),
        ("bid", "green"),
        ("-", "dim"),
        ("ask", "red"),
        (" spread", "dim"),
    )
    caption.justify = "center"
    return Group(Align.center(title), Align.center(caption))


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return "·" * width
    chars = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return chars[0] * len(vals)
    return "".join(chars[int((v - lo) / (hi - lo) * 7)] for v in vals)


def prob_bar(p: float, width: int = 16) -> Text:
    filled = round(p * width)
    bar = Text()
    bar.append("█" * filled, style="green" if p >= 0.5 else "red")
    bar.append("░" * (width - filled), style="dim")
    bar.append(f"  {p:.1%}", style="white")
    return bar


def feat_bar(val: float, lo: float, hi: float, width: int = 8) -> Text:
    clamped = max(lo, min(hi, val))
    frac = (clamped - lo) / max(hi - lo, 1e-12)
    bar = Text()
    bar.append("▮" * round(frac * width), style="white")
    bar.append("▯" * (width - round(frac * width)), style="dim")
    return bar


def signal_panel(
    symbol: str, mid: float, best_bid: float, best_ask: float,
    p_up: float, pred: int, threshold: float, status: str, ticks: int,
    mids: list[float], probs: list[float],
) -> Panel:
    sig = "▲  UP" if pred == 1 else "▼  DOWN"
    sig_style = "bold green" if pred == 1 else "bold red"
    st_style = "white" if status == "connected" else "dim"
    conf = abs(p_up - 0.5) / 0.5
    conf_lbl = "HIGH" if conf > 0.6 else "MED" if conf > 0.3 else "LOW"

    body = Group(
        Text.assemble(
            ("symbol ", "dim"), (symbol.upper(), "white"),
            ("  price ", "dim"), (f"${mid:,.2f}", "bold white"),
            ("  bid ", "dim"), (f"${best_bid:,.2f}", "white"),
            ("  ask ", "dim"), (f"${best_ask:,.2f}", "white"),
            ("  status ", "dim"), (status, st_style),
            ("  ticks:", "dim"), (str(ticks), "dim"),
        ),
        Text.assemble(
            ("signal ", "dim"), (f"{sig:<12}", sig_style),
            ("conf:", "dim"), (f"{conf_lbl}  ", "white"),
            ("p(up) ", "dim"), prob_bar(p_up),
            ("  threshold:", "dim"), (f"{threshold:.4f}", "dim"),
        ),
        Text.assemble(("price ", "dim"), (sparkline(mids), "white")),
        Text.assemble(("p(up) ", "dim"), (sparkline(probs), "dim white")),
    )
    return Panel(body, title="[dim]signal[/dim]", border_style="dim", padding=(0, 1))


def features_panel(feat: dict[str, float] | None, model_name: str, window_sec: int) -> Panel:
    header = Text.assemble(
        ("model:", "dim"), (f"{model_name}  ", "white"),
        ("window:", "dim"), (f"{window_sec}s  ", "white"),
        ("features: ", "dim"), ("OBI · Spread · DepthSkew · Momentum", "dim"),
    )
    if feat is None:
        return Panel(
            Group(header, Text("  warming up — waiting for ticks…", style="dim")),
            title="[dim]features[/dim]", border_style="dim", padding=(0, 1),
        )

    def frow(label: str, val: float, lo: float, hi: float, desc: str, note: str) -> Text:
        t = Text()
        t.append(f"  {label:<11}", style="white")
        t.append(f"{val:+.5f}  ", style="white")
        t.append_text(feat_bar(val, lo, hi))
        t.append(f"  {desc}  ", style="dim")
        t.append(f"→ {note}", style="dim")
        return t

    obi, sprd, dsk, mom = feat["obi"], feat["spread"], feat["depth_skew"], feat["momentum_10s"]
    rule = Rule(style="dim")
    return Panel(
        Group(
            header, Text(""),
            frow("OBI", obi, -1.0, 1.0, "(bestBidQty−bestAskQty)/sum", "Buyers dominate" if obi > 0 else "Sellers dominate"),
            rule,
            frow("Spread", sprd, 0.0, 5.0, "Best Ask − Best Bid", f"${sprd:.2f} gap"),
            rule,
            frow("Depth Skew", dsk, -1.0, 1.0, "top-5 bid vs ask depth", "More bids" if dsk > 0 else "More asks"),
            rule,
            frow("Momentum", mom, -0.01, 0.01, "10-tick mid-price Δ%", "Rising" if mom > 0 else "Falling"),
        ),
        title="[dim]features & how it predicts[/dim]", border_style="dim", padding=(0, 1),
    )


def history_table(rows: list[dict], max_rows: int) -> Table:
    t = Table(show_header=True, header_style="dim", border_style="dim", box=None, padding=(0, 1), expand=True)
    t.add_column("TIME", style="dim", width=9)
    t.add_column("PRICE", justify="right", width=12)
    t.add_column("P(UP)", justify="right", width=7)
    t.add_column("SIG", justify="center", width=6)
    t.add_column("OBI", justify="right", width=8)
    t.add_column("SPREAD", justify="right", width=7)
    t.add_column("DPTH SKW", justify="right", width=9)
    t.add_column("MOMENTUM", justify="right", width=11)
    for r in reversed(rows[-max_rows:]):
        sig = "▲ UP" if int(r["pred_up"]) == 1 else "▼ DN"
        sty = "green" if sig.startswith("▲") else "red"
        t.add_row(
            time.strftime("%H:%M:%S", time.localtime(float(r["ts"]))),
            f"${float(r['mid_price']):,.2f}",
            f"{float(r['proba_up']):.3f}",
            f"[{sty}]{sig}[/{sty}]",
            f"{float(r.get('obi', 0)):+.4f}",
            f"${float(r.get('spread', 0)):.2f}",
            f"{float(r.get('depth_skew', 0)):+.4f}",
            f"{float(r.get('momentum_10s', 0)):+.6f}",
        )
    return t


def footer_text(symbol: str, mode: str, ticks: int, status: str, model_name: str) -> Text:
    st_style = "white" if status == "connected" else "dim"
    t = Text()
    t.append(f"  mode:{mode}  ", "dim")
    t.append(f"symbol:{symbol.upper()}  ", "white")
    t.append(f"model:{model_name}  ", "dim")
    t.append(f"ticks:{ticks}  ", "dim")
    t.append("status:", "dim")
    t.append(status, st_style)
    t.append("  │  Ctrl+C to quit", "dim")
    return t

