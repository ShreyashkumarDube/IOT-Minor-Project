"""One Plotly look for every chart: transparent background (white in light mode), light horizontal grid,
12px text, legend at the bottom, the five status colours, thresholds as thin dotted lines labelled at the right.
Charts carry no title of their own; the section heading above them is the title."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

from .theme import ACCENT, ENGINE_COLORS, STATUS_COLORS

GRID = 'rgba(128,128,128,0.18)'
LINK = 'rgba(128,128,128,0.55)'
# Distinct but quiet line colours for per-device trust lines (status colours are reserved for meaning).
SERIES = ['#2563EB', '#0F766E', '#9333EA', '#B45309', '#BE185D', '#475569', '#0369A1', '#4D7C0F', '#A16207', '#7F1D1D']
ZONE_ORDER = ['Public IoT', 'Private IoT', 'Critical Zone', 'Untrusted']


def style(fig: go.Figure, height: int = 320, legend: bool = True, ytitle: Optional[str] = None, xtitle: Optional[str] = None) -> go.Figure:
    right = max(8, fig.layout.margin.r or 0)
    fig.update_layout(height=height, margin=dict(l=8, r=right, t=8, b=8), font={'size': 12},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=legend,
                      legend={'orientation': 'h', 'y': -0.18, 'x': 0, 'font': {'size': 10}, 'title': None})
    fig.update_xaxes(showgrid=False, zeroline=False, title=xtitle, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=ytitle)
    return fig


def add_thresholds(fig: go.Figure, allow: float, block: float, transform=lambda y: y) -> go.Figure:
    fig.update_layout(margin=dict(r=64))   # room for the labels at the right edge
    for y, name, colour in ((allow, 'ALLOW', STATUS_COLORS['Trusted']), (block, 'BLOCK', STATUS_COLORS['Isolated'])):
        fig.add_hline(y=transform(y), line_dash='dot', line_width=1, line_color=colour,
                      annotation_text=f'{name} {y:.2f}', annotation_position='right', annotation_font={'size': 10, 'color': colour})
    return fig


# ----- trust over time -----------------------------------------------------------------------------------
def trust_figure(worlds: Dict[str, object], devices: Sequence[str], attack_starts: Optional[Dict[str, int]] = None,
                 height: int = 360) -> go.Figure:
    """Trust per device: solid = Bayesian world, dashed = LLM world (gaps while the LLM is PENDING)."""
    fig = go.Figure()
    for i, d in enumerate(sorted(devices)):
        colour = SERIES[i % len(SERIES)]
        for engine, world in worlds.items():
            hist = pd.DataFrame(world.trust_history)
            if hist.empty or d not in hist:
                continue
            fig.add_trace(go.Scatter(x=hist['tick'], y=hist[d], mode='lines', name=f"{d} {'Bayes' if engine == 'bayesian' else 'LLM'}",
                                     line={'color': colour, 'width': 1.8, 'dash': 'solid' if engine == 'bayesian' else 'dash'},
                                     connectgaps=False, legendgroup=d))
    ctl = next(iter(worlds.values())).controller
    add_thresholds(fig, ctl.allow_threshold, ctl.block_threshold)
    for d, t in (attack_starts or {}).items():
        fig.add_vline(x=t, line_dash='dot', line_width=1, line_color=STATUS_COLORS['Probation'],
                      annotation_text=f'{d} attack', annotation_font={'size': 10})
    fig.update_yaxes(range=[0, 1])
    return style(fig, height, ytitle='trust', xtitle='tick')


# ----- topology ------------------------------------------------------------------------------------------
def zone_layout(device_zones: Dict[str, str]) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    """Fixed positions: one vertical column per zone (known zones first), the controller on the right."""
    zones = [z for z in ZONE_ORDER if z in device_zones.values()] + sorted(set(device_zones.values()) - set(ZONE_ORDER))
    pos = {}
    for x, zone in enumerate(zones):
        members = sorted(d for d, z in device_zones.items() if z == zone)
        for j, d in enumerate(members):
            pos[d] = (x, -(j - (len(members) - 1) / 2))
    pos['CONTROLLER'] = (len(zones) + 0.4, 0)
    return pos, zones


def topology_figure(devices: Dict[str, object], connections: Iterable[Tuple[str, str]], status: Dict[str, str],
                    hover: Dict[str, str], controller_label: str = 'Controller', height: int = 380) -> go.Figure:
    """devices: id -> object with .zone; status: id -> status name (colour); hover: id -> extra hover text."""
    pos, zones = zone_layout({d: dev.zone for d, dev in devices.items()})
    fig = go.Figure()
    for src, dst in connections:
        if src not in pos or dst not in pos:
            continue
        cut = status.get(src) == 'Quarantined' or status.get(dst) == 'Quarantined'
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        colour = STATUS_COLORS['Isolated'] if cut else LINK
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', hoverinfo='skip', showlegend=False,
                                 line={'width': 1.2, 'color': colour, 'dash': 'dash' if cut else 'solid'}))
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, text='',
                           arrowhead=2, arrowsize=0.9, arrowwidth=1, arrowcolor=colour, standoff=9)
    for d, dev in devices.items():
        x, y = pos[d]
        s = status.get(d, 'Pending')
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', text=[d], textposition='middle right', textfont={'size': 10},
                                 marker={'size': 13, 'color': STATUS_COLORS.get(s, STATUS_COLORS['Pending'])}, showlegend=False,
                                 hovertemplate=f'{d} · {hover.get(d, "")}<br>{s}<extra></extra>'))
    cx, cy = pos['CONTROLLER']
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers+text', text=[controller_label], textposition='bottom center',
                             textfont={'size': 11}, marker={'size': 15, 'symbol': 'square', 'color': ACCENT}, showlegend=False,
                             hoverinfo='skip'))
    top = max((y for _, y in pos.values()), default=0) + 0.8
    for x, zone in enumerate(zones):
        fig.add_annotation(x=x, y=top, text=zone, showarrow=False, font={'size': 11, 'color': '#6B7280'})
    for s, colour in STATUS_COLORS.items():   # one small legend row
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker={'size': 8, 'color': colour}, name=s))
    fig.update_xaxes(visible=False, range=[-0.6, cx + 0.6])
    fig.update_yaxes(visible=False)
    style(fig, height)
    fig.update_yaxes(showgrid=False)
    return fig


# ----- comparison charts --------------------------------------------------------------------------------
def bar_pair(categories, bayes_vals, llm_vals, ytitle: str, height: int = 300) -> go.Figure:
    fig = go.Figure([go.Bar(name='Bayesian', x=categories, y=bayes_vals, marker_color=ENGINE_COLORS['bayesian']),
                     go.Bar(name='LLM', x=categories, y=llm_vals, marker_color=ENGINE_COLORS['llm'])])
    fig.update_layout(barmode='group', bargap=0.35)
    return style(fig, height, ytitle=ytitle)


def latency_box(bayes_ms, llm_ms, height: int = 300) -> go.Figure:
    fig = go.Figure([go.Box(y=bayes_ms, name='Bayesian (per request)', marker_color=ENGINE_COLORS['bayesian']),
                     go.Box(y=llm_ms, name='LLM (per evaluation)', marker_color=ENGINE_COLORS['llm'])])
    fig.update_yaxes(type='log')
    return style(fig, height, legend=False, ytitle='ms (log scale)')


def line(x, y, ytitle: str, height: int = 300, yrange=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', line={'color': ACCENT, 'width': 1.6}))
    if yrange:
        fig.update_yaxes(range=yrange)
    return style(fig, height, legend=False, ytitle=ytitle, xtitle='tick')


def waterfall(labels: Sequence[str], deltas: Sequence[float], start: float, allow: float, block: float, logit,
              height: int = 320) -> go.Figure:
    fig = go.Figure(go.Waterfall(x=['L₀'] + list(labels) + ['L final'], measure=['absolute'] + ['relative'] * len(labels) + ['total'],
                                 y=[start] + list(deltas) + [0], connector={'line': {'color': GRID}},
                                 increasing={'marker': {'color': STATUS_COLORS['Trusted']}},
                                 decreasing={'marker': {'color': STATUS_COLORS['Isolated']}},
                                 totals={'marker': {'color': ACCENT}}))
    add_thresholds(fig, allow, block, transform=logit)
    return style(fig, height, legend=False, ytitle='log-odds L')
