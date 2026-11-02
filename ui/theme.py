"""Status palette, number formatters and small HTML helpers shared by every page.

Colour is used only for meaning, and every colour comes from STATUS_COLORS (plus the blue ACCENT
for interactive elements). The stylesheet reads the same colours through CSS variables.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

ACCENT = '#2563EB'
STATUS_COLORS = {
    'Trusted': '#16A34A',      # green  = Trusted / ALLOW
    'Suspicious': '#D97706',   # amber  = Suspicious / STEP-UP
    'Isolated': '#DC2626',     # red    = Isolated / BLOCK
    'Quarantined': '#6B7280',  # grey   = Quarantined
    'Pending': '#9CA3AF',      # grey   = Pending (LLM not evaluated yet)
    'Probation': '#7C3AED',    # purple = Probation
}
ENGINE_COLORS = {'bayesian': ACCENT, 'llm': STATUS_COLORS['Probation']}   # "Better" column and engine names
ACTION_STATUS = {'ALLOW': 'Trusted', 'STEP-UP': 'Suspicious', 'BLOCK': 'Isolated', 'DROPPED': 'Quarantined', 'PENDING': 'Pending'}
_CSS = Path(__file__).with_name('style.css')


def load_css() -> None:
    """Inject the stylesheet plus the palette as CSS variables (once per page run)."""
    variables = ';'.join(f'--zt-{k.lower()}:{v}' for k, v in STATUS_COLORS.items())
    st.markdown(f'<style>:root{{{variables}}}\n{_CSS.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


# ----- numbers -------------------------------------------------------------------------------------------
def fmt_trust(x) -> str:
    """Trust with 2 decimals; None means the LLM has not evaluated the device yet."""
    return 'PENDING' if x is None else f'{x:.2f}'


def fmt_pct(x) -> str:
    return '—' if x is None else f'{x:.0%}'


def fmt_ms(ms) -> str:
    """0.02 ms · 154 ms · 1.4 s"""
    if ms is None:
        return '—'
    if ms >= 1000:
        return f'{ms / 1000:.1f} s'
    return f'{ms:.2f} ms' if ms < 10 else f'{ms:.0f} ms'


def fmt_tick(t) -> str:
    return '—' if t is None else f't = {t}'


# ----- small pieces of markup ---------------------------------------------------------------------------
def status_color(status: str) -> str:
    """Colour for a status or an action word (falls back to the Pending grey)."""
    key = status.split()[0] if status else ''
    return STATUS_COLORS.get(key) or STATUS_COLORS.get(ACTION_STATUS.get(key, ''), STATUS_COLORS['Pending'])


def status_badge(status: str) -> str:
    """Plain-text badge for tables: a dot and the status (the table colours the dot)."""
    return f'● {status}'


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<div class="zt-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def status_line(parts: Iterable[str]) -> None:
    st.markdown('<div class="zt-status">' + ' · '.join(html.escape(p) for p in parts) + '</div>', unsafe_allow_html=True)


def kpi_row(items: Sequence[Tuple[str, object]], help_texts: Optional[Sequence[Optional[str]]] = None) -> None:
    """A row of small KPI cards (Streamlit metrics, styled by style.css)."""
    cols = st.columns(len(items))
    for i, (col, (label, value)) in enumerate(zip(cols, items)):
        col.metric(label, value, help=(help_texts[i] if help_texts else None))


_FINAL_CLASS = {'ALLOW': 'k-allow', 'STEP-UP': 'k-step', 'BLOCK': 'k-block', 'DROPPED': 'k-drop'}


def step_row_html(stages: Sequence[Tuple[str, str, str]], final_action: Optional[str] = None) -> str:
    """The 7-stage decision flow as a static row of numbered steps. stages = [(title, value, detail), ...]."""
    cards = []
    for i, (title, value, detail) in enumerate(stages):
        cls = 'zt-step' + (f" {_FINAL_CLASS.get(final_action, '')}" if i == len(stages) - 1 and final_action else '')
        cards.append(f'<div class="{cls}"><span class="n">{i + 1}</span><b>{html.escape(title)}</b>'
                     f'<div class="v">{html.escape(str(value))}</div><div class="d">{html.escape(str(detail))}</div></div>')
    return '<div class="zt-steps">' + ''.join(cards) + '</div>'
