"""Reusable UI components rendered as HTML blocks styled by style.css."""
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import theme

STYLESHEET = Path(__file__).parent / 'style.css'


def load_styles():
    st.html(f'<style>{theme.css_variables()}\n{STYLESHEET.read_text(encoding="utf-8")}</style>')


# Formatting
def brl(value):
    if pd.isna(value):
        return '–'
    if abs(value) >= 1e6:
        return f'BRL {value / 1e6:,.2f}M'
    return f'BRL {value:,.2f}' if abs(value) < 1e3 else f'BRL {value:,.0f}'


def compact(value):
    return f'{value / 1e6:,.2f}M' if abs(value) >= 1e6 else f'{value:,.0f}'


# Components
def page_header(title, subtitle):
    st.html(f'<h1 class="page-title">{escape(title)}</h1><p class="page-subtitle">{escape(subtitle)}</p>')


@dataclass
class Kpi:
    label: str
    value: str
    description: str
    change: float | None = None


def kpi_cards(kpis):
    """Responsive grid of KPI cards; the description appears when hovering the label."""
    cards = []
    for kpi in kpis:
        if kpi.change is None:
            delta = '<div class="kpi__delta kpi__delta--none">&nbsp;</div>'
        else:
            arrow, modifier = ('▲', 'up') if kpi.change >= 0 else ('▼', 'down')
            delta = (f'<div class="kpi__delta kpi__delta--{modifier}">'
                     f'{arrow} {abs(kpi.change):.1%} vs. previous period</div>')
        cards.append(
            f'<div class="kpi"><div class="kpi__label" title="{escape(kpi.description)}">{escape(kpi.label)}</div>'
            f'<div class="kpi__value">{escape(kpi.value)}</div>{delta}</div>'
        )
    st.html(f'<div class="kpi-grid">{"".join(cards)}</div>')


def insight(text):
    """One-sentence answer shown as a callout; **double asterisks** mark the key numbers."""
    st.html(f'<div class="insight">{re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", escape(text))}</div>')


def chart_card(chart):
    with st.container(border=True):
        st.altair_chart(chart, width='stretch')


def map_card(deck, caption, height=480):
    with st.container(border=True):
        st.pydeck_chart(deck, height=height)
        st.caption(caption)
