"""E-Commerce Performance Dashboard (entry point).

Run from the submission folder:  streamlit run dashboard/dashboard.py

Module layout:
    analytics.py   data loading, filtering, and every aggregate (no UI)
    charts.py      Altair chart builders
    components.py  reusable UI pieces (header, KPI cards, callouts, cards)
    views.py       one render function per tab
    theme.py       color tokens shared by the charts and style.css
    style.css      page styling (overlay filter drawer, cards, callouts)
"""
import pandas as pd
import streamlit as st

import analytics
import views
from components import Kpi, compact, kpi_cards, load_styles, page_header

st.set_page_config(page_title='E-Commerce Performance Dashboard', page_icon='📦', layout='wide',
                   initial_sidebar_state='expanded')
load_styles()

# ---------------------------------------------------------------- Filters
DEFAULT_RANGE = (analytics.PERIOD_START.date(), analytics.PERIOD_END.date())
st.session_state.setdefault('date_range', DEFAULT_RANGE)
st.session_state.setdefault('states', [])


def reset_filters():
    st.session_state['date_range'] = DEFAULT_RANGE
    st.session_state['states'] = []


with st.sidebar:
    st.header('Filters')
    date_range = st.date_input('Purchase date', key='date_range', format='DD/MM/YYYY',
                               min_value=DEFAULT_RANGE[0], max_value=DEFAULT_RANGE[1])
    states = st.multiselect('Customer state', options=sorted(analytics.STATE_NAMES), key='states',
                            format_func=lambda code: f'{code} · {analytics.STATE_NAMES[code]}',
                            placeholder='All states')
    st.button('Reset filters', on_click=reset_filters, width='stretch')
    with st.expander('About the data'):
        st.markdown(
            '- **Source:** E-Commerce Public Dataset (Olist, Brazil).\n'
            '- **Scope:** delivered orders purchased from 1 Jan 2017 to 31 Aug 2018.\n'
            '- **Revenue:** sum of item prices, excluding freight.\n'
            '- **Customer spending:** payment value, including freight.\n'
            '- **Repeat customer:** a customer with 2 or more orders in the selected period.'
        )

if len(date_range) != 2:
    st.info('Select both a start date and an end date in the Filters panel.')
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
selected_states = tuple(states)
items = analytics.filter_items(start, end, selected_states)
orders = analytics.to_orders(items)

state_text = ', '.join(analytics.STATE_NAMES[s] for s in states) or 'All states'
page_header('E-Commerce Performance Dashboard',
            f'Delivered orders · {start:%d %b %Y} – {end:%d %b %Y} · {state_text}')

if orders.empty:
    st.warning('No delivered orders match the selected filters. Widen the date range or clear the state filter.')
    st.stop()

# ---------------------------------------------------------------- KPIs
current = analytics.kpis(orders)
previous = {}
period = analytics.previous_period(start, end)
if period:
    previous_orders = analytics.to_orders(analytics.filter_items(*period, selected_states))
    if not previous_orders.empty:
        previous = analytics.kpis(previous_orders)


def change(metric):
    return analytics.pct_change(current[metric], previous.get(metric))


kpi_cards([
    Kpi('Orders', f'{current["orders"]:,}', 'Delivered orders purchased in the selected period.',
        change('orders')),
    Kpi('Revenue (BRL)', compact(current['revenue']), 'Sum of item prices, excluding freight.',
        change('revenue')),
    Kpi('Customers', f'{current["customers"]:,}', 'Unique customers with at least one delivered order.',
        change('customers')),
    Kpi('Median order (BRL)', f'{current["median_order"]:,.2f}',
        'Median of item prices per order, excluding freight.', change('median_order')),
    Kpi('Repeat customers', f'{current["repeat_rate"]:.1%}',
        'Share of customers with 2 or more orders in the selected period. '
        'Short periods give lower values because there is less time to buy again.'),
])

# ---------------------------------------------------------------- Tabs
overview, basket, retention, segments, geography = st.tabs(
    ['Overview', 'Basket size', 'Retention', 'Segments', 'Geography']
)
with overview:
    views.render_overview(items, orders)
with basket:
    views.render_basket(items, orders)
with retention:
    views.render_retention(analytics.to_customers(orders))
with segments:
    views.render_segments(start, end, selected_states, current['customers'])
with geography:
    views.render_geography(orders)
