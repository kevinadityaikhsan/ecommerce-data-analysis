"""Data layer: loads main_data.csv and computes every aggregate shown in the dashboard.

Functions here return DataFrames, dicts, or numbers and never render anything,
so the business logic can be read and tested apart from the layout.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / 'main_data.csv'
PERIOD_START = pd.Timestamp('2017-01-01')
PERIOD_END = pd.Timestamp('2018-08-31')
BASKET_LABELS = ['1 item', '2 items', '3 items', '4+ items']

SEGMENTS = pd.DataFrame({
    'segment': ['Loyal High-Value', 'Loyal', 'New High-Value', 'New Low-Value',
                'Lapsed High-Value', 'Lapsed Low-Value'],
    'definition': ['Repeat buyer, top-half spending', 'Repeat buyer, bottom-half spending',
                   'One order, recent, top-half spending', 'One order, recent, bottom-half spending',
                   'One order, not recent, top-half spending', 'One order, not recent, bottom-half spending'],
    'suggested_action': ['Reward and retain (loyalty perks, early access)',
                         'Increase basket size (bundles, free-shipping threshold)',
                         'Secure a second order (time-limited voucher)',
                         'Secure a second order with low-cost incentives',
                         'Win back (personalized offer)',
                         'Low-cost reactivation (newsletter)'],
})
SEGMENT_ORDER = SEGMENTS['segment'].tolist()

STATE_NAMES = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AM': 'Amazonas', 'AP': 'Amapá', 'BA': 'Bahia', 'CE': 'Ceará',
    'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão',
    'MG': 'Minas Gerais', 'MS': 'Mato Grosso do Sul', 'MT': 'Mato Grosso', 'PA': 'Pará',
    'PB': 'Paraíba', 'PE': 'Pernambuco', 'PI': 'Piauí', 'PR': 'Paraná', 'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte', 'RO': 'Rondônia', 'RR': 'Roraima', 'RS': 'Rio Grande do Sul',
    'SC': 'Santa Catarina', 'SE': 'Sergipe', 'SP': 'São Paulo', 'TO': 'Tocantins',
}


# ---------------------------------------------------------------- Loading & filtering
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['order_purchase_timestamp'])
    df['basket_group'] = pd.cut(df['item_count'], bins=[0, 1, 2, 3, np.inf], labels=BASKET_LABELS)
    return df


@st.cache_data
def filter_items(start, end, states):
    """Items purchased from `start` to `end` (inclusive dates) in the selected states."""
    df = load_data()
    mask = df['order_purchase_timestamp'].between(start, end + pd.Timedelta(days=1), inclusive='left')
    if states:
        mask &= df['customer_state'].isin(states)
    return df[mask]


def to_orders(items):
    # Order-level columns repeat on every item row, so keep one row per order.
    return items.drop_duplicates('order_id')


def to_customers(orders):
    customers = orders.groupby('customer_unique_id').agg(
        orders=('order_id', 'nunique'), total_spending=('payment_value', 'sum')
    )
    customers['spending_per_order'] = customers['total_spending'] / customers['orders']
    customers['customer_type'] = np.where(customers['orders'] > 1, 'Repeat', 'One-time')
    return customers


def previous_period(start, end):
    """The period of the same length right before `start`, or None if it starts before the data."""
    length = end - start + pd.Timedelta(days=1)
    previous_start = start - length
    if previous_start < PERIOD_START:
        return None
    return previous_start, start - pd.Timedelta(days=1)


# ---------------------------------------------------------------- KPIs
def kpis(orders):
    customers = to_customers(orders)
    return {
        'orders': len(orders),
        'revenue': orders['order_value'].sum(),
        'customers': len(customers),
        'median_order': orders['order_value'].median(),
        'repeat_rate': (customers['customer_type'] == 'Repeat').mean(),
    }


def pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0 or pd.isna(current):
        return None
    return current / previous - 1


# ---------------------------------------------------------------- Overview
def monthly_summary(orders):
    monthly = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).agg(
        orders=('order_id', 'count'),
        revenue=('order_value', 'sum'),
        multi_share=('item_count', lambda x: (x >= 2).mean()),
    ).reset_index()
    monthly['month'] = monthly['order_purchase_timestamp'].dt.to_timestamp()
    return monthly


def top_categories_by_revenue(items, n=10):
    top = (items.groupby('product_category')
           .agg(revenue=('price', 'sum'), items=('price', 'size'))
           .sort_values('revenue', ascending=False).head(n).reset_index())
    top['revenue_share'] = top['revenue'] / items['price'].sum()
    return top


# ---------------------------------------------------------------- Basket size
def basket_summary(orders):
    basket = orders.groupby('basket_group', observed=False).agg(
        orders=('order_id', 'count'), median_value=('order_value', 'median')
    ).reset_index()
    basket['share'] = basket['orders'] / basket['orders'].sum()
    return basket


def single_vs_multi(orders):
    """Median order value of single- and multi-item orders, and the multi-item share."""
    multi = orders['item_count'] >= 2
    return {
        'single_median': orders.loc[~multi, 'order_value'].median(),
        'multi_median': orders.loc[multi, 'order_value'].median(),
        'multi_share': multi.mean(),
    }


def top_categories_in_multi_item_orders(items, n=10):
    multi_items = items[items['item_count'] >= 2]
    top = (multi_items.groupby('product_category')['order_id'].nunique()
           .sort_values(ascending=False).head(n).rename('orders').reset_index())
    top['share'] = top['orders'] / multi_items['order_id'].nunique()
    return top


# ---------------------------------------------------------------- Retention
def customer_type_summary(customers):
    summary = customers.groupby('customer_type').agg(
        customers=('orders', 'count'),
        spending=('total_spending', 'sum'),
        median_total=('total_spending', 'median'),
        median_per_order=('spending_per_order', 'median'),
    ).reindex(['One-time', 'Repeat'])
    summary['customer_share'] = summary['customers'] / summary['customers'].sum()
    summary['spending_share'] = summary['spending'] / summary['spending'].sum()
    return summary


# ---------------------------------------------------------------- RFM segments
def quartile_score(values, labels):
    try:
        return pd.qcut(values, q=4, labels=labels).astype(int)
    except ValueError:
        # Small filtered samples can have duplicate quartile edges; ranking breaks the ties.
        return pd.qcut(values.rank(method='first'), q=4, labels=labels).astype(int)


@st.cache_data
def build_rfm(start, end, states):
    """RFM metrics and segments, using the same rules as the notebook."""
    orders = to_orders(filter_items(start, end, states))
    rfm = orders.groupby('customer_unique_id').agg(
        last_purchase=('order_purchase_timestamp', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum'),
    )
    rfm['recency'] = (end + pd.Timedelta(days=1) - rfm['last_purchase']).dt.days
    rfm['r_score'] = quartile_score(rfm['recency'], labels=[4, 3, 2, 1])
    rfm['m_score'] = quartile_score(rfm['monetary'], labels=[1, 2, 3, 4])
    is_repeat = rfm['frequency'] > 1
    is_recent = rfm['r_score'] >= 3
    is_high_value = rfm['m_score'] >= 3
    rfm['segment'] = np.select(
        [is_repeat & is_high_value, is_repeat, is_recent & is_high_value, is_recent, is_high_value],
        SEGMENT_ORDER[:5], default=SEGMENT_ORDER[5],
    )
    return rfm


def segment_summary(rfm):
    segments = rfm.groupby('segment').agg(
        customers=('recency', 'count'),
        median_recency=('recency', 'median'),
        median_monetary=('monetary', 'median'),
        spending=('monetary', 'sum'),
    ).reindex(SEGMENT_ORDER).fillna(0).reset_index()
    segments['customer_share'] = segments['customers'] / segments['customers'].sum()
    segments['spending_share'] = segments['spending'] / segments['spending'].sum()
    return segments.merge(SEGMENTS, on='segment')


# ---------------------------------------------------------------- Geography
def state_summary(orders):
    state = (orders.assign(freight_ratio=orders['freight_value'] / orders['order_value'])
             .groupby('customer_state').agg(orders=('order_id', 'count'), revenue=('order_value', 'sum'),
                                             freight_ratio=('freight_ratio', 'median'))
             .reset_index())
    state['state_name'] = state['customer_state'].map(STATE_NAMES)
    state['revenue_share'] = state['revenue'] / state['revenue'].sum()
    return state


def map_points(orders):
    """One point per customer zip code area with its order count and revenue."""
    return (orders.dropna(subset=['geolocation_lat'])
            .groupby(['geolocation_lat', 'geolocation_lng', 'customer_city', 'customer_state'])
            .agg(orders=('order_id', 'count'), revenue=('order_value', 'sum'))
            .reset_index())
