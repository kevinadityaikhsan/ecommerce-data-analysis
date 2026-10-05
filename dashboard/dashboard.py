from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Same visual identity as the notebook: one highlight color, one secondary
# color for comparisons, and gray for context.
HIGHLIGHT = '#2a78d6'
SECONDARY = '#eb6834'
MUTED = '#c7c7c7'
TEXT = '#0b0b0b'
sns.set_theme(style='white', rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.titlepad': 12,
    'axes.labelcolor': '#52514e',
    'xtick.color': '#52514e',
    'ytick.color': '#52514e',
})

BASKET_LABELS = ['1 item', '2 items', '3 items', '4+ items']
SEGMENT_ORDER = ['Loyal High-Value', 'Loyal', 'New High-Value', 'New Low-Value',
                 'Lapsed High-Value', 'Lapsed Low-Value']


@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / 'main_data.csv', parse_dates=['order_purchase_timestamp'])
    df['basket_group'] = pd.cut(df['item_count'], bins=[0, 1, 2, 3, np.inf], labels=BASKET_LABELS)
    return df


def annotate_bars(ax, fmt='{:,.0f}'):
    for container in ax.containers:
        ax.bar_label(container, labels=[fmt.format(v) for v in container.datavalues],
                     padding=3, fontsize=10, color=TEXT)


def top3_colors(n, color=HIGHLIGHT):
    """Highlight the first three bars of a ranked chart and gray out the rest."""
    return [color if i < 3 else MUTED for i in range(n)]


def build_rfm(orders, reference_date):
    """RFM metrics and rule-based segments, using the same rules as the notebook."""
    rfm = orders.groupby('customer_unique_id').agg(
        last_purchase=('order_purchase_timestamp', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum'),
    )
    rfm['recency'] = (reference_date - rfm['last_purchase']).dt.days
    # rank() avoids duplicate bin edges when the filtered sample is small.
    rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

    is_repeat = rfm['frequency'] > 1
    is_recent = rfm['r_score'] >= 3
    is_high_value = rfm['m_score'] >= 3
    rfm['segment'] = np.select(
        [is_repeat & is_high_value, is_repeat, is_recent & is_high_value, is_recent, is_high_value],
        SEGMENT_ORDER[:5],
        default=SEGMENT_ORDER[5],
    )
    return rfm


# ---------------------------------------------------------------- Data & filters
st.set_page_config(page_title='E-Commerce Dashboard', page_icon='🛒', layout='wide')
items_df = load_data()

with st.sidebar:
    st.header('Filters')
    min_date = items_df['order_purchase_timestamp'].min().date()
    max_date = items_df['order_purchase_timestamp'].max().date()
    date_range = st.date_input('Purchase date', value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    states = st.multiselect('Customer state', sorted(items_df['customer_state'].unique()),
                            placeholder='All states')
    st.caption('Data: delivered orders from the E-Commerce Public Dataset (Olist), '
               'January 2017 – August 2018. Values in BRL.')

if len(date_range) != 2:
    st.info('Select a start date and an end date.')
    st.stop()
start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

mask = items_df['order_purchase_timestamp'].between(start_date, end_date, inclusive='left')
if states:
    mask &= items_df['customer_state'].isin(states)
items = items_df[mask]
# Order-level columns repeat on every item row, so keep one row per order.
orders = items.drop_duplicates('order_id')

if orders.empty:
    st.warning('No orders match the selected filters.')
    st.stop()

customers = orders.groupby('customer_unique_id').agg(
    orders=('order_id', 'nunique'), total_spending=('payment_value', 'sum')
)
customers['customer_type'] = np.where(customers['orders'] > 1, 'Repeat', 'One-time')

# ---------------------------------------------------------------- Header & KPIs
st.title('E-Commerce Public Dataset Dashboard')
st.caption(f'{date_range[0]:%d %b %Y} – {date_range[1]:%d %b %Y} · '
           f'{"all states" if not states else ", ".join(states)}')

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Delivered orders', f'{len(orders):,}')
col2.metric('Revenue (item prices)', f'BRL {orders["order_value"].sum():,.0f}')
col3.metric('Customers', f'{len(customers):,}')
col4.metric('Median order value', f'BRL {orders["order_value"].median():,.2f}')
col5.metric('Repeat customers', f'{(customers["customer_type"] == "Repeat").mean():.2%}')

tab_basket, tab_retention, tab_rfm, tab_geo, tab_trend = st.tabs(
    ['Basket Size', 'Customer Retention', 'RFM Segments', 'Geography', 'Monthly Trend']
)

# ---------------------------------------------------------------- Basket size
with tab_basket:
    st.subheader('How much more are multi-item orders worth?')
    basket = orders.groupby('basket_group', observed=False).agg(
        orders=('order_id', 'count'), median_value=('order_value', 'median')
    )
    basket['orders_pct'] = basket['orders'] / basket['orders'].sum() * 100

    multi = orders['item_count'] >= 2
    single_median = orders.loc[~multi, 'order_value'].median()
    multi_median = orders.loc[multi, 'order_value'].median()
    c1, c2, c3 = st.columns(3)
    c1.metric('Multi-item orders', f'{multi.mean():.2%}')
    c2.metric('Median single-item order', f'BRL {single_median:,.2f}')
    c3.metric('Median multi-item order', f'BRL {multi_median:,.2f}',
              delta=f'{multi_median / single_median - 1:+.1%} vs. single-item' if multi.any() else None)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(BASKET_LABELS, basket['orders_pct'], color=[HIGHLIGHT, MUTED, MUTED, MUTED])
    annotate_bars(axes[0], '{:.1f}%')
    axes[0].set_title('Share of orders by items per order')
    axes[0].set_xlabel('Items per order')
    axes[0].set_ylabel('Share of orders (%)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    axes[1].bar(BASKET_LABELS, basket['median_value'].fillna(0), color=[MUTED, HIGHLIGHT, HIGHLIGHT, HIGHLIGHT])
    annotate_bars(axes[1], 'BRL {:,.0f}')
    axes[1].set_title('Median order value by items per order')
    axes[1].set_xlabel('Items per order')
    axes[1].set_ylabel('Median order value (BRL)')
    plt.tight_layout()
    st.pyplot(fig)

    multi_items = items[items['item_count'] >= 2]
    if not multi_items.empty:
        top_categories = (multi_items.groupby('product_category')['order_id'].nunique()
                          .sort_values(ascending=False).head(10))
        share = top_categories / multi_items['order_id'].nunique() * 100
        fig, ax = plt.subplots(figsize=(15, 5.5))
        ax.barh(top_categories.index, top_categories.values, color=top3_colors(len(top_categories)))
        ax.invert_yaxis()
        ax.bar_label(ax.containers[0], labels=[f'{n:,} ({p:.1f}%)' for n, p in zip(top_categories, share)],
                     padding=3, fontsize=10, color=TEXT)
        ax.set_title('Top 10 product categories in multi-item orders')
        ax.set_xlabel('Number of multi-item orders containing the category')
        ax.set_xlim(0, top_categories.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------- Retention
with tab_retention:
    st.subheader('How much more do repeat customers spend?')
    summary = customers.groupby('customer_type').agg(
        customers=('orders', 'count'),
        revenue=('total_spending', 'sum'),
        median_spending=('total_spending', 'median'),
    ).reindex(['One-time', 'Repeat']).fillna(0)
    summary['Share of customers'] = summary['customers'] / summary['customers'].sum() * 100
    summary['Share of revenue'] = summary['revenue'] / summary['revenue'].sum() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric('Repeat customers', f'{int(summary.loc["Repeat", "customers"]):,}')
    c2.metric('Median spending, one-time', f'BRL {summary.loc["One-time", "median_spending"]:,.2f}')
    c3.metric('Median spending, repeat', f'BRL {summary.loc["Repeat", "median_spending"]:,.2f}')

    share_df = summary[['Share of customers', 'Share of revenue']].reset_index().melt(
        id_vars='customer_type', var_name='metric', value_name='pct'
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=share_df, x='metric', y='pct', hue='customer_type', hue_order=['One-time', 'Repeat'],
                palette=[MUTED, HIGHLIGHT], saturation=1, ax=axes[0])
    annotate_bars(axes[0], '{:.1f}%')
    axes[0].set_title('Share of customers vs. share of revenue')
    axes[0].set_xlabel(None)
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim(0, 110)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].legend(title='Customer type', frameon=False)

    axes[1].bar(summary.index, summary['median_spending'], color=[MUTED, HIGHLIGHT])
    annotate_bars(axes[1], 'BRL {:,.0f}')
    axes[1].set_title('Median total spending per customer')
    axes[1].set_xlabel('Customer type')
    axes[1].set_ylabel('Median total spending (BRL)')
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------- RFM
with tab_rfm:
    st.subheader('RFM customer segments')
    st.caption('Recency is measured from the day after the selected end date. Recency and monetary '
               'are binned into quartile scores (1–4); frequency separates one-time and repeat customers.')
    if len(customers) < 4:
        st.info('At least 4 customers are needed to build quartile scores.')
    else:
        rfm = build_rfm(orders, end_date)
        segments = rfm.groupby('segment').agg(
            customers=('recency', 'count'),
            median_recency_days=('recency', 'median'),
            median_monetary=('monetary', 'median'),
            revenue=('monetary', 'sum'),
        ).reindex(SEGMENT_ORDER).fillna(0)
        segments['Share of customers'] = segments['customers'] / segments['customers'].sum() * 100
        segments['Share of revenue'] = segments['revenue'] / segments['revenue'].sum() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric('Median recency', f'{rfm["recency"].median():,.0f} days')
        c2.metric('Mean frequency', f'{rfm["frequency"].mean():.2f} orders')
        c3.metric('Median monetary', f'BRL {rfm["monetary"].median():,.2f}')

        seg_share = segments[['Share of customers', 'Share of revenue']].reset_index().melt(
            id_vars='segment', var_name='metric', value_name='pct'
        )
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data=seg_share, y='segment', x='pct', hue='metric', order=SEGMENT_ORDER,
                    palette=[HIGHLIGHT, SECONDARY], saturation=1, ax=ax)
        annotate_bars(ax, '{:.1f}%')
        ax.set_title('Share of customers vs. share of revenue by segment')
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel(None)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_xlim(0, seg_share['pct'].max() * 1.15)
        ax.legend(title=None, frameon=False, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            segments[['customers', 'median_recency_days', 'median_monetary', 'revenue']]
            .style.format({'customers': '{:,.0f}', 'median_recency_days': '{:,.0f}',
                           'median_monetary': 'BRL {:,.2f}', 'revenue': 'BRL {:,.0f}'}),
            width='stretch',
        )

# ---------------------------------------------------------------- Geography
with tab_geo:
    st.subheader('Where are the orders?')
    located = orders.dropna(subset=['geolocation_lat'])
    points = (located.groupby(['geolocation_lat', 'geolocation_lng']).size()
              .rename('orders').reset_index())
    points['size'] = 500 + np.sqrt(points['orders']) * 500  # radius in meters
    st.map(points, latitude='geolocation_lat', longitude='geolocation_lng', size='size', color=HIGHLIGHT)
    st.caption(f'Each point is a customer zip code area; larger points have more orders. '
               f'{len(located) / len(orders):.2%} of orders have coordinates.')

    state = orders.groupby('customer_state').agg(
        orders=('order_id', 'count'),
        revenue=('order_value', 'sum'),
        freight=('freight_value', 'sum'),
    ).sort_values('revenue', ascending=False)
    state['revenue_pct'] = state['revenue'] / state['revenue'].sum() * 100
    state['freight_ratio_pct'] = state['freight'] / state['revenue'] * 100
    top_states = state.head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].barh(top_states.index, top_states['revenue_pct'], color=top3_colors(len(top_states)))
    axes[0].invert_yaxis()
    annotate_bars(axes[0], '{:.1f}%')
    axes[0].set_title('Top 10 customer states by revenue share')
    axes[0].set_xlabel('Share of revenue (%)')
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].set_xlim(0, top_states['revenue_pct'].max() * 1.15)

    top_freight = state.sort_values('freight_ratio_pct', ascending=False).head(10)
    axes[1].barh(top_freight.index, top_freight['freight_ratio_pct'],
                 color=top3_colors(len(top_freight), SECONDARY))
    axes[1].invert_yaxis()
    annotate_bars(axes[1], '{:.1f}%')
    axes[1].set_title('Top 10 states by freight as % of revenue')
    axes[1].set_xlabel('Total freight / total item revenue (%)')
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[1].set_xlim(0, top_freight['freight_ratio_pct'].max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------- Monthly trend
with tab_trend:
    st.subheader('Monthly orders and basket size')
    monthly = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).agg(
        orders=('order_id', 'count'),
        multi_item_pct=('item_count', lambda x: (x >= 2).mean() * 100),
    )
    monthly.index = monthly.index.to_timestamp()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(monthly.index, monthly['orders'], color=HIGHLIGHT, linewidth=2, marker='o', markersize=4)
    peak = monthly['orders'].idxmax()
    axes[0].annotate(f'Peak: {monthly.loc[peak, "orders"]:,}', (peak, monthly.loc[peak, 'orders']),
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10, color=TEXT)
    axes[0].set_title('Delivered orders per month')
    axes[0].set_ylabel('Orders')
    axes[0].set_ylim(0, monthly['orders'].max() * 1.15)

    axes[1].plot(monthly.index, monthly['multi_item_pct'], color=HIGHLIGHT, linewidth=2, marker='o', markersize=4)
    axes[1].set_title('Share of multi-item orders per month')
    axes[1].set_ylabel('Multi-item orders (%)')
    axes[1].set_ylim(0, max(20, monthly['multi_item_pct'].max() * 1.2))
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for ax in axes:
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.caption('Copyright © Kevin Aditya Ikhsan · Proyek Analisis Data')
