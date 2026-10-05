from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Same visual identity as the notebook: one highlight color for the key
# message, one secondary color for comparisons, and gray for context.
HIGHLIGHT = '#2a78d6'
SECONDARY = '#eb6834'
MUTED = '#b4b4b0'

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


# ---------------------------------------------------------------- Data
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / 'main_data.csv', parse_dates=['order_purchase_timestamp'])
    df['basket_group'] = pd.cut(df['item_count'], bins=[0, 1, 2, 3, np.inf], labels=BASKET_LABELS)
    return df


@st.cache_data
def filter_items(start, end, states):
    """Items purchased in [start, end] (inclusive dates) from the selected states."""
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


# ---------------------------------------------------------------- Formatting & charts
def brl(value):
    if pd.isna(value):
        return '–'
    if abs(value) >= 1e6:
        return f'BRL {value / 1e6:,.2f}M'
    return f'BRL {value:,.2f}' if abs(value) < 1e3 else f'BRL {value:,.0f}'


def compact(value):
    return f'{value / 1e6:,.2f}M' if abs(value) >= 1e6 else f'{value:,.0f}'


def pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0 or pd.isna(current):
        return None
    return f'{current / previous - 1:+.1%} vs. previous period'


def bar_chart(df, x, y, color, title, x_title, y_title, label_fmt, tooltip, horizontal=False,
              sort=None, height=320, axis_fmt=None):
    """Single-series bar chart with value labels; `color` is a column holding hex colors."""
    value, category = (x, y) if horizontal else (y, x)
    enc_value = alt.X if horizontal else alt.Y
    enc_cat = alt.Y if horizontal else alt.X
    base = alt.Chart(df).encode(
        enc_cat(f'{category}:N', sort=sort, title=y_title if horizontal else x_title,
                axis=alt.Axis(labelAngle=0, labelLimit=220)),
        enc_value(f'{value}:Q', title=x_title if horizontal else y_title,
                  scale=alt.Scale(domainMin=0, nice=True), axis=alt.Axis(format=axis_fmt or label_fmt, grid=True)),
        tooltip=tooltip,
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=alt.Color(f'{color}:N', scale=None, legend=None))
    labels = base.mark_text(
        align='left' if horizontal else 'center', baseline='middle' if horizontal else 'bottom',
        dx=4 if horizontal else 0, dy=0 if horizontal else -4, fontSize=12,
    ).encode(text=alt.Text(f'{value}:Q', format=label_fmt))
    return (bars + labels).properties(title=title, height=height)


def grouped_bar_chart(df, category, value, group, groups, colors, title, value_title, label_fmt,
                      tooltip, horizontal=False, sort=None, height=320, axis_fmt=None):
    """Bars grouped by `group` within each `category`, with a legend and value labels."""
    enc_value = alt.X if horizontal else alt.Y
    enc_cat = alt.Y if horizontal else alt.X
    offset = (alt.YOffset if horizontal else alt.XOffset)(f'{group}:N', sort=groups)
    base = alt.Chart(df).encode(
        enc_cat(f'{category}:N', sort=sort, title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)),
        enc_value(f'{value}:Q', title=value_title, scale=alt.Scale(domainMin=0, nice=True),
                  axis=alt.Axis(format=axis_fmt or label_fmt)),
        offset,
        tooltip=tooltip,
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color(f'{group}:N', sort=groups, scale=alt.Scale(domain=groups, range=colors),
                        legend=alt.Legend(title=None, orient='top'))
    )
    labels = base.mark_text(
        align='left' if horizontal else 'center', baseline='middle' if horizontal else 'bottom',
        dx=4 if horizontal else 0, dy=0 if horizontal else -4, fontSize=12,
    ).encode(text=alt.Text(f'{value}:Q', format=label_fmt))
    return (bars + labels).properties(title=title, height=height)


def line_chart(df, x, y, title, y_title, y_fmt, tooltip, height=300):
    base = alt.Chart(df).encode(
        alt.X(f'{x}:T', title=None, axis=alt.Axis(format='%b %Y', labelAngle=0, tickCount=6)),
        alt.Y(f'{y}:Q', title=y_title, scale=alt.Scale(domainMin=0, nice=True), axis=alt.Axis(format=y_fmt)),
    )
    line = base.mark_line(color=HIGHLIGHT, strokeWidth=2, point=alt.OverlayMarkDef(size=36, color=HIGHLIGHT))
    hover = alt.selection_point(fields=[x], nearest=True, on='pointerover', empty=False)
    rule = base.mark_rule(color=MUTED).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip
    ).add_params(hover)
    return (line + rule).properties(title=title, height=height)


def show(chart):
    st.altair_chart(chart, width='stretch')


# ---------------------------------------------------------------- Page & filters
st.set_page_config(page_title='E-Commerce Performance Dashboard', page_icon='📦', layout='wide')

DEFAULT_RANGE = (PERIOD_START.date(), PERIOD_END.date())
if 'date_range' not in st.session_state:
    st.session_state['date_range'] = DEFAULT_RANGE
    st.session_state['states'] = []


def reset_filters():
    st.session_state['date_range'] = DEFAULT_RANGE
    st.session_state['states'] = []


with st.sidebar:
    st.header('Filters')
    date_range = st.date_input('Purchase date', key='date_range', format='DD/MM/YYYY',
                               min_value=PERIOD_START.date(), max_value=PERIOD_END.date())
    states = st.multiselect('Customer state', options=sorted(STATE_NAMES), key='states',
                            format_func=lambda code: f'{code} · {STATE_NAMES[code]}',
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
    st.info('Select both a start date and an end date in the sidebar.')
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
selected_states = tuple(states)
items = filter_items(start, end, selected_states)
orders = to_orders(items)

st.title('E-Commerce Performance Dashboard')
state_text = 'all states' if not states else ', '.join(STATE_NAMES[s] for s in states)
st.caption(f'Delivered orders · {start:%d %b %Y} – {end:%d %b %Y} · {state_text}')

if orders.empty:
    st.warning('No delivered orders match the selected filters. Widen the date range or clear the state filter.')
    st.stop()

customers = to_customers(orders)

# Previous period of the same length, used for the KPI deltas.
length = end - start + pd.Timedelta(days=1)
previous_orders = to_orders(filter_items(start - length, start - pd.Timedelta(days=1), selected_states))
has_previous = (start - length) >= PERIOD_START and not previous_orders.empty


def previous(metric):
    return metric(previous_orders) if has_previous else None


# ---------------------------------------------------------------- KPIs
kpis = st.columns(5)
kpis[0].metric('Orders', f'{len(orders):,}', border=True,
               delta=pct_change(len(orders), previous(len)),
               help='Delivered orders purchased in the selected period.')
kpis[1].metric('Revenue (BRL)', compact(orders['order_value'].sum()), border=True,
               delta=pct_change(orders['order_value'].sum(), previous(lambda o: o['order_value'].sum())),
               help='Sum of item prices, excluding freight.')
kpis[2].metric('Customers', f'{len(customers):,}', border=True,
               delta=pct_change(len(customers), previous(lambda o: o['customer_unique_id'].nunique())),
               help='Unique customers (customer_unique_id) with at least one delivered order.')
kpis[3].metric('Median order (BRL)', f"{orders['order_value'].median():,.2f}", border=True,
               delta=pct_change(orders['order_value'].median(), previous(lambda o: o['order_value'].median())),
               help='Median of item prices per order, excluding freight.')
kpis[4].metric('Repeat customers', f'{(customers["customer_type"] == "Repeat").mean():.1%}', border=True,
               help='Share of customers with 2 or more orders in the selected period. '
                    'Short periods give lower values because there is less time to buy again.')

tab_overview, tab_basket, tab_retention, tab_segments, tab_geo = st.tabs(
    ['Overview', 'Basket size', 'Retention', 'Segments', 'Geography']
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    monthly = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).agg(
        orders=('order_id', 'count'), revenue=('order_value', 'sum')
    ).reset_index()
    monthly['month'] = monthly['order_purchase_timestamp'].dt.to_timestamp()
    peak = monthly.loc[monthly['orders'].idxmax()]
    st.markdown(f'The busiest month was **{peak["month"]:%B %Y}** with **{peak["orders"]:,} orders** '
                f'and **{brl(peak["revenue"])}** in revenue.')

    left, right = st.columns(2)
    with left:
        show(line_chart(monthly, 'month', 'orders', 'Orders per month', 'Orders', ',.0f',
                        [alt.Tooltip('month:T', title='Month', format='%B %Y'),
                         alt.Tooltip('orders:Q', title='Orders', format=',')]))
    with right:
        show(line_chart(monthly, 'month', 'revenue', 'Revenue per month (BRL)', 'Revenue (BRL)', '~s',
                        [alt.Tooltip('month:T', title='Month', format='%B %Y'),
                         alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f')]))

    top_categories_all = (items.groupby('product_category')
                          .agg(revenue=('price', 'sum'), items=('price', 'size'))
                          .sort_values('revenue', ascending=False).head(10).reset_index())
    top_categories_all['revenue_pct'] = top_categories_all['revenue'] / items['price'].sum()
    top_categories_all['color'] = MUTED
    top_categories_all.loc[:2, 'color'] = HIGHLIGHT
    show(bar_chart(top_categories_all, 'revenue', 'product_category', 'color',
                   'Top 10 product categories by revenue (BRL)', 'Revenue (BRL)', None, '.3~s',
                   [alt.Tooltip('product_category:N', title='Category'),
                    alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f'),
                    alt.Tooltip('revenue_pct:Q', title='Share of revenue', format='.1%'),
                    alt.Tooltip('items:Q', title='Items sold', format=',')],
                   horizontal=True, sort='-x', height=360))

# ---------------------------------------------------------------- Basket size
with tab_basket:
    basket = orders.groupby('basket_group', observed=False).agg(
        orders=('order_id', 'count'), median_value=('order_value', 'median')
    ).reset_index()
    basket['share'] = basket['orders'] / basket['orders'].sum()
    multi = orders['item_count'] >= 2
    single_median = orders.loc[~multi, 'order_value'].median()
    multi_median = orders.loc[multi, 'order_value'].median()

    if multi.any() and (~multi).any():
        st.markdown(
            f'Multi-item orders have a median value of **{brl(multi_median)}**, '
            f'**{multi_median / single_median - 1:.0%} higher** than single-item orders ({brl(single_median)}), '
            f'but they are only **{multi.mean():.1%} of orders**.'
        )
    left, right = st.columns(2)
    with left:
        basket['color'] = [HIGHLIGHT, MUTED, MUTED, MUTED]
        show(bar_chart(basket, 'basket_group', 'share', 'color', 'Share of orders by items per order',
                       'Items per order', 'Share of orders', '.1%',
                       [alt.Tooltip('basket_group:N', title='Items per order'),
                        alt.Tooltip('orders:Q', title='Orders', format=','),
                        alt.Tooltip('share:Q', title='Share of orders', format='.1%')], sort=BASKET_LABELS,
                       axis_fmt='.0%'))
    with right:
        basket['color'] = [MUTED, HIGHLIGHT, HIGHLIGHT, HIGHLIGHT]
        show(bar_chart(basket.dropna(subset=['median_value']), 'basket_group', 'median_value', 'color',
                       'Median order value by items per order (BRL)', 'Items per order',
                       'Median order value (BRL)', ',.0f',
                       [alt.Tooltip('basket_group:N', title='Items per order'),
                        alt.Tooltip('median_value:Q', title='Median order value (BRL)', format=',.2f'),
                        alt.Tooltip('orders:Q', title='Orders', format=',')], sort=BASKET_LABELS))

    multi_items = items[items['item_count'] >= 2]
    left, right = st.columns(2)
    with left:
        if multi_items.empty:
            st.info('No multi-item orders in the selected filters.')
        else:
            category_orders = (multi_items.groupby('product_category')['order_id'].nunique()
                               .sort_values(ascending=False).head(10).rename('orders').reset_index())
            category_orders['share'] = category_orders['orders'] / multi_items['order_id'].nunique()
            category_orders['color'] = MUTED
            category_orders.loc[:2, 'color'] = HIGHLIGHT
            show(bar_chart(category_orders, 'orders', 'product_category', 'color',
                           'Top categories in multi-item orders', 'Multi-item orders containing the category',
                           None, ',.0f',
                           [alt.Tooltip('product_category:N', title='Category'),
                            alt.Tooltip('orders:Q', title='Multi-item orders', format=','),
                            alt.Tooltip('share:Q', title='Share of multi-item orders', format='.1%')],
                           horizontal=True, sort='-x', height=360))
    with right:
        multi_monthly = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).agg(
            multi_share=('item_count', lambda x: (x >= 2).mean()), orders=('order_id', 'count')
        ).reset_index()
        multi_monthly['month'] = multi_monthly['order_purchase_timestamp'].dt.to_timestamp()
        show(line_chart(multi_monthly, 'month', 'multi_share', 'Share of multi-item orders per month',
                        'Multi-item orders', '.0%',
                        [alt.Tooltip('month:T', title='Month', format='%B %Y'),
                         alt.Tooltip('multi_share:Q', title='Multi-item share', format='.1%'),
                         alt.Tooltip('orders:Q', title='Orders', format=',')], height=360))

# ---------------------------------------------------------------- Retention
with tab_retention:
    summary = customers.groupby('customer_type').agg(
        customers=('orders', 'count'),
        revenue=('total_spending', 'sum'),
        median_total=('total_spending', 'median'),
        median_per_order=('spending_per_order', 'median'),
    ).reindex(['One-time', 'Repeat'])

    if summary['customers'].notna().all():
        repeat, one_time = summary.loc['Repeat'], summary.loc['One-time']
        st.markdown(
            f'**{repeat["customers"] / summary["customers"].sum():.1%} of customers** bought more than once. '
            f'They spent a median of **{brl(repeat["median_total"])}** in total, '
            f'**{repeat["median_total"] / one_time["median_total"] - 1:.0%} more** than one-time customers, '
            f'while spending per order is similar ({brl(repeat["median_per_order"])} vs. '
            f'{brl(one_time["median_per_order"])}).'
        )
    else:
        st.info('The selected filters contain only one customer type, so the groups cannot be compared.')

    summary = summary.dropna(subset=['customers']).reset_index()
    summary['Share of customers'] = summary['customers'] / summary['customers'].sum()
    summary['Share of spending'] = summary['revenue'] / summary['revenue'].sum()
    left, right = st.columns(2)
    with left:
        shares = summary.melt(id_vars='customer_type', value_vars=['Share of customers', 'Share of spending'],
                              var_name='metric', value_name='share')
        show(grouped_bar_chart(shares, 'metric', 'share', 'customer_type', ['One-time', 'Repeat'],
                               [MUTED, HIGHLIGHT], 'Share of customers vs. share of spending', 'Share',
                               '.1%', [alt.Tooltip('customer_type:N', title='Customer type'),
                                       alt.Tooltip('metric:N', title='Metric'),
                                       alt.Tooltip('share:Q', title='Share', format='.1%')], axis_fmt='.0%'))
    with right:
        spending = summary.rename(columns={'median_total': 'Total per customer',
                                           'median_per_order': 'Per order'}).melt(
            id_vars='customer_type', value_vars=['Total per customer', 'Per order'],
            var_name='metric', value_name='brl')
        show(grouped_bar_chart(spending, 'metric', 'brl', 'customer_type', ['One-time', 'Repeat'],
                               [MUTED, HIGHLIGHT], 'Median spending (BRL)', 'Median spending (BRL)',
                               ',.0f', [alt.Tooltip('customer_type:N', title='Customer type'),
                                        alt.Tooltip('metric:N', title='Metric'),
                                        alt.Tooltip('brl:Q', title='Median (BRL)', format=',.2f')]))

# ---------------------------------------------------------------- Segments
with tab_segments:
    if len(customers) < 4:
        st.info('At least 4 customers are needed to build RFM segments. Widen the filters.')
    else:
        rfm = build_rfm(start, end, selected_states)
        segments = rfm.groupby('segment').agg(
            customers=('recency', 'count'),
            median_recency=('recency', 'median'),
            median_monetary=('monetary', 'median'),
            revenue=('monetary', 'sum'),
        ).reindex(SEGMENT_ORDER).fillna(0).reset_index()
        segments['share_customers'] = segments['customers'] / segments['customers'].sum()
        segments['share_revenue'] = segments['revenue'] / segments['revenue'].sum()

        top = segments.sort_values('revenue', ascending=False).iloc[0]
        st.markdown(f'**{top["segment"]}** customers generate the largest share of spending: '
                    f'**{top["share_revenue"]:.1%}** from **{top["share_customers"]:.1%}** of customers. '
                    f'Recency is measured in days from {end + pd.Timedelta(days=1):%d %b %Y}.')

        shares = segments.rename(columns={'share_customers': 'Share of customers',
                                          'share_revenue': 'Share of spending'}).melt(
            id_vars='segment', value_vars=['Share of customers', 'Share of spending'],
            var_name='metric', value_name='share')
        show(grouped_bar_chart(shares, 'segment', 'share', 'metric', ['Share of customers', 'Share of spending'],
                               [HIGHLIGHT, SECONDARY], 'Share of customers vs. share of spending by segment',
                               'Share', '.1%', [alt.Tooltip('segment:N', title='Segment'),
                                                alt.Tooltip('metric:N', title='Metric'),
                                                alt.Tooltip('share:Q', title='Share', format='.1%')],
                               horizontal=True, sort=SEGMENT_ORDER, height=420, axis_fmt='.0%'))

        st.dataframe(
            segments.merge(SEGMENTS, on='segment')[
                ['segment', 'definition', 'customers', 'median_recency', 'median_monetary', 'suggested_action']
            ],
            hide_index=True, width='stretch',
            column_config={
                'segment': 'Segment',
                'definition': 'Definition',
                'customers': st.column_config.NumberColumn('Customers', format='localized'),
                'median_recency': st.column_config.NumberColumn('Median recency (days)', format='%d'),
                'median_monetary': st.column_config.NumberColumn('Median spending (BRL)', format='%.2f'),
                'suggested_action': 'Suggested action',
            },
        )

# ---------------------------------------------------------------- Geography
with tab_geo:
    state = orders.assign(freight_ratio=orders['freight_value'] / orders['order_value']).groupby(
        'customer_state').agg(
        orders=('order_id', 'count'), revenue=('order_value', 'sum'), freight_ratio=('freight_ratio', 'median')
    ).reset_index()
    state['state_name'] = state['customer_state'].map(STATE_NAMES)
    state['revenue_share'] = state['revenue'] / state['revenue'].sum()
    top_state = state.sort_values('revenue', ascending=False).iloc[0]
    st.markdown(f'**{top_state["state_name"]}** generates **{top_state["revenue_share"]:.1%}** of revenue. '
                f'The median freight cost ranges from **{state["freight_ratio"].min():.1%}** to '
                f'**{state["freight_ratio"].max():.1%}** of order value across states.')

    points = (orders.dropna(subset=['geolocation_lat'])
              .groupby(['geolocation_lat', 'geolocation_lng', 'customer_city', 'customer_state'])
              .agg(orders=('order_id', 'count'), revenue=('order_value', 'sum')).reset_index())
    points['radius'] = 1500 + np.sqrt(points['orders']) * 1500
    points['revenue_text'] = points['revenue'].map(brl)
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        layers=[pdk.Layer(
            'ScatterplotLayer', data=points, get_position='[geolocation_lng, geolocation_lat]',
            get_radius='radius', get_fill_color=[42, 120, 214, 90], pickable=True,
            radius_min_pixels=1.5, radius_max_pixels=30,
        )],
        initial_view_state=pdk.ViewState(latitude=-16.5, longitude=-53.0, zoom=3.0),
        tooltip={'text': '{customer_city} ({customer_state})\n{orders} orders · {revenue_text}'},
    ), height=480)
    st.caption('Each circle is a customer zip code area; larger circles have more orders, and darker areas '
               'have more overlapping circles. Hover for details.')

    left, right = st.columns(2)
    with left:
        top_revenue = state.sort_values('revenue', ascending=False).head(10).reset_index(drop=True)
        top_revenue['color'] = MUTED
        top_revenue.loc[:2, 'color'] = HIGHLIGHT
        show(bar_chart(top_revenue, 'revenue_share', 'state_name', 'color', 'Top 10 states by revenue share',
                       'Share of revenue', None, '.1%',
                       [alt.Tooltip('state_name:N', title='State'),
                        alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f'),
                        alt.Tooltip('revenue_share:Q', title='Share of revenue', format='.1%'),
                        alt.Tooltip('orders:Q', title='Orders', format=',')],
                       horizontal=True, sort='-x', height=360, axis_fmt='.0%'))
    with right:
        top_freight = state.sort_values('freight_ratio', ascending=False).head(10).reset_index(drop=True)
        top_freight['color'] = MUTED
        top_freight.loc[:2, 'color'] = SECONDARY
        show(bar_chart(top_freight, 'freight_ratio', 'state_name', 'color',
                       'Top 10 states by median freight cost (% of order value)', 'Median freight / order value',
                       None, '.0%',
                       [alt.Tooltip('state_name:N', title='State'),
                        alt.Tooltip('freight_ratio:Q', title='Median freight / order value', format='.1%'),
                        alt.Tooltip('orders:Q', title='Orders', format=',')],
                       horizontal=True, sort='-x', height=360))
