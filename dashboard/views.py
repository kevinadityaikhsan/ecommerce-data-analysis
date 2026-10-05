"""Tab views that arrange components and charts built from the aggregates in analytics.py."""
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import analytics
from charts import bar_chart, grouped_bar_chart, highlight_top, line_chart
from components import brl, chart_card, insight, map_card
from theme import HIGHLIGHT, MUTED, SECONDARY

MONTH_TOOLTIP = alt.Tooltip('month:T', title='Month', format='%B %Y')
CUSTOMER_TYPES = ['One-time', 'Repeat']


def render_overview(items, orders):
    monthly = analytics.monthly_summary(orders)
    peak = monthly.loc[monthly['orders'].idxmax()]
    insight(f'The busiest month was **{peak["month"]:%B %Y}** with **{peak["orders"]:,} orders** '
            f'and **{brl(peak["revenue"])}** in revenue.')

    left, right = st.columns(2)
    with left:
        chart_card(line_chart(
            monthly, x='month', y='orders', title='Orders per month', y_title='Orders', y_fmt=',.0f',
            tooltip=[MONTH_TOOLTIP, alt.Tooltip('orders:Q', title='Orders', format=',')],
        ))
    with right:
        chart_card(line_chart(
            monthly, x='month', y='revenue', title='Revenue per month (BRL)', y_title='Revenue (BRL)', y_fmt='~s',
            tooltip=[MONTH_TOOLTIP, alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f')],
        ))

    chart_card(bar_chart(
        highlight_top(analytics.top_categories_by_revenue(items)),
        category='product_category', value='revenue', title='Top 10 product categories by revenue (BRL)',
        value_title='Revenue (BRL)', label_fmt='.3~s', horizontal=True, sort='-x', height=360,
        tooltip=[alt.Tooltip('product_category:N', title='Category'),
                 alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f'),
                 alt.Tooltip('revenue_share:Q', title='Share of revenue', format='.1%'),
                 alt.Tooltip('items:Q', title='Items sold', format=',')],
    ))


def render_basket(items, orders):
    basket = analytics.basket_summary(orders)
    comparison = analytics.single_vs_multi(orders)
    if not (pd.isna(comparison['single_median']) or pd.isna(comparison['multi_median'])):
        uplift = comparison['multi_median'] / comparison['single_median'] - 1
        insight(f'Multi-item orders have a median value of **{brl(comparison["multi_median"])}**, '
                f'**{uplift:.0%} higher** than single-item orders ({brl(comparison["single_median"])}), '
                f'but they are only **{comparison["multi_share"]:.1%} of orders**.')

    basket_tooltip = [alt.Tooltip('basket_group:N', title='Items per order'),
                      alt.Tooltip('orders:Q', title='Orders', format=',')]
    left, right = st.columns(2)
    with left:
        chart_card(bar_chart(
            basket.assign(color=[HIGHLIGHT, MUTED, MUTED, MUTED]),
            category='basket_group', value='share', title='Share of orders by items per order',
            category_title='Items per order', value_title='Share of orders', label_fmt='.1%', axis_fmt='.0%',
            sort=analytics.BASKET_LABELS,
            tooltip=basket_tooltip + [alt.Tooltip('share:Q', title='Share of orders', format='.1%')],
        ))
    with right:
        chart_card(bar_chart(
            basket.assign(color=[MUTED, HIGHLIGHT, HIGHLIGHT, HIGHLIGHT]).dropna(subset=['median_value']),
            category='basket_group', value='median_value', title='Median order value by items per order (BRL)',
            category_title='Items per order', value_title='Median order value (BRL)', label_fmt=',.0f',
            sort=analytics.BASKET_LABELS,
            tooltip=basket_tooltip + [alt.Tooltip('median_value:Q', title='Median order value (BRL)',
                                                  format=',.2f')],
        ))

    left, right = st.columns(2)
    with left:
        categories = analytics.top_categories_in_multi_item_orders(items)
        if categories.empty:
            st.info('No multi-item orders in the selected filters.')
        else:
            chart_card(bar_chart(
                highlight_top(categories),
                category='product_category', value='orders', title='Top categories in multi-item orders',
                value_title='Multi-item orders containing the category', label_fmt=',.0f',
                horizontal=True, sort='-x', height=360,
                tooltip=[alt.Tooltip('product_category:N', title='Category'),
                         alt.Tooltip('orders:Q', title='Multi-item orders', format=','),
                         alt.Tooltip('share:Q', title='Share of multi-item orders', format='.1%')],
            ))
    with right:
        chart_card(line_chart(
            analytics.monthly_summary(orders), x='month', y='multi_share',
            title='Share of multi-item orders per month', y_title='Multi-item orders', y_fmt='.0%', height=360,
            tooltip=[MONTH_TOOLTIP, alt.Tooltip('multi_share:Q', title='Multi-item share', format='.1%'),
                     alt.Tooltip('orders:Q', title='Orders', format=',')],
        ))


def render_retention(customers):
    summary = analytics.customer_type_summary(customers)
    if summary['customers'].notna().all():
        repeat, one_time = summary.loc['Repeat'], summary.loc['One-time']
        insight(f'**{repeat["customer_share"]:.1%} of customers** bought more than once. '
                f'They spent a median of **{brl(repeat["median_total"])}** in total, '
                f'**{repeat["median_total"] / one_time["median_total"] - 1:.0%} more** than one-time customers, '
                f'while spending per order is similar ({brl(repeat["median_per_order"])} vs. '
                f'{brl(one_time["median_per_order"])}).')
    else:
        st.info('The selected filters contain only one customer type, so the groups cannot be compared.')

    summary = summary.dropna(subset=['customers']).rename_axis('customer_type').reset_index()
    group_tooltip = [alt.Tooltip('customer_type:N', title='Customer type'), alt.Tooltip('metric:N', title='Metric')]
    left, right = st.columns(2)
    with left:
        shares = summary.rename(columns={'customer_share': 'Share of customers',
                                         'spending_share': 'Share of spending'}).melt(
            id_vars='customer_type', value_vars=['Share of customers', 'Share of spending'],
            var_name='metric', value_name='share')
        chart_card(grouped_bar_chart(
            shares, category='metric', value='share', group='customer_type', groups=CUSTOMER_TYPES,
            colors=[MUTED, HIGHLIGHT], title='Share of customers vs. share of spending', value_title='Share',
            label_fmt='.1%', axis_fmt='.0%',
            tooltip=group_tooltip + [alt.Tooltip('share:Q', title='Share', format='.1%')],
        ))
    with right:
        spending = summary.rename(columns={'median_total': 'Total per customer',
                                           'median_per_order': 'Per order'}).melt(
            id_vars='customer_type', value_vars=['Total per customer', 'Per order'],
            var_name='metric', value_name='brl')
        chart_card(grouped_bar_chart(
            spending, category='metric', value='brl', group='customer_type', groups=CUSTOMER_TYPES,
            colors=[MUTED, HIGHLIGHT], title='Median spending (BRL)', value_title='Median spending (BRL)',
            label_fmt=',.0f',
            tooltip=group_tooltip + [alt.Tooltip('brl:Q', title='Median (BRL)', format=',.2f')],
        ))


def render_segments(start, end, states, customer_count):
    if customer_count < 4:
        st.info('At least 4 customers are needed to build RFM segments. Widen the filters.')
        return

    segments = analytics.segment_summary(analytics.build_rfm(start, end, states))
    top = segments.sort_values('spending', ascending=False).iloc[0]
    insight(f'**{top["segment"]}** customers generate the largest share of spending: '
            f'**{top["spending_share"]:.1%}** from **{top["customer_share"]:.1%}** of customers. '
            f'Recency is measured in days from {end + pd.Timedelta(days=1):%d %b %Y}.')

    shares = segments.rename(columns={'customer_share': 'Share of customers',
                                      'spending_share': 'Share of spending'}).melt(
        id_vars='segment', value_vars=['Share of customers', 'Share of spending'],
        var_name='metric', value_name='share')
    chart_card(grouped_bar_chart(
        shares, category='segment', value='share', group='metric',
        groups=['Share of customers', 'Share of spending'], colors=[HIGHLIGHT, SECONDARY],
        title='Share of customers vs. share of spending by segment', value_title='Share',
        label_fmt='.1%', axis_fmt='.0%', horizontal=True, sort=analytics.SEGMENT_ORDER, height=420,
        tooltip=[alt.Tooltip('segment:N', title='Segment'), alt.Tooltip('metric:N', title='Metric'),
                 alt.Tooltip('share:Q', title='Share', format='.1%')],
    ))

    st.dataframe(
        segments[['segment', 'definition', 'customers', 'median_recency', 'median_monetary', 'suggested_action']],
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


def render_geography(orders):
    state = analytics.state_summary(orders)
    top_state = state.sort_values('revenue', ascending=False).iloc[0]
    insight(f'**{top_state["state_name"]}** generates **{top_state["revenue_share"]:.1%}** of revenue. '
            f'The median freight cost ranges from **{state["freight_ratio"].min():.1%}** to '
            f'**{state["freight_ratio"].max():.1%}** of order value across states.')

    points = analytics.map_points(orders)
    points['radius'] = 1500 + np.sqrt(points['orders']) * 1500
    points['revenue_text'] = points['revenue'].map(brl)
    map_card(
        pdk.Deck(
            map_style=None,  # Use Streamlit's theme-aware basemap.
            layers=[pdk.Layer(
                'ScatterplotLayer', data=points, get_position='[geolocation_lng, geolocation_lat]',
                get_radius='radius', get_fill_color=[42, 120, 214, 90], pickable=True,
                radius_min_pixels=1.5, radius_max_pixels=30,
            )],
            initial_view_state=pdk.ViewState(latitude=-16.5, longitude=-53.0, zoom=3.0),
            tooltip={'text': '{customer_city} ({customer_state})\n{orders} orders · {revenue_text}'},
        ),
        caption='Each circle is a customer zip code area; larger circles have more orders, and darker areas '
                'have more overlapping circles. Hover for details.',
    )

    state_tooltip = [alt.Tooltip('state_name:N', title='State'), alt.Tooltip('orders:Q', title='Orders', format=',')]
    left, right = st.columns(2)
    with left:
        chart_card(bar_chart(
            highlight_top(state.sort_values('revenue', ascending=False).head(10)),
            category='state_name', value='revenue_share', title='Top 10 states by revenue share',
            value_title='Share of revenue', label_fmt='.1%', axis_fmt='.0%', horizontal=True, sort='-x',
            height=360,
            tooltip=state_tooltip + [alt.Tooltip('revenue:Q', title='Revenue (BRL)', format=',.2f'),
                                     alt.Tooltip('revenue_share:Q', title='Share of revenue', format='.1%')],
        ))
    with right:
        chart_card(bar_chart(
            highlight_top(state.sort_values('freight_ratio', ascending=False).head(10), color=SECONDARY),
            category='state_name', value='freight_ratio',
            title='Top 10 states by median freight cost (% of order value)',
            value_title='Median freight / order value', label_fmt='.0%', horizontal=True, sort='-x', height=360,
            tooltip=state_tooltip + [alt.Tooltip('freight_ratio:Q', title='Median freight / order value',
                                                 format='.1%')],
        ))
