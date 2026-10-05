"""Altair chart builders. Every chart starts at zero, labels its values, and shows a tooltip."""
import altair as alt

from theme import HIGHLIGHT, MUTED


def highlight_top(df, n=3, color=HIGHLIGHT):
    """Add a `color` column: the first `n` rows in `color`, the rest in gray."""
    return df.assign(color=[color if i < n else MUTED for i in range(len(df))])


def _value_labels(base, value, label_fmt, horizontal):
    return base.mark_text(
        align='left' if horizontal else 'center',
        baseline='middle' if horizontal else 'bottom',
        dx=4 if horizontal else 0,
        dy=0 if horizontal else -4,
        fontSize=12,
    ).encode(text=alt.Text(f'{value}:Q', format=label_fmt))


def bar_chart(df, *, category, value, title, label_fmt, tooltip, value_title=None, category_title=None,
              axis_fmt=None, horizontal=False, sort=None, height=320):
    """Single-series bars colored by the `color` column (see `highlight_top`)."""
    category_channel, value_channel = (alt.Y, alt.X) if horizontal else (alt.X, alt.Y)
    base = alt.Chart(df).encode(
        category_channel(f'{category}:N', sort=sort, title=category_title,
                         axis=alt.Axis(labelAngle=0, labelLimit=220)),
        value_channel(f'{value}:Q', title=value_title, scale=alt.Scale(domainMin=0, nice=True),
                      axis=alt.Axis(format=axis_fmt or label_fmt)),
        tooltip=tooltip,
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=alt.Color('color:N', scale=None, legend=None))
    return (bars + _value_labels(base, value, label_fmt, horizontal)).properties(title=title, height=height)


def grouped_bar_chart(df, *, category, value, group, groups, colors, title, label_fmt, tooltip,
                      value_title=None, axis_fmt=None, horizontal=False, sort=None, height=320):
    """Bars for each `group` side by side within each `category`, with a legend on top."""
    category_channel, value_channel = (alt.Y, alt.X) if horizontal else (alt.X, alt.Y)
    offset_channel = alt.YOffset if horizontal else alt.XOffset
    base = alt.Chart(df).encode(
        category_channel(f'{category}:N', sort=sort, title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)),
        value_channel(f'{value}:Q', title=value_title, scale=alt.Scale(domainMin=0, nice=True),
                      axis=alt.Axis(format=axis_fmt or label_fmt)),
        offset_channel(f'{group}:N', sort=groups),
        tooltip=tooltip,
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color(f'{group}:N', sort=groups, scale=alt.Scale(domain=groups, range=colors),
                        legend=alt.Legend(title=None, orient='top'))
    )
    return (bars + _value_labels(base, value, label_fmt, horizontal)).properties(title=title, height=height)


def line_chart(df, *, x, y, title, y_title, y_fmt, tooltip, height=300):
    """Monthly line with points and a hover rule that shows the tooltip of the nearest month."""
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
