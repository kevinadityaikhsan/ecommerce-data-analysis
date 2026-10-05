"""Color tokens shared by the charts and the stylesheet, matching the notebook palette."""

HIGHLIGHT = '#2a78d6'
SECONDARY = '#eb6834'
MUTED = '#b4b4b0'
POSITIVE = '#1a9e5c'
NEGATIVE = '#d6453d'


def css_variables():
    """Expose the tokens to style.css as CSS custom properties."""
    return (f':root {{ --accent: {HIGHLIGHT}; --secondary: {SECONDARY}; --muted: {MUTED}; '
            f'--positive: {POSITIVE}; --negative: {NEGATIVE}; }}')
