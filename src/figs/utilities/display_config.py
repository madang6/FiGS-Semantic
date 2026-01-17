"""
Global configuration for figure display (matplotlib AND plotly).

This module provides a global flag to control whether figures are displayed.
Used to prevent memory leaks during batch processing like RL training.
"""

# Global flag to suppress figure display during batch processing (e.g., RL training)
# Set to False to disable plt.show() and fig.show() calls and prevent memory leaks
_ENABLE_FIGURE_DISPLAY = True


def set_figure_display(enable: bool):
    """
    Control whether figures (matplotlib AND plotly) are displayed.
    
    Args:
        enable: If True, figures will display normally.
                If False, figure display is suppressed (useful during RL training).
    """
    global _ENABLE_FIGURE_DISPLAY
    _ENABLE_FIGURE_DISPLAY = enable
    
    # Also configure Plotly renderer
    try:
        import plotly.io as pio
        if enable:
            pio.renderers.default = 'browser'
        else:
            pio.renderers.default = None  # Suppress all output
    except ImportError:
        pass  # Plotly not installed


def get_figure_display() -> bool:
    """
    Get current figure display setting.
    
    Returns:
        bool: True if figures should be displayed, False otherwise.
    """
    return _ENABLE_FIGURE_DISPLAY
