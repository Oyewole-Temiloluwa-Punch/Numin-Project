"""
Utility modules for the Stock Market Dashboard
"""

# Conditional import for AWS utils (requires boto3)
try:
    from .aws_utils import save_cache_to_s3, load_cache_from_s3, cache_exists_in_s3
    _aws_available = True
except ImportError:
    # AWS utils not available (boto3 not installed)
    _aws_available = False
    def save_cache_to_s3(*args, **kwargs):
        raise ImportError("AWS utils not available - boto3 not installed")
    def load_cache_from_s3(*args, **kwargs):
        raise ImportError("AWS utils not available - boto3 not installed")
    def cache_exists_in_s3(*args, **kwargs):
        raise ImportError("AWS utils not available - boto3 not installed")
from .pattern_utils import (
    is_repetitive_pattern,
    generate_pattern_at_index,
    find_pattern_occurrences,
    generate_cache,
    analyze_current_day_pattern_outcomes
)
from .chart_utils import (
    build_projection_chart,
    compute_stepwise_highlights,
    render_highlights
)
from .data_utils import load_data
from .api_utils import fetch_cycle_pattern_data, fetch_secondary_patterns
from .chart_builder import (
    build_base_candlestick_chart,
    update_chart_ranges,
    add_vertical_marker,
    add_projection_line,
    add_aggregate_line,
    add_regression_channel
)
from .clustering_utils import (
    get_adaptive_cluster_mean,
    cluster_projection_points,
    calculate_mean_aggregates,
    prepare_aggregate_line_data
)
from .projection_utils import (
    get_selected_dates,
    get_pattern_position_and_close,
    process_pattern_projections,
    add_projection_overlays,
    add_aggregate_lines
)
from .ui_utils import (
    initialize_session_state,
    create_pattern_filter,
    create_chart_controls,
    create_pattern_analysis_ui,
    display_pattern_analysis_results,
    handle_cache_generation,
    show_analysis_prompt,
    clear_analysis_cache
)
from .technical_utils import (
    calculate_rsi,
    group_days,
    format_group_display,
    get_group_summary_stats,
    calculate_regression_channel
)

__all__ = [
    'save_cache_to_s3',
    'load_cache_from_s3', 
    'cache_exists_in_s3',
    'is_repetitive_pattern',
    'generate_pattern_at_index',
    'find_pattern_occurrences',
    'generate_cache',
    'analyze_current_day_pattern_outcomes',
    'build_projection_chart',
    'compute_stepwise_highlights',
    'render_highlights',
    'load_data',
    'fetch_cycle_pattern_data',
    'fetch_secondary_patterns',
    'build_base_candlestick_chart',
    'update_chart_ranges',
    'add_vertical_marker',
    'add_projection_line',
    'add_aggregate_line',
    'add_regression_channel',
    'get_adaptive_cluster_mean',
    'cluster_projection_points',
    'calculate_mean_aggregates',
    'prepare_aggregate_line_data',
    'get_selected_dates',
    'get_pattern_position_and_close',
    'process_pattern_projections',
    'add_projection_overlays',
    'add_aggregate_lines',
    'initialize_session_state',
    'create_pattern_filter',
    'create_chart_controls',
    'create_pattern_analysis_ui',
    'display_pattern_analysis_results',
    'handle_cache_generation',
    'show_analysis_prompt',
    'clear_analysis_cache',
    'calculate_rsi',
    'group_days',
    'format_group_display',
    'get_group_summary_stats',
    'calculate_regression_channel'
]
