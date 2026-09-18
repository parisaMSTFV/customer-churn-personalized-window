from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the synthetic customer and transaction generator."""

    n_customers: int = 4_000
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-30"
    seed: int = 42


@dataclass(frozen=True)
class DatasetConfig:
    """Rules used to estimate cadence and create actionable snapshots."""

    min_successful_orders: int = 4
    recent_gaps: int = 6
    score_frequency_days: int = 14
    alert_fraction: float = 0.55
    dispersion_buffer: float = 0.75
    min_window_days: int = 7
    max_window_days: int = 140
    feature_lookback_days: int = 180


@dataclass(frozen=True)
class InputConfig:
    """Resource and shape limits for externally supplied transaction files."""

    min_rows: int = 100
    max_rows: int = 5_000_000
    max_file_bytes: int = 250 * 1024 * 1024
    min_customers: int = 10
    min_history_days: int = 180
