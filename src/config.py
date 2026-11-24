"""
Pipeline configuration containing centralized thresholds, domain guards, and metadata mappings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Default File Paths
DEFAULT_RAW_FILE = RAW_DATA_DIR / "2026-10-27_EV_Analysis_Data.csv"
CLEANING_REPORT_PATH = OUTPUTS_DIR / "cleaning_report.json"
PREPROCESSOR_JOB_PATH = PROCESSED_DATA_DIR / "preprocessor.joblib"

# Pipeline Thresholds
MISSING_COL_THRESHOLD: float = 0.60
CORR_THRESHOLD: float = 0.90
IQR_MULTIPLIER: float = 1.5
SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
RANDOM_STATE: int = 42

# EV Telemetry Physical Domain Guards
DOMAIN_GUARDS: Dict[str, Dict[str, Any]] = {
    # Speed: EV physical limits (0 to 150 km/h)
    "speed": {"min": 0.0, "max": 150.0},
    "speed_kmh": {"min": 0.0, "max": 150.0},
    "speed_raw": {"min": 0.0, "max": 150.0},
    "speed_valid": {"min": 0.0, "max": 150.0},
    "speed_alt3": {"min": 0.0, "max": 150.0},
    
    # Cell Delta: voltage difference between max and min cell (millivolts: valid range 0 to 500 mV)
    "cell_delta_mv": {"min": 0.0, "max": 500.0},
    "cell_delta": {"min": 0.0, "max": 500.0},
    
    # State of Charge (SOC): percentage [0, 100]
    "soc": {"min": 0.0, "max": 100.0},
    "soh": {"min": 0.0, "max": 100.0},
    
    # Cell Voltages in mV (Li-ion EV safe operating boundaries ~2000mV to 4500mV)
    "min_cell_mv": {"min": 2000.0, "max": 4600.0},
    "max_cell_mv": {"min": 2000.0, "max": 4600.0},
    "avg_cell_mv": {"min": 2000.0, "max": 4600.0},
    
    # Pack Voltage (positive values, typically 200V - 500V)
    "pack_voltage": {"min": 100.0, "max": 600.0},
    
    # Temperatures (°C)
    "batt_temp": {"min": -30.0, "max": 80.0},
    "cell_temp_min": {"min": -30.0, "max": 80.0},
    "cell_temp_max": {"min": -30.0, "max": 80.0},
    "inverter_temp": {"min": -30.0, "max": 120.0},
    "motor_winding_temp": {"min": -30.0, "max": 160.0},
}

# Metadata, Raw Flags, and Identifier Columns to automatically exclude from model features
METADATA_COLUMNS: List[str] = [
    "source_file",
    "ts_utc_original",
    "ts",
    "timestamp",
    "ts_utc",
    "ist_time",
    "time",
    "datetime",
    "anomaly_flags",
    "__last_motion_at",
    "__sample_at",
    "__current_sign_verified",
]

# Supported scalers and split modes
VALID_SCALERS = ["standard", "minmax", "robust"]
VALID_SPLIT_MODES = ["chronological", "random"]


@dataclass
class PipelineConfig:
    """Configurable parameters for running the pipeline."""
    input_path: str = str(DEFAULT_RAW_FILE)
    target_col: str = "power_kw"
    split_mode: str = "chronological"
    scaler_type: str = "standard"
    drop_unconfirmed: bool = False
    outlier_strategy: str = "cap"  # 'cap' or 'remove'
    missing_col_threshold: float = MISSING_COL_THRESHOLD
    corr_threshold: float = CORR_THRESHOLD
    iqr_multiplier: float = IQR_MULTIPLIER
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS
    k_best_features: int = 0  # 0 means retain all selected features
    random_state: int = RANDOM_STATE
    output_dir: str = str(PROCESSED_DATA_DIR)
    plots_dir: str = str(PLOTS_DIR)
    report_path: str = str(CLEANING_REPORT_PATH)
    extra_metadata_cols: List[str] = field(default_factory=list)
