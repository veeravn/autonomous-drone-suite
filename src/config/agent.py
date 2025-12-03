# src/config/agent.py

AGENT_CONFIG = {
    # Main loop control
    "main_loop_hz": 15.0,

    # Capture / NBV behavior
    "capture_period_s": 3.0,      # seconds between capture attempts
    "max_yaw_delta_deg": 45.0,    # clamp per-step yaw change

    # Defaults (overridable by CLI / main)
    # NBV weights
    "w_novelty": 2.0,
    "w_turn_cost": 1.0,
    "w_energy_cost": 1.0,
    "w_geo_penalty": 3.0,
    "semantic_nbv_default": True,
    "min_rel_alt_m_default": 1.0,
    "max_rel_alt_m_default": 15.0,
    "rtl_battery_pct_default": 20.0,

    # Offboard / yaw limits
    "max_yaw_rate_deg_s": 45.0,
    "max_yaw_delta_deg": 90.0,
}
