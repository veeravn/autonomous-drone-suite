# src/config/jetson.py

JETSON_CONFIG = {
    "enable_trt": True,
    "max_cpu_threads": 4,
    "gpu_mem_limit": None,
    "power_mode": 0,          # nvpmodel -m 0
    "auto_jetson_clocks": True,
}
