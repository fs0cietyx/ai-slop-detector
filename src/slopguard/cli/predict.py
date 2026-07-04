import sys
import time
from typing import Final

from slopguard.core.config import logger
from slopguard.core.engine import InferenceEngine

# --- Cinematic UI Components ---
BANNER: Final[str] = r"""
    ___    ____   _____ __    ____  ____ 
   /   |  /  _/  / ___// /   / __ \/ __ \
  / /| |  / /    \__ \/ /   / / / / /_/ /
 / ___ |_/ /    ___/ / /___/ /_/ / ____/ 
/_/  |_/___/   /____/_____/\____/_/      
                                         
    DETECTOR | ENTERPRISE EDITION v1.0
"""

HR: Final[str] = " " + "─" * 45
GLOW_DOT: Final[str] = "●"

def print_ui_header() -> None:
    """Renders the cinematic high-fidelity banner."""
    print("\033[1;36m" + BANNER + "\033[0m")
    print(f"\033[1;30m{HR}\033[0m")

def run_cli() -> None:
    """
    Enterprise CLI Entrypoint for AI Slop Detection.
    
    Implements high-fidelity visual reporting and hardware acceleration telemetry.
    """
    if len(sys.argv) < 2:
        print_ui_header()
        print("\n \033[1;33m[!] USAGE:\033[0m")
        print(" python -m src.predict \"Your payload here...\"\n")
        sys.exit(0)

    payload = sys.argv[1]
    print_ui_header()
    
    try:
        # [Visual Flow] Simulated system check for cinematic effect
        print(f" \033[1;34m{GLOW_DOT}\033[0m INITIALIZING_INFERENCE_ENGINE...")
        
        start_time = time.time()
        
        # Singleton engine initialization (hardware detection happens here)
        engine = InferenceEngine()
        
        print(f" \033[1;34m{GLOW_DOT}\033[0m ANALYZING_TEXT_VECTORS...")
        label, confidence = engine.predict(payload)
        
        latency_ms = (time.time() - start_time) * 1000

        # --- Visual Inference Report ---
        status_color = "\033[1;32m" if label == "HUMAN-WRITTEN" else "\033[1;31m"
        conf_color = "\033[1;32m" if confidence > 0.8 else "\033[1;33m"

        print(f"\033[1;30m{HR}\033[0m")
        print(f"  \033[1;37mIDENTIFICATION :\033[0m {status_color}{label}\033[0m")
        print(f"  \033[1;37mCONFIDENCE     :\033[0m {conf_color}{confidence:.2%}\033[0m")
        print(f"  \033[1;37mCOMPUTE_TIME   :\033[0m \033[1;34m{latency_ms:.2f}ms\033[0m")
        print(f"  \033[1;37mHW_CONTEXT     :\033[0m \033[1;35m{engine._device.type.upper()}\033[0m")
        print(f"\033[1;30m{HR}\033[0m")
        
        # Bottom decorative bar
        print(" \033[1;32m[✓]\033[0m INFERENCE_AUDIT_COMPLETE\n")

    except KeyboardInterrupt:
        logger.info("CLI_SHUTDOWN: Operation cancelled by operator.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"CLI_FATAL_ERROR: {str(e)}")
        print("\n \033[1;31m[!] CRITICAL_FAILURE: System panic. Check audit logs.\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
