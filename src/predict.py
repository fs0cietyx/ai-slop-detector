import sys

from src.core.config import get_logger
from src.core.engine import InferenceEngine

logger = get_logger(__name__)


def main() -> None:
    """
    Standard CLI Entrypoint for AI slop detection.

    Demonstrates architectural modularity and defensive programming.
    """
    if len(sys.argv) < 2:
        print("\n🚀 AI Slop Detector v1.0")
        print('Usage: python -m src.predict "Your text here..."\n')
        sys.exit(1)

    payload = sys.argv[1]

    try:
        # Lazy initialization of the engine
        engine = InferenceEngine()
        label, confidence = engine.predict(payload)

        # Standardized output for analysis reports
        print("\n" + "═" * 50)
        print("   🔍 ARCHITECTURAL ANALYSIS REPORT")
        print("═" * 50)
        print(f" CLASSIFICATION : {label}")
        print(f" CONFIDENCE     : {confidence:.4%}")
        print(" STATUS         : VERIFIED_INFERENCE")
        print("═" * 50 + "\n")

    except KeyboardInterrupt:
        logger.info("Interrupt received. Shutdown complete.")
        sys.exit(0)
    except Exception as e:
        # Secure error handling: log the detail, print the generic failure
        logger.critical(f"Panic in CLI execution: {str(e)}")
        print("\n[!] CRITICAL: System-wide inference failure. Check logs.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
