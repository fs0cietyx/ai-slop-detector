import typer

app = typer.Typer(help="SlopGuard ML Enterprise Suite CLI")


@app.command()
def predict(text: str) -> None:
    """Run real-time inference on provided text."""
    # We delay imports to keep CLI responsive
    import time

    from slopguard.cli.predict import BANNER
    from slopguard.core.config import logger
    from slopguard.core.engine import InferenceEngine

    logger.info(BANNER)
    logger.info("● INITIALIZING_INFERENCE_ENGINE...")
    start_time = time.time()

    engine = InferenceEngine()

    logger.info("● ANALYZING_TEXT_VECTORS...")
    label, confidence = engine.predict(text)

    compute_time = (time.time() - start_time) * 1000

    logger.info("─────────────────────────────────────────────")
    logger.info(f" IDENTIFICATION : {label}")
    logger.info(f" CONFIDENCE     : {confidence * 100:.2f}%")
    logger.info(f" COMPUTE_TIME   : {compute_time:.2f}ms")
    logger.info("─────────────────────────────────────────────")
    logger.info("[✓] INFERENCE_AUDIT_COMPLETE")


@app.command()
def train() -> None:
    """Trigger the adversarial training pipeline."""
    from slopguard.cli.train_model import run_training

    run_training()


if __name__ == "__main__":
    app()
