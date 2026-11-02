"""Use the reviewed runner with isolated short-prompt data and manifest."""
from pathlib import Path
import runner
import runner_config
VARIANT=Path(__file__).resolve().parent/'short-prompt'
runner.ROOT=VARIANT
runner.verify_manifest=lambda: runner_config.verify_manifest(VARIANT)
if __name__=='__main__':
    runner.verify_manifest()
    runner.main()
    runner.verify_manifest()
