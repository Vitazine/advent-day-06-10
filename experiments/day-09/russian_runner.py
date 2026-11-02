"""Isolated Russian compressed-prompt inference."""
from pathlib import Path
import runner,runner_config
root=Path(__file__).resolve().parent/'short-prompt/russian'
runner.ROOT=root
runner.verify_manifest=lambda:runner_config.verify_manifest(root)
if __name__=='__main__':
 runner.verify_manifest();runner.main();runner.verify_manifest()
