"""Isolated, fixed-row benchmark; first argument is long or short."""
import sys
from pathlib import Path
import runner,runner_config
variant=sys.argv.pop(1)
if variant not in ('long','short'):raise ValueError('Unknown variant')
root=Path(__file__).resolve().parent/'short-prompt/benchmark'/variant
runner.ROOT=root
runner.verify_manifest=lambda:runner_config.verify_manifest(root)
if __name__=='__main__':
 runner.verify_manifest();runner.main();runner.verify_manifest()
