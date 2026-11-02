"""One-shot Day 8 chat using the frozen evaluation settings; no output files."""
import argparse
import json
import sys
from pathlib import Path

DAY = Path(__file__).resolve().parent
sys.path.insert(0, str(DAY / 'training-v1'))
from core import MODEL, ROOT, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['chat'])
    parser.add_argument('--name', required=True, choices=['base', 'checkpoint-704'])
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error('--text must not be empty')
    verify()
    adapter = None
    if args.name != 'base':
        adapter = ROOT / 'checkpoints' / args.name
        selection = json.loads((ROOT / 'selection.json').read_text())
        if sha(adapter / 'adapters.safetensors') != selection['selected_adapter_sha256']:
            raise ValueError('Adapter differs from the frozen selection')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(str(MODEL), adapter_path=str(adapter) if adapter else None,
                            tokenizer_config={'trust_remote_code': False})
    tokenizer.chat_template = (DAY / 'chat_template.jinja').read_text()
    messages = [{'role': 'system', 'content': (DAY / 'system.txt').read_text().strip()},
                {'role': 'user', 'content': args.text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    mx.random.seed(42)
    for part in stream_generate(model, tokenizer, prompt=prompt, max_tokens=128,
                                sampler=make_sampler(temp=0.0)):
        print(part.text, end='', flush=True)
    print()


if __name__ == '__main__':
    main()
