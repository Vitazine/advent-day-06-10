import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = Path(__import__('os').environ.get('DAY6_SOURCE_DIR', 'source-day-06'))
MODEL = OLD / 'models/Qwen3-4B-Instruct-2507-4bit'
SYSTEM = 'Перепиши фразу нейтрально, сохрани смысл. Верни только новую фразу.'
CONTACT = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}|(?<!\w)@\w{3,}|(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)', re.I)
TOXIC = re.compile(r'\b(?:хуй\w*|хуе\w*|хуё\w*|хуя\w*|хер\w*|нахуй|нахер|похуй|похер|пизд\w*|еба\w*|ебну\w*|ёба\w*|уеб\w*|уёб\w*|заеб\w*|заёб\w*|бля\w*|говн\w*|дерьм\w*|сран\w*|сраный|обоср\w*|мудак\w*|мудил\w*|долбо\w*|далба\w*|дебил\w*|идиот\w*|кретин\w*|придур\w*|ублюд\w*|мраз\w*|сволоч\w*|сука|суки|суку|сучка|тупица|тупой|тупые|тупого|дурак|дура|дуры|дурой|безмозгл\w*|рукожоп\w*|криворук\w*|болван\w*|балбес\w*|олух\w*|хамло|козёл|козел)\b', re.I)

def toxic_target(text):
    return bool(TOXIC.search(text))

def key(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', s).casefold().replace('ё', 'е')))

def eligible(source, target):
    return bool(key(source) and key(target) and key(source) != key(target)
                and not CONTACT.search(source + ' ' + target)
                and not re.search(r'<\|[^>]*\|>|</?think>', source + target)
                and not any(unicodedata.category(c) in {'Cc', 'Cf'} and not c.isspace() for c in source + target))

def assert_disjoint(a, b):
    ka = {key(r[f]) for r in a for f in ('source', 'target')}
    kb = {key(r[f]) for r in b for f in ('source', 'target')}
    if ka & kb:
        raise ValueError('Source/target overlap between splits')

def messages(source, target=None):
    result = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': source}]
    if target is not None:
        result.append({'role': 'assistant', 'content': target})
    return result

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]

def configure(tokenizer):
    template = ROOT / 'chat_template.jinja'
    tokenizer.chat_template = template.read_text()
    return tokenizer
