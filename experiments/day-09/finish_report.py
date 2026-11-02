"""One-shot local report completion for the already running Day9 job."""
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
started=time.monotonic()
while time.monotonic()-started<6*3600:
    path=ROOT/'status.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state.get('state')=='failed':
            print('Pipeline failed; see status.json and logs. No success report generated.',flush=True)
            sys.exit(1)
        if state.get('state')=='complete':
            subprocess.run([sys.executable,str(ROOT/'build_report.py')],check=True)
            p=ROOT/'README.md';s=p.read_text();s=s.replace('Статус: датасет подготовлен и проверен; новый baseline выполнен, GPU-пилоты продолжаются. Основное обучение ещё не запускалось.','Статус: обучение и итоговое сравнение завершены. Результаты и ограничения — в final-report.md; рекомендация — в recommendation.json.')
            p.write_text(s)
            print('Report ready: '+str(ROOT/'final-report.md'),flush=True)
            break
    time.sleep(10)
else:
    print('Report wait timeout after 6h; inspect status.json. Completion not claimed.',flush=True)
    sys.exit(2)
