"""Pairwise summaries of explicit manual labels; never judges meaning automatically."""
import random

def paired_stats(base, tuned):
    a={r['id']:r['pass'] for r in base}
    b={r['id']:r['pass'] for r in tuned}
    if not a or a.keys()!=b.keys() or len(a)!=len(base) or len(b)!=len(tuned):
        raise ValueError('Expected identical nonempty sets of unique case IDs')
    if not all(isinstance(x,bool) for x in [*a.values(),*b.values()]):
        raise ValueError('Explicit boolean pass labels required')
    diffs=[int(b[k])-int(a[k]) for k in sorted(a)]
    n=len(diffs)
    rng=random.Random(901)
    boot=sorted(sum(rng.choices(diffs,k=n))/n*100 for _ in range(10000))
    return {'n':n, 'base_pass':sum(a.values()),'tuned_pass':sum(b.values()),
            'improved':diffs.count(1),'regressed':diffs.count(-1),'unchanged_binary':diffs.count(0),
            'net_percentage_points':sum(diffs)/n*100,
            'paired_bootstrap_95_percentile_pp':[boot[249],boot[9749]],
            'limitations':'CI reflects sampling variation conditional on these subjective labels; not annotator bias or distribution shift.'}
