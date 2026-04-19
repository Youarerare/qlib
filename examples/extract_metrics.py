# extract_metrics.py
import pickle, pandas as pd, numpy as np, sys
from pathlib import Path

recorders = [
    ('169ef8c59bfc441880d7cd8d24286658', 'Alpha360'),
    ('07705a281aa34466ae2db549f77ac8ec', 'Alpha158'),
]

for rid, name in recorders:
    p = Path(f'mlruns/706158807618320602/{rid}/artifacts/sig_analysis')
    ic = pickle.load(open(p / 'ic.pkl', 'rb'))
    ric = pickle.load(open(p / 'ric.pkl', 'rb'))
    mean_ic = ic.mean()
    icir = ic.mean() / ic.std() if ic.std() != 0 else np.nan
    mean_ric = ric.mean()
    ricir = ric.mean() / ric.std() if ric.std() != 0 else np.nan
    sys.stdout.write(f'{name}: mean IC={mean_ic:.6f}, ICIR={icir:.6f}; mean RankIC={mean_ric:.6f}, RankICIR={ricir:.6f}\n')
    sys.stdout.flush()

