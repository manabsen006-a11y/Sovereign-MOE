"""Compare solves with and without conflict analysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from sovopt.io.mps import read_mps
from sovopt.mip.tree import solve_mip, MIPParams
from bench.verify import verify
from bench.fetch import read_reference

print('%-9s %-9s %-11s %-14s %-14s %-7s %-8s %-4s %s' % (
    'inst','conflict','status','objective','reference','nodes','time','chk','learned'))
for nm in ['p0201','misc07','gr4x6','flugpl','gt2']:
    ref = read_reference(f'data/instances/{nm}.mps').get('best_soln')
    got = {}
    for on in (False, True):
        p = read_mps(f'data/instances/{nm}.mps')
        t = time.perf_counter()
        s = solve_mip(p, MIPParams(device='cpu', time_limit=60, conflict=on))
        dt = time.perf_counter() - t
        chk = 'ok' if (s.x is not None and verify(p, s.x, s.objective, feas_tol=1e-6, int_tol=1e-6).ok) \
              else ('BAD' if s.x is not None else '-')
        info = getattr(s, 'info', {}) or {}
        cst = info.get('conflict', {})
        got[on] = (s.objective if s.x is not None else float('nan'), s.status.name)
        print('%-9s %-9s %-11s %-14.10g %-14s %-7d %-8.2f %-4s %s' % (
            nm, 'on' if on else 'off', s.status.name,
            s.objective if s.x is not None else float('nan'),
            ('%.10g' % ref) if ref else 'n/a', s.nodes, dt, chk,
            cst if on else ''))
    a, b = got[False], got[True]
    if a[1] == 'OPTIMAL' and b[1] == 'OPTIMAL':
        ok = abs(a[0] - b[0]) < 1e-6 * max(1, abs(a[0]))
        print('   -> optimum preserved: %s%s' % (ok, '' if ok else '  *** UNSOUND ***'))
