# /brag plan — SOVOPT

**What it is:** an LP / MILP / QP optimization engine for refinery planning (SIH 2026, PS 26119, MRPL), written from the published mathematics — no solver library anywhere in it.
**For:** refinery planners (and the judges) who need a plan they can trust, and a solver India owns outright.
**What sets it apart:** the problem statement forbids building on any solver library — so every algorithm is from the papers, and every answer is checked by something that isn't the solver (an independent verifier, three engines that disagree in how they work, HiGHS as a comparator).
**Most impressive claim:** three independent engines, 11/11 MIPLIB to published values; agrees with HiGHS 11/11 to 7.0e-16.
**Visual hook:** the forbidden-imports list being struck out, one solver name after another.
**Real UI/flow:** `sovopt blend components.csv products.csv` → a named plan: OPTIMAL, margin 802,000, the recipe, `independent check: ACCEPTED`.
**Tone:** `polished` leaning `default` — dark engineering terminal, confident, no hype.
**Share caption:** see share-copy.txt.

## Visual identity (from ui/server.py)
bg #0e1418 · panel #151f25 · line #243139 · ink #dbe4e8 · dim #8496a0 · amber #d5a34a (CPU) · cyan #4fc2ce (GPU) · green #62ac84 (ok) · red #d97867 (bad). Monospace (Cascadia Mono / Consolas), letter-spaced caps for headings.

## Storyboard — 20.0 s, 1920×1080, 30 fps

| # | t | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–3.4 | **Hook** | Forbidden list from CLAUDE.md types in: `CBC  HiGHS  SCIP  GLPK  OR-Tools  scipy.optimize`, each struck through in red as it lands. Then: **"No solver library."** |
| 2 | 3.4–7.0 | **Reveal** | `SOVOPT` letter-spaced, big. "LP · MILP · QP — built from the mathematics up." Small: SIH 2026 · PS 26119 · MRPL |
| 3 | 7.0–12.2 | **In use** | Terminal: `sovopt blend components.csv products.csv` typed; real output rows reveal; Premium recipe bars grow; margin **802,000** counts up; `independent check: ACCEPTED` stamps green. |
| 4 | 12.2–16.4 | **Three engines** | Three cards: revised simplex / interior point / GPU first-order — each "11/11 MIPLIB" ticking in. Line: "Three engines. One answer." Chip: "vs HiGHS: 11/11, agree to 7.0e-16" |
| 5 | 16.4–20.0 | **Punchline** | "Built from scratch." / "Checked by everything else." then `python -m sovopt.cli demo` |

## Sound
~100 BPM, A minor → C major lift at the reveal. Soft pad + muted pluck arpeggio + light kick/hat from scene 3. Typing = quiet filtered ticks; strike-throughs = soft pitched swishes in key; ACCEPTED = a quiet two-note chime (E–A). Effects bus –14 dB under music, shared reverb.
