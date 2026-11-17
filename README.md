[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · gta6-twitch-ccu-forecast-01 · rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/RizzZzze/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![make.fun · gta6-twitch-ccu-forecast-01 · reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/RizzZzze/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU quantile forecast

Entry for `gta6-twitch-ccu-forecast-01`: forecast the five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of Twitch concurrent viewers in the Grand Theft
Auto VI category for each of hours 2–24 of release day (2026-11-19 UTC),
given everything recorded up to the end of hour 1. Scored by mean pinball
loss on the log scale.

## Method

`main.py` (Python 3.13, standard library only, no network) fits at runtime
from the case's own comparables:

- Each past launch gives a normalised release-day shape,
  ln(V_h / V_1) for h = 2..24. The forecast is hour 1's print times the
  pool-median shape, with a per-hour spread from the pool's dispersion
  (IQR / 1.349, Gaussian quantiles, outer tails inflated).
- Two pools: launches whose day rose steeply from hour 1 (a single global
  unlock after a quiet first hour) and the rest (rolling / already-live
  launches). The target joins the steep pool only when its own hour 1 is
  quiet in absolute terms.
- Schedule calibration: when the target's notes say its launch already
  happened before the checkpoint (early access, or a global simultaneous
  unlock at or before it), the shape's rising part is halved — no major
  unlock step remains ahead, so the day is post-peak.
- Fallbacks: a pool too small to fit, or any failure in the model path,
  degrades to the house persistence forecast; any failure in a file's
  processing degrades to a valid fallback output. The program never crashes
  and never emits an invalid shape. It is deterministic: fixed iteration
  order, no randomness, no clock.

## Local validation

Backtest on 37 leave-one-out pseudo-cases built from the kit's comparables
(each launch scored against its own release day with itself excluded from
the fit pool), scored by an exact replica of the rules' scorer. Pinball
loss, lower is better:

| set | house persistence | this entry |
|---|---|---|
| rolling launches (n=12), mean | 0.356804 | 0.304029 |
| rolling launches (n=12), median | 0.254468 | 0.220402 |
| all 37 launches, mean | 0.929650 | 0.426615 |

No rolling case regresses against the house by more than 0.023; rolling
90% coverage is 0.906. Runtime is about 0.07 s per case (limit 8 s).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case: `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for
hours 2–24.
