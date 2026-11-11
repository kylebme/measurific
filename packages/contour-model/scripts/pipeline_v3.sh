#!/usr/bin/env bash
# Corpus v3 training-scale build + first training runs (code: branch contour-corpus-v3, 7dd1e04).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT="/media/kyle/T7 Shield/inspection-contour-training/v3-20261018"
LOG="$OUT/logs"; mkdir -p "$LOG" "$OUT/runs" "$OUT/bench"
stage() { echo "$(date '+%F %T') $*" | tee -a "$LOG/pipeline.log"; }

stage "corpus: 8000 designs"
$PY -m contour_model.corpus_v3 "$OUT/corpus" --count 8000 --workers 10 > "$LOG/corpus.log" 2>&1

stage "extraction: PAE (CPU) then edge model (GPU)"
EXTRACT_THREADS=2 $PY -m contour_model.extract_v3 "$OUT/corpus" "$OUT/extract" --workers 10 \
  --plan train:4000:2000,val:500:500,test:500:500,holdout:500:500,stress:250:250 > "$LOG/extract.log" 2>&1

stage "mix: clean x1, PAE x2, edge x2"
$PY - "$OUT" <<'EOF' > "$LOG/mix.log" 2>&1
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); idx = json.loads((out/'extract'/'index.json').read_text())
for kind in ('pae', 'edge'):
    d = out/f'extract-{kind}'; d.mkdir(exist_ok=True)
    rows = [dict(e, file=str(out/'extract'/e['file'])) for e in idx if e['extractor'] == kind]
    (d/'index.json').write_text(json.dumps(rows)); (d/'config.json').write_text((out/'extract'/'config.json').read_text())
    print(kind, len(rows))
EOF
$PY autoresearch/make_mix.py "$OUT/mix" "$OUT/corpus:1" "$OUT/extract-pae:2" "$OUT/extract-edge:2" >> "$LOG/mix.log" 2>&1

COMMON=(--eval-every 1000 --val-limit 200 --direct-global-context --full-context --pooled-tokens 256 --dropout .1
        --form-error .003 --boundary-positive-weight 4 --boundary-weight .3 --query-weight 1.5 --mirror --val-extractor pae)
stage "training: warm (from release-v6, 32k, lr 1e-4) and scratch (48k, lr 3e-4) in parallel"
$PY -m contour_model.train_primitives "$OUT/mix" "$OUT/runs/warm-v6" --steps 32000 --lr .0001 --seed 311 \
  --init release-v6/model.pt "${COMMON[@]}" > "$LOG/train-warm.log" 2>&1 &
WARM=$!
$PY -m contour_model.train_primitives "$OUT/mix" "$OUT/runs/scratch" --steps 48000 --lr .0003 --seed 312 \
  "${COMMON[@]}" > "$LOG/train-scratch.log" 2>&1 &
SCRATCH=$!

SETS=v3_pae,v3_pae_val,v3_edge,v3_edge_val,v3_pae_holdout,v3_edge_holdout,v3_pae_stress,v3_edge_stress,v3_clean,edge,edge_val,orig
stage "baseline: release-v6 on v3 and old bench sets"
$PY scripts/bench.py release-v6/model.pt "$OUT/bench/release-v6.json" --sets "$SETS" > "$LOG/bench-v6.log" 2>&1 || stage "baseline bench failed"

wait $WARM && stage "warm run finished" || stage "warm run FAILED"
wait $SCRATCH && stage "scratch run finished" || stage "scratch run FAILED"
for run in warm-v6 scratch; do
  if [ -f "$OUT/runs/$run/best.pt" ]; then
    stage "bench: $run"
    $PY scripts/bench.py "$OUT/runs/$run/best.pt" "$OUT/bench/$run.json" --sets "$SETS" > "$LOG/bench-$run.log" 2>&1 || stage "bench $run failed"
  fi
done
stage "done"
