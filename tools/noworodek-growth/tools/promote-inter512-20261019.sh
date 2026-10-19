#!/usr/bin/env bash
# AIONS Noworodek: one safe MLP width growth at a cycle boundary.
# Owner approved 2026-10-19; no Python; live source only swapped after exact GPU parity.
set -Eeuo pipefail
: "${AIONS_NOWORODEK_GPU_UUID:?set AIONS_NOWORODEK_GPU_UUID before executing}"
umask 077
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D="$HOME/.local/share/aions/noworodek-train"
L="$HOME/noworodek/growth-lab"
BIN="$HOME/noworodek/gpu-trainer/target/debug/noworodek-grow"
SERVICE=aions-noworodek-train.service
stamp="$(date +%Y%m%dT%H%M%S)"
backup="$L/pre-growth-$stamp.bin.bak"
candidate="$L/grown-inter512-$stamp.bin"
remote="/garage/noworodek/growth-lab/promotion-$stamp"
swapped=0
finished=0
stop_owned=0
rollback() {
  rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    echo "FAIL: promotion aborted (rc=$rc), rollback if needed" >&2
    if (( swapped == 1 )) && [[ -f "$backup" ]]; then
      cp -p "$backup" "$D/best.bin.rollback-tmp"
      mv -f "$D/best.bin.rollback-tmp" "$D/best.bin"
      echo "ROLLBACK: original checkpoint restored" >&2
    fi
    if (( stop_owned == 1 )); then rm -f "$D/STOP"; fi
    if (( stop_owned == 1 )) && ! systemctl --user is-active --quiet "$SERVICE"; then
      systemctl --user start "$SERVICE" || true
    fi
  fi
  exit "$rc"
}
trap rollback EXIT
[[ ! -e "$D/STOP" ]] || { echo 'REFUSED: STOP already present'; exit 2; }
[[ -x "$BIN" ]] || { echo 'REFUSED: no tested growth binary'; exit 2; }
systemctl --user is-active --quiet "$SERVICE" || { echo 'REFUSED: training not running'; exit 2; }
[[ ! -e "$backup" && ! -e "$candidate" ]] || exit 2
echo "BEGIN growth-at-boundary $(date -Is), stamp=$stamp"
touch "$D/STOP"
stop_owned=1
for i in $(seq 1 360); do
  if grep -q '"state": "stopped"' "$D/status.json" && ! systemctl --user is-active --quiet "$SERVICE"; then
    finished=1
    break
  fi
  sleep 2
done
(( finished == 1 )) || { echo 'WAIT_TIMEOUT: current cycle not finished'; exit 3; }
echo "CYCLE_FINISHED $(date -Is)"
cp -p "$D/best.bin" "$backup"
sha256sum "$backup"
"$BIN" --source "$backup" --output "$candidate" --target-inter 512
sha256sum "$candidate"
# Compare on same immutable held-out lesson set, with the GPU used by production.
ssh -o BatchMode=yes -o ConnectTimeout=5 darkstar "mkdir -p '$remote'"
scp -q "$backup" "$candidate" "darkstar:$remote/"
ssh -o BatchMode=yes -o ConnectTimeout=5 darkstar "set -e
cd '$remote'
export LD_LIBRARY_PATH=/opt/aions-cuda/lib
export CUDA_VISIBLE_DEVICES=${AIONS_NOWORODEK_GPU_UUID}
timeout 90 nice -n 10 /garage/noworodek/gpu/noworodek-gpu --ids /garage/noworodek/gpu/lekcje-20261001.u16 --load '$(basename "$backup")' --steps 0 --batch 8 --eval-windows 48 > base.eval.log 2>&1
timeout 90 nice -n 10 /garage/noworodek/gpu/noworodek-gpu --ids /garage/noworodek/gpu/lekcje-20261001.u16 --load '$(basename "$candidate")' --steps 0 --batch 8 --eval-windows 48 > grown.eval.log 2>&1
b1=\$(grep '^odlozone po wczytaniu:' base.eval.log | awk '{print \$4}')
g1=\$(grep '^odlozone po wczytaniu:' grown.eval.log | awk '{print \$4}')
b2=\$(grep '^zgodnosc24 po wczytaniu:' base.eval.log | awk '{print \$4}')
g2=\$(grep '^zgodnosc24 po wczytaniu:' grown.eval.log | awk '{print \$4}')
echo \"GPU_PARITY: heldout=\$b1/\$g1 24=\$b2/\$g2\"
test -n \"\$b1\" && test -n \"\$b2\" && test \"\$b1\" = \"\$g1\" && test \"\$b2\" = \"\$g2\"
"
echo "PROMOTION_GATE_PASS"
cp -p "$candidate" "$D/best.bin.growth-staging-$stamp"
mv -f "$D/best.bin.growth-staging-$stamp" "$D/best.bin"
swapped=1
rm -f "$D/STOP"
systemctl --user start "$SERVICE"
systemctl --user is-active --quiet "$SERVICE"
echo "PROMOTION_PASS $(date -Is) old=$(stat -c '%s' "$backup") bytes new=$(stat -c '%s' "$D/best.bin") bytes"
echo "BACKUP=$backup CANDIDATE=$candidate"
trap - EXIT
