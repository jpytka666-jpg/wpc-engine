#!/usr/bin/env bash
# Rozmowa z lokalnym Qwenem w terminalu.
#
#   qwen "napisz funkcje sortujaca"   - jedno pytanie, jedna odpowiedz
#   qwen                              - rozmowa, konczysz slowem: koniec
#
# Model: Qwen3-Coder-30B spakowany do 4 bitow (15 GB). Okolo 120 slow na minute.
# Pierwsze uruchomienie po starcie komputera jest wolniejsze - dysk musi sie rozgrzac.
set -uo pipefail

BIN=/home/aions/wpc-workspace/target/release/wpc-runtime
MODEL=/home/aions/qwen3-coder-run
WPC=/home/aions/qwen3-coder-wpc4
SCHEME=v4
TOKENS=${QWEN_TOKENS:-150}

if [ ! -x "$BIN" ]; then
  echo "Nie znajduje silnika w $BIN"
  echo "Zbuduj go:  cd /home/aions/wpc-workspace && cargo build --release"
  exit 1
fi

zapytaj() {
  local PYTANIE="$1"
  # Silnik dopisuje wlasna odpowiedz do polecenia, wiec odcinamy poczatek,
  # zeby nie ogladac swojego pytania drugi raz.
  "$BIN" --model "$MODEL" --wpc "$WPC" --scheme "$SCHEME" \
         --prompt "$PYTANIE" --max-tokens "$TOKENS" 2>/dev/null \
    | tail -n +1 | sed "s|^${PYTANIE}||"
}

if [ $# -gt 0 ]; then
  zapytaj "$*"
  exit 0
fi

echo "Qwen 30B, lokalnie. Pisz pytanie i Enter. Zeby wyjsc: koniec"
echo "Odpowiada okolo 120 slow na minute, wiec chwile to trwa."
echo

while true; do
  printf "\n> "
  IFS= read -r LINIA || break
  case "$LINIA" in
    koniec|exit|quit|"") [ -z "$LINIA" ] && continue; echo "Do zobaczenia."; break ;;
  esac
  echo
  zapytaj "$LINIA"
done
