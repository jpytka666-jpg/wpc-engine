#!/usr/bin/env bash
# Rozmowa z malym Qwenem w terminalu.
#
#   qwen-maly "write a short function that sorts a list"   - jedno pytanie, jedna odpowiedz
#   qwen-maly                                              - rozmowa, konczysz slowem: koniec
#
# Model: Qwen3-4B-Instruct-2507 spakowany do 4 bitow (2.0 GB). Okolo 240 slow na minute,
# czyli ponad piec razy szybciej niz duzy 30B.
#
# PISZ PO ANGIELSKU. Sprawdzone 2026-08-25: po angielsku odpowiada pelnymi, poprawnymi
# zdaniami; po polsku miesza jezyki i wtraca chinskie znaki. To nie wina kompresji -
# wariant 6-bitowy (wpc3) robi dokladnie to samo, a liczy wolniej. Maly model po prostu
# slabo zna polski.
#
# Pierwsze uruchomienie po starcie komputera jest wolniejsze - dysk musi sie rozgrzac.
set -uo pipefail

BIN=/home/aions/wpc-workspace/target/release/wpc-runtime
MODEL=/mnt/e/models-src/qwen3-4b-it
WPC=/home/aions/qwen3-4b-wpc4
SCHEME=v4
TOKENS=${QWEN_TOKENS:-120}

if [ ! -x "$BIN" ]; then
  echo "Nie znajduje silnika w $BIN"
  echo "Zbuduj go:  cd /home/aions/wpc-workspace && cargo build --release"
  exit 1
fi

if [ ! -f "$WPC/model_v4.wpc" ]; then
  echo "Nie znajduje spakowanych wag w $WPC"
  exit 1
fi

if [ ! -f "$MODEL/config.json" ]; then
  echo "Nie znajduje konfiguracji i tokenizera w $MODEL"
  exit 1
fi

zapytaj() {
  local PYTANIE="$1"
  # --chat owija pytanie w znaczniki rozmowy modelu. Bez tego model instruktazowy
  # traktuje pytanie jak niedokonczony dokument i dopisuje ciag dalszy zamiast odpowiedzi.
  #
  # 2>/dev/null wycisza gadanine silnika - wczytywanie 36 warstw, czasy, numery tokenow.
  # To wszystko idzie kanalem bledow, wiec na ekranie zostaje sama odpowiedz.
  # sed odcina poczatek, bo silnik dokleja odpowiedz do powtorzonego pytania.
  "$BIN" --model "$MODEL" --wpc "$WPC" --scheme "$SCHEME" --chat \
         --prompt "$PYTANIE" --max-tokens "$TOKENS" 2>/dev/null \
    | sed "s|^${PYTANIE}||"
}

if [ $# -gt 0 ]; then
  zapytaj "$*"
  exit 0
fi

echo "Maly Qwen (4B), lokalnie. Pisz pytanie i Enter. Zeby wyjsc: koniec"
echo "Odpowiada okolo 240 slow na minute. PISZ PO ANGIELSKU - tak odpowiada poprawnie."
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
