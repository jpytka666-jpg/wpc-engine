# AIONS / NOWORODEK — kontrolowane automatyczne powiększanie mózgu

<!-- PROVENANCE: OWNER=explicit approval "Włącz automatyczne powiększanie mózgu", 2026-10-19 ~16:46 BST; AUTHOR=Nez GPT-6; OBSERVED_ON=ZBOOK + Darkstar 2026-10-19 16:49–17:07 BST; CODE=Rust grower + Node24 ES module; NO PYTHON on Darkstar. -->

## Co WŁĄCZONO

- Mechanizm `tools/auto-growth.mjs` jest importowany i wywoływany **wyłącznie w zakończonym cyklu** `~/.local/share/aions/noworodek-train/train-loop.mjs`, po zapisie `history.jsonl`, danych SQLite i obecnych wag.
- Produkcyjny trener pozostaje **jedynym piszącym** `best.bin`. Nie uruchomiono drugiego niezależnego trenera, schedulera ani procesu zapisującego wagi.
- Stan automatyki: `~/noworodek/growth-lab/auto-growth-state.json`, **enabled=true**, inicjalnie `growthAt=2026-10-19T15:34:55Z`, `promotions=1` (dotychczasowa ręcznie wykonana rozbudowa).
- Zdarzenia: `~/noworodek/growth-lab/auto-growth-events.jsonl`. Błędy obsługiwane bez zawieszania treningu, dodatkowo `growth-errors.jsonl` w katalogu trenera.
- Dokładny snapshot wdrożonego skryptu: `tools/train-loop-auto-growth-20261019.mjs`. Oryginał sprzed zmiany `~/.local/share/aions/noworodek-train/train-loop.mjs.bak-20261019-auto-growth`.

## Warunki AUTOMATYCZNEJ decyzji

1. Co najmniej **24 godziny po poprzednim powiększeniu**.
2. Nie częściej niż jeden porównywalny pomiar co **4 godziny**, wyłącznie między cyklami treningu.
3. Co najmniej **6 pomiarów na tym samym, niezmiennym teście technicznym**, oddalonych łącznie o co najmniej 18 godzin; ostatnia połowa wyników nie poprawiła się przynajmniej o **0,5%** wobec poprzedniej.
4. Co najmniej **20 GiB wolnego miejsca** na ZBooku; identyfikacja M60A po pełnym GPU UUID. Plik nowych wag nie może przekroczyć 10% VRAM, szacowane szczytowe zużycie GPU nie może przekroczyć 65% VRAM.
5. Jednorazowy wzrost: **podwojenie szerokości MLP**, np. 512 → 1024 → 2048 itd., bez arbitralnego dożywotniego limitu, ale każdorazowo z powtórzonymi bramkami.
6. Kopia bezpieczeństwa `before.bin.bak`, wersja próbna tworzona Rust `noworodek-grow`, sprawdzenie zgodności działania obu modeli na GPU (wyniki zaokrąglone do 4 miejsc).
7. Test A/B po **64 krokach nauki na identycznych danych**, dwukrotnie z różnymi ziarnami; większa wersja musi mieć wynik błędu co najmniej **0,5% niższy** niż mniejsza w OBU próbach.
8. Dopiero wtedy atomowy zapis większych wag i rejestr zdarzenia `promoted`. W przeciwnym razie brak zmiany produkcji, zapis powodu odrzucenia i przynajmniej **24-godzinny odstęp** do ponownej próby.

**Uwaga o pomiarach:** niezmienny plik `/garage/noworodek/gpu/lekcje-20261001.u16` jest technicznym testem kontrolnym, NIE niezależnym sprawdzianem rozumowania i NIE gwarantuje generalizacji ani bezpieczeństwa AGI. Nie należy interpretować pojedynczego loss jako IQ. Potrzebny jest osobny ślepy zestaw ewaluacyjny.

## Sprawdzone na żywo

- 2026-10-19 ~16:49 BST: aktywny trener, 436 zakończonych cykli, M60A 92%, format NWRD=**15 186 304 parametrów**, intermediate=512.
- `node --check` dla modułu i produkcyjnego trenera: PASS.
- `node auto-growth.mjs --self-test`: PASS (plateau, brak plateau, zbyt mało próbek).
- `AUTO_GROW_DRY_RUN=1` uruchomione na rzeczywistym best.bin: PASS, rozpoznano 512/128/15 186 304 parametrów i 60 745 774 bajty.
- 2026-10-19 **17:05:31–17:07:06 BST**: sterowany STOP wyłącznie po ukończeniu bieżącego cyklu, potem uruchomienie `aions-noworodek-train.service` z nowym modułem. Nowy PID=3739918, ActiveState=active, SubState=running, NRestarts=0, STOP absent, usługa systemd enabled na starcie użytkownika.
- O 17:07:37 nowa pętla `training`, cykl `20261019T160713`, 1022 lekcje, 9 171 886 tokenów. Weryfikacja pierwszych pełnych cykli po **NOWEJ automatyce** pozostaje bieżącym zadaniem kontroli; nie należy nazywać przyszłej automatycznej rozbudowy zakończoną.

## Zakres i ograniczenia

- **Włączone monitorowanie oraz przyszła, warunkowa automatyczna promocja**, a nie bezwarunkowe dokładanie wag. Obecne 15 186 304 parametrów pozostają bez zmian, dopóki wszystkie warunki nie zostaną spełnione.
- Rozbudowa obejmuje MLP w **jednowarstwowym** dekoderze, nie wymianę architektury na duże LLM, nie rozszerzenie alfabetu CBMS, nie niezależne zwiększanie liczby warstw czy hidden.
- Żadnych zmian do M60B, P40, GPT-OSS CPU, WPC, Gangu, sieci, telemetrii. Darkstar bez Pythona.
- Aby **wyłączyć sam wzrost bez zatrzymywania treningu**, ustawić `enabled:false` w `auto-growth-state.json`; funkcja sprawdza ustawienie na końcu każdego cyklu. Włączyć ponownie przez `enabled:true`.
- W przypadku nieudanych testów, braku zasobów lub błędu skrypt **nie powinien promować wag**, a główny trener kontynuuje pracę. Monitorować dziennik zdarzeń.
- Archiwum kopii i logów rośnie wraz z obserwacjami. Należy regularnie obserwować wolną przestrzeń; NIE usuwać starych snapshotów produkcyjnych bez osobno zatwierdzonej polityki.
