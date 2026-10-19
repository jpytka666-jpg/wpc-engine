# NOWORODEK / Mikernel Codeone — pierwsze kontrolowane powiększenie mózgu

<!-- AIONS provenance: AUTHOR=Nez/GPT-6; OWNER approval 2026-10-19; OBSERVED 2026-10-19 16:07–16:37 BST; CODE=Rust; PRODUCTION=promotion verified; SECURITY=no Python on Darkstar. -->

## Stan przed i po

- Oryginał: NWRD v1, 12 tensorów, 58 296 tokenów, hidden=128, intermediate=256, jedna warstwa, **15 088 000 parametrów**, plik 60 352 558 bajtów.
- Wersja produkcyjna po zmianie: intermediate=512, **15 186 304 parametrów**, plik 60 745 774 bajty. **+98 304 parametry** (+0,65%).
- Wzrost dotyczy trzech macierzy przetwarzania MLP. Wszystkie stare wartości zachowane; dodane gate/up z deterministycznymi wagami startowymi, down z zerami, aby odpowiedzi na starcie pozostały bez zmian.
- Kod narzędzia: `src/bin/noworodek-grow.rs`. Potrafi zwiększać wielkość w etapach maks. dwukrotnych, **bez stałego końcowego limitu**; rzeczywiste limity narzuca GPU/RAM, niezależne testy i kontrola. Automatyczny wyzwalacz kolejnych wzrostów NIE jest jeszcze wdrożony.

## Dowody i błędy

- Pierwszy eksperymentalny układ macierzy NIE przeszedł testu GPU: stary loss 18,5623, powiększony 15,7347, więc odrzucony bez zmian produkcji.
- Poprawiony układ, 4/4 testów Rust PASS, w tym podwójne poszerzanie i zgodność wyniku. Realna M60A: heldout **18,5623 = 18,5623**, zestaw 24 **18,2388 = 18,2388**.
- Produkcyjna porcja 256 przykładów, lr=0.005: stary model **19,5058 → 12,7160**, większy **19,5058 → 12,7155**. Próby na jednym przykładzie dawały gorsze wyniki, dlatego nie traktowano ich jako miary treningu produkcyjnego.
- **19 października 16:34:31 BST:** zatrzymanie po zakończeniu pełnego cyklu, bez przerwania obliczeń. Backup `~/noworodek/growth-lab/pre-growth-20261019T163409.bin.bak`, SHA-256 `b1ade34fd11da6c838d09d30da02e2e0630ebd86f26b9a67793b3578cea8975a`.
- Kandydat z najnowszego backupu `~/noworodek/growth-lab/grown-inter512-20261019T163409.bin`, SHA-256 `3237ef0fe6fce59cbc378c7ee407024512bea6d71ea66c61b2874127bd79bc5a`. Ostatnia zgodność GPU na świeżych wagach: **20,4153 = 20,4153** i **21,4349 = 21,4349**.
- **16:34:55:** atomowa promocja nowego best.bin, START trenera; **16:35:34:** usługa `active/running`, MainPID=3512513, NRestarts=0, STOP nie istnieje, CUDA pracuje na M60A.
- Dwa pierwsze ukończone cykle na większym modelu przyjęte: `20261019T153502` loss **2,2677 → 0,2316**, `20261019T153621` **0,2860 → 0,2307**, oba exit=0.

## Bezpieczeństwo i kolejny krok

- Kopia starego mózgu zachowana, historyczne snapshoty nietknięte. Rollback tylko na bezpiecznej granicy cyklu.
- Zapisany skrypt promocji: `tools/promote-inter512-20261019.sh` — historyczna, jednorazowa procedura; nie uruchamiać ponownie na modelu intermediate=512.
- Pełne logi testowe zachowane na Darkstarze `/garage/noworodek/growth-lab/` i w katalogu promocji `promotion-20261019T163409`.
- GPU M60B (Qwen), P40, CPU, Paperclip i sieć niezmieniane.
- Następne kroki: niezależne testy rozumowania na niewidzianych danych; monitorowanie zmian i zużycia pamięci/dysku; dopiero potem projekt automatycznego rozrastania na podstawie potwierdzonego plateau, nie arbitralnego licznika cykli.
- Każda wersja wag ma ok. 61 MB; wiele snapshotów zużywa duży dysk. Nie usuwać historycznych kopii bez zaakceptowanej polityki retencji.
