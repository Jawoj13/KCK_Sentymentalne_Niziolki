# Debug segmentera

Ten plik opisuje pola wyświetlane w panelu debug aplikacji. Służy do szybkiego sprawdzenia, czy system widzi ruch, czy
segmenter wykrywa początek powtórzenia oraz czy potrafi zakończyć powtórzenie.

Przykładowy debug:

```text
exercise_type: arms_only
segmenter_state: idle
segmenter_event: None
motion_signal: 0.000
stillness_frames: 0
recorded_frames: 0
feature_confidence: 0.91
confidence_ok: True
front_wrist_velocity: 0.000
front_ankle_velocity: 0.000
wrist_extension_change: 0.000
wrist_extension: 0.735
front_ankle_displacement: 0.000
```

---

## 1. `exercise_type`

```text
exercise_type: arms_only
```

Oznacza typ ćwiczenia, które aktualnie ocenia aplikacja.

Możliwe wartości:

```text
arms_only
step_only
full
```

Znaczenie:

- `arms_only` oznacza wyprost rąk bez kroku.
- `step_only` oznacza sam krok.
- `full` oznacza ćwiczenie pełne, czyli ręce plus krok.

To pole jest ważne, ponieważ segmenter używa innego sygnału ruchu dla każdego ćwiczenia.

Dla `arms_only` segmenter patrzy głównie na ruch nadgarstka i zmianę wyprostu ręki.

```python
motion_signal = front_wrist_velocity + wrist_extension_change
```

Dla `step_only` segmenter patrzy głównie na ruch kostki nogi prowadzącej.

```python
motion_signal = front_ankle_velocity
```

Dla `full` segmenter patrzy na rękę i nogę jednocześnie.

```python
motion_signal = 0.5 * front_wrist_velocity + 0.5 * front_ankle_velocity
```

Jeżeli użytkownik rusza ręką, ale `exercise_type` jest ustawiony na `step_only`, system może nie wykryć powtórzenia, bo
czeka na ruch nogi.

---

## 2. `segmenter_state`

```text
segmenter_state: idle
```

Oznacza aktualny stan segmentera.

Możliwe wartości:

```text
idle
recording_attack
finished
```

### `idle`

Segmenter czeka na rozpoczęcie ruchu.

Jeżeli użytkownik wykonuje ruch, ale stan cały czas pozostaje `idle`, sprawdź przede wszystkim:

- `motion_signal`,
- `feature_confidence`,
- `confidence_ok`,
- próg `motion_start_threshold` w konfiguracji segmentera.

### `recording_attack`

Segmenter wykrył początek powtórzenia i zapisuje kolejne klatki.

Jeżeli stan długo pozostaje `recording_attack`, to znaczy, że start działa, ale system nie wykrywa końca powtórzenia.

Najczęstsze przyczyny:

- `motion_signal` nie spada poniżej `stillness_threshold`,
- YOLO generuje jitter punktów,
- `stillness_frames` jest ustawione zbyt wysoko,
- brakuje fallbacku kończenia po peak ruchu.

### `finished`

Segmenter uznał, że powtórzenie się skończyło.

Po tym stanie backend powinien:

1. pobrać powtórzenie przez `get_repetition()`,
2. przekazać sekwencję do `evaluate_repetition()`,
3. wyświetlić wynik,
4. zresetować segmenter.

---

## 3. `segmenter_event`

```text
segmenter_event: started
```

Oznacza event zwrócony przez ostatnie wywołanie:

```python
segmenter.update(...)
```

Możliwe wartości:

```text
None
started
finished
```

### `None`

Brak nowego zdarzenia. Segmenter nadal czeka albo kontynuuje zapis.

### `started`

Wykryto początek powtórzenia.

Jeżeli widzisz `started`, ale później nie pojawia się wynik, to znaczy, że segmenter startuje, ale nie kończy
powtórzenia.

### `finished`

Wykryto koniec powtórzenia. W tym momencie powinien zostać wygenerowany wynik.

Jeżeli `finished` pojawia się w debug, ale wynik nie trafia do UI, problem jest prawdopodobnie w backendzie, emitowaniu
sygnału albo obsłudze `evaluation_ready`.

---

## 4. `motion_signal`

```text
motion_signal: 0.084
```

To główny sygnał ruchu używany przez segmenter.

Segmenter używa go do dwóch rzeczy:

1. wykrycia początku powtórzenia,
2. wykrycia zatrzymania ruchu.

Start powtórzenia następuje, gdy:

```python
motion_signal >= motion_start_threshold
```

Zatrzymanie jest liczone, gdy:

```python
motion_signal <= stillness_threshold
```

Interpretacja:

```text
0.000 do 0.010
prawie brak ruchu albo ruch odcięty przez deadzone

0.010 do 0.040
mały ruch, możliwy jitter albo bardzo delikatne poruszenie

0.040 do 0.100
ruch zwykle wystarczający do wykrycia startu przy luźnych progach MVP

powyżej 0.100
wyraźny ruch
```

Jeżeli użytkownik wykonuje ruch, ale `motion_signal` jest blisko zera, sprawdź:

- czy YOLO widzi właściwe keypointy,
- czy `dominant_side` jest dobrze ustawione,
- czy `exercise_type` odpowiada wykonywanemu ruchowi,
- czy deadzone velocity nie jest zbyt wysokie,
- czy ruch jest widoczny z kamery bocznej.

---

## 5. `stillness_frames`

```text
stillness_frames: 4
```

Oznacza liczbę kolejnych klatek, w których ruch był wystarczająco mały.

Klatka jest liczona jako spokojna, gdy:

```python
motion_signal <= stillness_threshold
```

Segmenter kończy powtórzenie, gdy:

```python
stillness_frames >= segmenter_config["stillness_frames"]
```

Przykład:

```python
"stillness_frames": 4
```

oznacza, że system potrzebuje 4 spokojnych klatek z rzędu, żeby uznać powtórzenie za zakończone.

Jeżeli `stillness_frames` cały czas wynosi `0`, mimo że użytkownik już się zatrzymał, to prawdopodobnie YOLO generuje
jitter.

Możliwe rozwiązania:

- zwiększyć `stillness_threshold`,
- zwiększyć deadzone velocity,
- wygładzić keypointy,
- dodać fallback `post_peak`.

---

## 6. `recorded_frames`

```text
recorded_frames: 18
```

Oznacza liczbę klatek zapisanych do aktualnego powtórzenia.

Interpretacja:

```text
0
segmenter jeszcze nie zaczął powtórzenia

1 do kilku klatek
segmenter dopiero zaczął zapis

kilkanaście lub kilkadziesiąt klatek
powtórzenie jest nagrywane

bardzo duża liczba i brak wyniku
segmenter nie wykrywa końca powtórzenia
```

Jeżeli `recorded_frames` rośnie bez końca, problemem zwykle jest zakończenie powtórzenia, nie start.

Sprawdź wtedy:

- `motion_signal`,
- `stillness_frames`,
- `stillness_threshold`,
- `max_duration_sec`,
- czy `_finish()` faktycznie zwraca `"finished"`.

---

## 7. `feature_confidence`

```text
feature_confidence: 0.91
```

Oznacza średnią pewność keypointów używanych do obliczania cech biomechanicznych.

Zakres:

```text
0.0 do 1.0
```

Interpretacja:

```text
0.80 do 1.00
bardzo dobra detekcja

0.50 do 0.80
akceptowalna detekcja

0.30 do 0.50
niestabilna detekcja

poniżej 0.30
wynik może być niewiarygodny
```

Niska wartość oznacza, że YOLO nie widzi dobrze części ciała.

Najczęstsze przyczyny:

- użytkownik stoi za daleko,
- część ciała wychodzi poza kadr,
- słabe oświetlenie,
- kamera jest ustawiona pod złym kątem,
- broń lub ręka zasłania keypointy.

---

## 8. `confidence_ok`

```text
confidence_ok: True
```

To flaga mówiąca, czy confidence jest wystarczające do użycia klatki.

Zwykle jest liczona na podstawie warunku:

```python
feature_confidence >= MIN_KEYPOINT_CONFIDENCE
```

Jeżeli `confidence_ok` ma wartość `False`, segmenter może ignorować klatkę.

Interpretacja:

```text
True
klatka może być użyta do segmentacji i oceny

False
klatka jest niewiarygodna i może zostać pominięta
```

Jeżeli cały czas widzisz `False`, system może w ogóle nie wykrywać powtórzeń.

Na czas MVP można tymczasowo obniżyć próg:

```python
MIN_KEYPOINT_CONFIDENCE = 0.25
```

Docelowo lepiej poprawić kadr i oświetlenie niż zbyt mocno obniżać próg.

---

## 9. `front_wrist_velocity`

```text
front_wrist_velocity: 0.063
```

Oznacza prędkość nadgarstka ręki prowadzącej.

Wartość jest normalizowana przez skalę ciała, więc nie powinna mocno zależeć od tego, czy użytkownik stoi bliżej lub
dalej od kamery.

Najważniejsze dla ćwiczeń:

```text
arms_only
full
```

Interpretacja:

```text
blisko 0.000
nadgarstek się nie porusza albo ruch jest zbyt mały

0.020 do 0.050
mały ruch

0.050 do 0.150
wyraźny ruch ręki

powyżej 0.150
bardzo szybki ruch albo możliwy jitter
```

Jeżeli użytkownik wykonuje wyprost ręki, ale `front_wrist_velocity` jest bliskie zera, sprawdź:

- czy ustawiona jest właściwa strona dominująca,
- czy kamera widzi nadgarstek,
- czy YOLO nie gubi ręki,
- czy ręka prowadząca jest faktycznie po stronie `dominant_side`.

---

## 10. `front_ankle_velocity`

```text
front_ankle_velocity: 0.041
```

Oznacza prędkość kostki nogi prowadzącej.

Najważniejsze dla ćwiczeń:

```text
step_only
full
```

Interpretacja:

```text
blisko 0.000
noga się nie porusza albo ruch jest zbyt mały

0.020 do 0.050
mały ruch nogi

0.050 do 0.150
wyraźny krok

powyżej 0.150
bardzo szybki ruch albo możliwy jitter
```

Jeżeli testujesz `arms_only`, ta wartość może być bliska zeru i to jest normalne.

Jeżeli testujesz `step_only`, a ta wartość jest bliska zeru podczas kroku, sprawdź:

- czy kamera boczna widzi kostkę,
- czy `dominant_side` wskazuje właściwą nogę,
- czy stopa nie wychodzi poza kadr,
- czy filmik rzeczywiście pokazuje ruch w osi obrazu.

---

## 11. `wrist_extension_change`

```text
wrist_extension_change: 0.027
```

Oznacza tempo zmiany wyciągnięcia ręki.

To nie jest pozycja ręki, tylko szybkość zmiany metryki `wrist_extension`.

Jeżeli ręka szybko przechodzi z pozycji zgiętej do wyprostowanej, ta wartość powinna wzrosnąć.

Najważniejsze dla:

```text
arms_only
full
```

Interpretacja:

```text
blisko 0.000
wyprost ręki się nie zmienia

0.020 do 0.050
powolna zmiana wyprostu

0.050 do 0.150
wyraźna zmiana wyprostu
```

Jeżeli `front_wrist_velocity` jest małe, ale `wrist_extension_change` rośnie, segmenter nadal może wykryć wyprost ręki.

---

## 12. `wrist_extension`

```text
wrist_extension: 0.735
```

Oznacza poziom wyciągnięcia ręki prowadzącej.

W uproszczeniu jest to:

```python
distance(front_wrist, front_shoulder) / front_arm_length
```

To pole mówi, jak bardzo nadgarstek jest oddalony od barku względem długości ręki.

Interpretacja:

```text
niższa wartość
ręka bliżej ciała albo bardziej zgięta

wyższa wartość
ręka bardziej wyciągnięta
```

Uwaga: to nie jest to samo co `front_elbow_angle`.

- `wrist_extension` pomaga znaleźć moment maksymalnego wyciągnięcia ręki.
- `front_elbow_angle` mówi, czy łokieć jest dobrze wyprostowany.

W scoringu jakości wyprostu ważniejszy jest kąt łokcia, ale `wrist_extension` jest przydatne do segmentacji i wyboru
klatki końcowej.

---

## 13. `front_ankle_displacement`

```text
front_ankle_displacement: 0.182
```

Oznacza przemieszczenie kostki nogi prowadzącej względem pozycji startowej.

Wartość jest normalizowana przez skalę ciała.

Najważniejsze dla:

```text
step_only
full
```

Interpretacja:

```text
0.000
noga jest w pozycji startowej

0.050 do 0.150
mały krok

0.150 do 0.250
krótki lub średni krok

0.250 do 0.700
zwykle poprawny zakres kroku według obecnych progów

powyżej 0.700
bardzo długi krok albo możliwy problem ze skalą/keypointami
```

Jeżeli użytkownik robi krok, ale `front_ankle_displacement` pozostaje blisko zera, sprawdź:

- czy wykrywana jest właściwa noga,
- czy kamera boczna widzi ruch w osi poziomej,
- czy stopa nie jest zasłonięta,
- czy `dominant_side` jest ustawione poprawnie.

---

# Typowe diagnozy

## Brak wykrycia początku powtórzenia

Objawy:

```text
segmenter_state: idle
segmenter_event: None
motion_signal: 0.003
confidence_ok: True
```

Znaczenie:

System widzi klatki, ale ruch jest zbyt słaby względem progu startu.

Co sprawdzić:

1. Czy `exercise_type` pasuje do ruchu.
2. Czy `front_wrist_velocity` albo `front_ankle_velocity` rośnie podczas ruchu.
3. Czy `motion_start_threshold` nie jest za wysoki.
4. Czy deadzone velocity nie ucina ruchu.

Możliwe szybkie poprawki:

```python
motion_start_threshold = 0.02
MOTION_DEADZONE = 0.01
```

---

## Start działa, ale nie ma wyniku

Objawy:

```text
segmenter_state: recording_attack
recorded_frames: 80
stillness_frames: 0
motion_signal: 0.035
```

Znaczenie:

Segmenter rozpoczął powtórzenie, ale nie wykrywa końca.

Co sprawdzić:

1. Czy `motion_signal` spada po zakończeniu ruchu.
2. Czy `stillness_threshold` nie jest za niski.
3. Czy jitter YOLO nie utrzymuje sztucznego ruchu.
4. Czy działa timeout albo fallback `post_peak`.

Możliwe szybkie poprawki:

```python
stillness_threshold = 0.05
stillness_frames = 3
```

---

## Confidence blokuje segmentację

Objawy:

```text
feature_confidence: 0.22
confidence_ok: False
segmenter_state: idle
```

Znaczenie:

Keypointy są zbyt mało wiarygodne, więc system ignoruje klatki.

Co sprawdzić:

1. Czy całe ciało jest w kadrze.
2. Czy jest dobre światło.
3. Czy kamera nie jest zbyt blisko albo zbyt daleko.
4. Czy ręce i nogi nie są zasłonięte.

Tymczasowa poprawka MVP:

```python
MIN_KEYPOINT_CONFIDENCE = 0.25
```

---

## Złe ćwiczenie ustawione w aplikacji

Objawy:

```text
exercise_type: step_only
front_wrist_velocity: 0.120
front_ankle_velocity: 0.000
motion_signal: 0.000
```

Znaczenie:

Użytkownik rusza ręką, ale aplikacja oczekuje kroku.

Poprawka:

W `main.py` ustaw właściwe ćwiczenie:

```python
exercise_type = "arms_only"
```

---

# Minimalne progi testowe dla MVP

Na czas debugowania można użyć luźniejszych wartości.

```python
DEFAULT_COOLDOWN_SEC = 0.4
MIN_VALID_FRAMES = 3
MIN_TOTAL_MOTION = 0.05
START_CONFIRMATION_FRAMES = 1
MOTION_DEADZONE = 0.01
```

Dla `arms_only`:

```python
"segmenter": {
	"motion_start_threshold": 0.02,
	"stillness_threshold": 0.025,
	"stillness_frames": 3,
	"min_duration_sec": 0.15,
	"max_duration_sec": 4.0,
}
```

Docelowo te progi trzeba zaostrzyć po potwierdzeniu, że cały pipeline działa.

---

# Najważniejsza zasada debugowania

Najpierw sprawdzaj segmentację, dopiero potem scoring.

Jeżeli segmenter nie wykrywa poprawnie `started` i `finished`, wynik oceny nie ma znaczenia, bo scoring dostaje złą
sekwencję albo nie dostaje jej wcale.

Kolejność diagnozy:

1. Czy `exercise_type` jest poprawne.
2. Czy `confidence_ok` jest `True`.
3. Czy podczas ruchu rośnie `motion_signal`.
4. Czy pojawia się `segmenter_event: started`.
5. Czy po ruchu rośnie `stillness_frames`.
6. Czy pojawia się `segmenter_event: finished`.
7. Dopiero wtedy sprawdzaj `score` i `main_feedback`.
