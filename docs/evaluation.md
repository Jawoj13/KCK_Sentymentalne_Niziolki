# Dokumentacja modułu `evaluation`

## 1. Cel modułu

Moduł `evaluation` odpowiada za pełną ocenę ćwiczeń wykonywanych przez użytkownika na podstawie keypointów wykrytych
przez YOLO Pose. System analizuje ruch z kamer, dzieli go na pojedyncze powtórzenia, wyciąga cechy biomechaniczne,
porównuje wykonanie z nagraniem referencyjnym oraz generuje wynik liczbowy i jeden najważniejszy komunikat zwrotny.

System obsługuje trzy typy ćwiczeń:

- `arms_only` - wyprost rąk z bronią bez kroku.
- `step_only` - sam krok w postawie szermierczej.
- `full` - pełne ćwiczenie łączące wyprost rąk i krok.

Głównym założeniem modułu jest to, że wynik nie jest liczony wyłącznie z podobieństwa surowych punktów ciała. System
używa interpretowalnych cech ruchu, takich jak kąty stawów, długość kroku, pochylenie tułowia, szerokość pozycji i
synchronizacja ręki ze stopą. Dzięki temu możliwe jest nie tylko obliczenie wyniku `0-100`, ale też wskazanie
konkretnego błędu technicznego.

---

## 2. Architektura katalogu

```text
evaluation/
├── __init__.py
├── exercise_config.py
├── geometry.py
├── features.py
├── segmenter.py
├── scoring.py
├── dtw.py
├── feedback.py
└── session.py
```

### Odpowiedzialności plików

- `exercise_config.py` zawiera konfigurację ćwiczeń, progi, wagi scoringu, typy kamer, mapowanie keypointów i komunikaty
  błędów.
- `geometry.py` zawiera funkcje matematyczne niezależne od ćwiczeń.
- `features.py` przekształca keypointy YOLO na cechy biomechaniczne.
- `segmenter.py` wykrywa początek i koniec pojedynczego powtórzenia.
- `scoring.py` liczy wynik pojedynczego powtórzenia.
- `dtw.py` porównuje sekwencję użytkownika z sekwencją referencyjną przez Dynamic Time Warping.
- `feedback.py` wybiera najważniejszy błąd do pokazania użytkownikowi.
- `session.py` zarządza całą serią ćwiczeń i podsumowaniem po zadanej liczbie powtórzeń.

---

## 3. Przepływ danych w systemie

```text
Kamera boczna / kamera frontalna
        │
        ▼
YOLO Pose
        │
        ▼
Keypointy [17, 2] albo [17, 3]
        │
        ▼
features.extract_frame_features()
        │
        ▼
Cechy biomechaniczne jednej klatki
        │
        ▼
segmenter.RepetitionSegmenter.update()
        │
        ▼
Pojedyncze wykryte powtórzenie
        │
        ▼
scoring.evaluate_repetition()
        │
        ▼
Wynik powtórzenia + lista błędów
        │
        ▼
session.TrainingSession.add_repetition_result()
        │
        ▼
Podsumowanie serii + jeden główny feedback
```

Moduł `backend.py` nie powinien zawierać logiki oceniania. Jego zadaniem jest pobranie klatek z kamer, uruchomienie YOLO
Pose i przekazanie keypointów do modułu `evaluation`.

---

## 4. Typy kamer

System rozróżnia trzy źródła metryk:

```python
CAMERA_SIDE = "side"
CAMERA_FRONT = "front"
CAMERA_ANY = "any"
```

### Kamera boczna: `side`

Kamera boczna służy do oceny ruchu w przód i tył. Z tej kamery liczone są przede wszystkim:

- wyprost ręki do przodu,
- długość kroku,
- pochylenie tułowia,
- synchronizacja ręki i stopy,
- moment rozpoczęcia ruchu ręką,
- moment rozpoczęcia ruchu stopą,
- powrót do pozycji wyjściowej w osi przód-tył.

### Kamera frontalna: `front`

Kamera frontalna służy do oceny błędów bocznych. Z tej kamery liczone są przede wszystkim:

- zapadanie kolana do środka,
- zwężanie pozycji,
- ustawienie stóp na jednej linii,
- stabilność boczna.

### Dowolna kamera: `any`

Metryki oznaczone jako `any` mogą być liczone z dowolnej kamery, jeżeli wymagane keypointy mają wystarczające
confidence. Dotyczy to głównie prostych kątów stawów, takich jak kąt łokcia lub kąt kolana.

---

## 5. `exercise_config.py`

Plik `exercise_config.py` przechowuje konfigurację systemu oceniania. Nie zawiera logiki obliczeniowej.

### Główne elementy

```python
EXERCISE_ARMS_ONLY = "arms_only"
EXERCISE_STEP_ONLY = "step_only"
EXERCISE_FULL = "full"
```

```python
YOLO_KEYPOINTS = {
	"left_shoulder": 5,
	"right_shoulder": 6,
	"left_elbow": 7,
	"right_elbow": 8,
	"left_wrist": 9,
	"right_wrist": 10,
	"left_hip": 11,
	"right_hip": 12,
	"left_knee": 13,
	"right_knee": 14,
	"left_ankle": 15,
	"right_ankle": 16,
}
```

### `EXERCISE_CONFIG`

`EXERCISE_CONFIG` definiuje dla każdego ćwiczenia:

- wymagane kamery,
- wagi komponentów wyniku,
- progi biomechaniczne,
- progi segmentacji,
- limity czasu powtórzenia.

Przykładowo dla `arms_only` konfiguracja zawiera:

- minimalny akceptowalny kąt łokcia,
- próg wykrycia zamachu,
- maksymalną zmianę pochylenia tułowia,
- wagi: wyprost ręki, brak zamachu, stabilność tułowia, podobieństwo do wzorca.

### `ERROR_PRIORITIES`

`ERROR_PRIORITIES` określa, które błędy są ważniejsze przy wyborze głównego feedbacku. Przykład:

```python
ERROR_PRIORITIES = {
	"step_only": {
		"KNEE_COLLAPSES_INWARD": 1.6,
		"BACK_LEG_STRAIGHT": 1.4,
		"STANCE_NARROWS": 1.2,
		"STEP_TOO_SHORT": 1.0,
		"TORSO_LEANS_FORWARD": 0.9,
	}
}
```

Błąd z wyższym priorytetem może zostać pokazany użytkownikowi nawet wtedy, gdy wystąpił rzadziej niż mniej istotny błąd.

### `ERROR_MESSAGES`

`ERROR_MESSAGES` mapuje kody błędów na komunikaty dla użytkownika.

Przykład:

```python
"BACK_LEG_STRAIGHT": "Nie prostuj całkowicie tylnej nogi."
```

---

## 6. `geometry.py`

Plik `geometry.py` zawiera niskopoziomowe funkcje matematyczne. Funkcje z tego pliku nie wiedzą nic o ćwiczeniach,
kamerach ani feedbacku.

### `angle(a, b, c)`

Liczy kąt `ABC`, gdzie punkt `b` jest wierzchołkiem kąta.

Użycie:

```python
elbow_angle = angle(shoulder, elbow, wrist)
knee_angle = angle(hip, knee, ankle)
```

### `distance(a, b)`

Liczy odległość euklidesową między dwoma punktami.

Używana między innymi do:

- długości segmentów ciała,
- przemieszczenia nadgarstka,
- przemieszczenia kostki,
- normalizacji cech względem skali ciała.

### `center(a, b)`

Zwraca środek dwóch punktów. Używana do obliczania:

- środka barków,
- środka bioder,
- osi tułowia.

### `point_line_distance(point, line_a, line_b)`

Liczy odległość punktu od prostej. Używana do oceny, czy kolano nogi wykrocznej ucieka względem linii biodro-kostka.

### `safe_divide(numerator, denominator, default)`

Chroni przed dzieleniem przez zero. Używana we wszystkich metrykach normalizowanych.

### `clamp(value, minimum, maximum)`

Ogranicza wartość do zakresu. Używana głównie przy score i severity.

---

## 7. `features.py`

Plik `features.py` odpowiada za przekształcenie keypointów YOLO w cechy biomechaniczne.

### `extract_frame_features(keypoints, camera_view, dominant_side, timestamp)`

Przetwarza jedną klatkę.

Wejście:

```python
keypoints  # tablica [17, 2] albo [17, 3]
camera_view  # "side" albo "front"
dominant_side  # "left" albo "right"
timestamp  # czas klatki w sekundach
```

Wyjście:

```python
{
	"timestamp": 12.34,
	"camera_view": "side",
	"dominant_side": "left",
	"front_elbow_angle": 172.5,
	"rear_elbow_angle": 148.0,
	"front_knee_angle": 141.3,
	"back_knee_angle": 166.8,
	"torso_lean_angle": 11.2,
	"front_knee_line_error": 0.06,
	"stance_width": 1.05,
	"wrist_extension": 0.91,
	"front_wrist": [x, y],
	"front_ankle": [x, y],
	"hip_center": [x, y],
	"shoulder_center": [x, y],
	"feature_confidence": 0.86,
	"confidence_ok": True,
}
```

### `extract_sequence_features(frames, camera_view, dominant_side)`

Przetwarza listę klatek i dodaje cechy czasowe.

Cechy czasowe:

```python
"front_wrist_velocity"
"front_ankle_velocity"
"wrist_extension_change"
"front_ankle_displacement"
"front_wrist_displacement"
"hip_center_displacement"
```

### Dominująca strona

System zakłada, że dominująca strona jest ustawiana w UI. Na jej podstawie wyznaczane są:

```python
front_side
rear_side
front_arm
rear_arm
front_leg
back_leg
```

Przykład:

Jeżeli `dominant_side = "left"`, to:

```python
front_side = "left"
rear_side = "right"
```

---

## 8. `segmenter.py`

Plik `segmenter.py` odpowiada za wykrywanie pojedynczego powtórzenia.

### Klasa `RepetitionSegmenter`

```python
segmenter = RepetitionSegmenter(
	exercise_type="full",
	dominant_side="left",
)
```

### Stany segmentera

```text
idle
recording_attack
hold_or_return
finished
```

#### `idle`

System czeka na rozpoczęcie ruchu.

#### `recording_attack`

System zapisuje klatki od momentu wykrycia startu powtórzenia.

#### `hold_or_return`

System obserwuje krótkie zatrzymanie pozycji końcowej lub powrót do pozycji wyjściowej.

#### `finished`

Powtórzenie jest zakończone i gotowe do oceny.

### `update(timestamp, side_features, front_features=None)`

Aktualizuje segmenter nowymi cechami z kamer.

Zwracane wartości:

```python
None  # brak zmiany stanu
"started"  # wykryto start powtórzenia
"finished"  # wykryto koniec powtórzenia
```

### Sygnały ruchu

Dla `arms_only`:

```python
motion_signal = wrist_velocity + wrist_extension_change
```

Dla `step_only`:

```python
motion_signal = front_ankle_velocity
```

Dla `full`:

```python
motion_signal = 0.5 * front_wrist_velocity + 0.5 * front_ankle_velocity
```

### `get_repetition()`

Zwraca gotowe powtórzenie:

```python
{
	"exercise_type": "full",
	"dominant_side": "left",
	"finish_reason": "stillness",
	"records": [...],
	"side_sequence": [...],
	"front_sequence": [...],
}
```

---

## 9. `dtw.py`

Plik `dtw.py` odpowiada za porównanie wykonania użytkownika z nagraniem referencyjnym.

DTW rozwiązuje problem różnego tempa wykonania. Użytkownik może wykonać ruch szybciej lub wolniej niż osoba na nagraniu
wzorcowym, a system nadal dopasuje odpowiadające sobie fazy ruchu.

### `compute_dtw_score(user_features, reference_features, feature_names)`

Porównuje sekwencję użytkownika z sekwencją referencyjną.

Wejście:

```python
user_features  # lista cech użytkownika
reference_features  # lista cech z poprawnego nagrania
feature_names  # lista nazw cech używanych do porównania
```

Przykład dla `arms_only`:

```python
feature_names = [
	"front_elbow_angle",
	"rear_elbow_angle",
	"torso_lean_angle",
	"wrist_extension",
]
```

Przykład dla `step_only`:

```python
feature_names = [
	"front_knee_angle",
	"back_knee_angle",
	"torso_lean_angle",
	"front_ankle_displacement",
]
```

Przykład dla `full`:

```python
feature_names = [
	"front_elbow_angle",
	"front_knee_angle",
	"back_knee_angle",
	"torso_lean_angle",
	"wrist_extension",
	"front_ankle_displacement",
]
```

### Wynik DTW

Funkcja zwraca wynik w skali `0-100`.

```python
100  # bardzo podobne do wzorca
0  # bardzo dalekie od wzorca
```

DTW nie generuje głównego feedbacku. Feedback pochodzi z reguł biomechanicznych, ponieważ są bardziej interpretowalne.

---

## 10. `scoring.py`

Plik `scoring.py` liczy wynik pojedynczego powtórzenia.

### `evaluate_repetition()`

Główna funkcja oceny.

```python
evaluate_repetition(
	exercise_type,
	side_sequence,
	front_sequence,
	dominant_side,
	reference_template,
)
```

Zwraca:

```python
{
	"exercise_type": "full",
	"score": 84.5,
	"component_scores": {
		"arm_extension": 90.0,
		"step_quality": 82.0,
		"synchronization": 75.0,
		"posture": 88.0,
		"dtw_reference": 86.0
	},
	"errors": [...],
	"main_feedback": {...},
	"confidence": 0.91
}
```

### `score_arms_only()`

Ocenia wyprost ręki bez kroku.

Komponenty:

```python
arms_only_score = (
		0.50 * front_elbow_extension_score
		+ 0.25 * no_windup_score
		+ 0.15 * torso_stability_score
		+ 0.10 * dtw_reference_score
)
```

Błędy:

- `ARM_NOT_EXTENDED`
- `WINDUP`
- `WEAPON_TOO_HIGH_OR_LOW`
- `TORSO_LEANS_FORWARD`

### `score_step_only()`

Ocenia sam krok.

Komponenty:

```python
step_only_score = (
		0.25 * step_length_score
		+ 0.25 * front_knee_alignment_score
		+ 0.20 * back_knee_bend_score
		+ 0.15 * torso_posture_score
		+ 0.10 * stance_width_score
		+ 0.05 * dtw_reference_score
)
```

Błędy:

- `KNEE_COLLAPSES_INWARD`
- `BACK_LEG_STRAIGHT`
- `STANCE_NARROWS`
- `STEP_TOO_SHORT`
- `TORSO_LEANS_FORWARD`

### `score_full()`

Ocenia pełne ćwiczenie.

Komponenty:

```python
full_score = (
		0.30 * arm_extension_score
		+ 0.20 * step_quality_score
		+ 0.25 * synchronization_score
		+ 0.15 * posture_score
		+ 0.10 * dtw_reference_score
)
```

Błędy:

- `FOOT_STARTS_BEFORE_HAND`
- `HAND_FOOT_NOT_SYNCED`
- `ARM_NOT_EXTENDED`
- `UNSTABLE_LEGS`
- `TORSO_LEANS_FORWARD`

---

## 11. `feedback.py`

Plik `feedback.py` odpowiada za wybór jednego głównego komunikatu dla użytkownika.

System nie pokazuje wielu błędów naraz. Po serii ćwiczeń wybierany jest jeden najważniejszy błąd, żeby użytkownik
wiedział, na czym ma się skupić.

### `choose_main_feedback(error_history, exercise_type)`

Wybiera błąd na podstawie wzoru:

```python
error_score = priority_weight * severity_mean * occurrence_count
```

Gdzie:

- `priority_weight` pochodzi z `ERROR_PRIORITIES`,
- `severity_mean` oznacza średnią powagę błędu,
- `occurrence_count` oznacza liczbę wystąpień błędu w serii.

Przykład wyniku:

```python
{
	"code": "BACK_LEG_STRAIGHT",
	"message": "Nie prostuj całkowicie tylnej nogi.",
	"camera": "side",
	"score": 3.2
}
```

---

## 12. `session.py`

Plik `session.py` zarządza całą serią ćwiczeń.

### `TrainingSession`

```python
session = TrainingSession(
	exercise_type="step_only",
	target_repetitions=10,
	dominant_side="left",
)
```

### `add_repetition_result(result)`

Dodaje wynik jednego powtórzenia do sesji.

### `is_complete()`

Zwraca `True`, gdy użytkownik wykonał wymaganą liczbę powtórzeń.

### `get_summary()`

Zwraca podsumowanie serii:

```python
{
	"exercise_type": "step_only",
	"target_repetitions": 10,
	"repetitions_done": 10,
	"average_score": 78.4,
	"best_score": 91.0,
	"worst_score": 62.0,
	"main_feedback": {
		"code": "BACK_LEG_STRAIGHT",
		"message": "Nie prostuj całkowicie tylnej nogi.",
		"camera": "side"
	},
	"results": [...]
}
```

---

## 13. Progi i interpretacja metryk

### Kąt łokcia

```python
elbow_angle = angle(shoulder, elbow, wrist)
```

Interpretacja:

```text
175-180°: bardzo dobry wyprost
165-175°: dobry wyprost
150-165°: częściowy wyprost
<150°: ręka za bardzo zgięta
```

### Kąt kolana

```python
knee_angle = angle(hip, knee, ankle)
```

Interpretacja:

```text
175-180°: noga prawie całkowicie wyprostowana
165-175°: prawie prosta, ostrzeżenie dla tylnej nogi
130-165°: aktywne, lekkie ugięcie
100-130°: mocne ugięcie
<90°: bardzo głęboko
```

### Pochylenie tułowia

```python
torso_lean_angle = angle_between(torso_vector, vertical_axis)
```

Interpretacja:

```text
0-10°: bardzo dobrze
10-20°: akceptowalne
20-30°: zbyt duże pochylenie
>30°: błąd
```

### Zapadanie kolana do środka

```python
front_knee_line_error = distance(knee, line(hip, ankle)) / leg_length
```

Interpretacja:

```text
0.00-0.08: dobrze
0.08-0.15: ostrzeżenie
>0.15: kolano ucieka
```

### Szerokość pozycji

```python
stance_width = abs(front_ankle_x - back_ankle_x) / shoulder_width
```

Interpretacja:

```text
>= 0.75 * stance_width_start: dobrze
< 0.75 * stance_width_start: pozycja się zwęża
```

---

## 14. Przykładowa integracja z backendem

Backend powinien przekazywać do `evaluation` dane w formacie:

```python
pose_packet = {
	"camera": "side",
	"timestamp": timestamp,
	"keypoints": keypoints,
}
```

Przykładowy przepływ:

```python
from features import extract_frame_features
from segmenter import RepetitionSegmenter
from scoring import evaluate_repetition
from evaluation import TrainingSession

segmenter = RepetitionSegmenter("full", dominant_side="left")
session = TrainingSession("full", target_repetitions=10, dominant_side="left")

side_features = extract_frame_features(
	side_keypoints,
	camera_view="side",
	dominant_side="left",
	timestamp=timestamp,
)

front_features = extract_frame_features(
	front_keypoints,
	camera_view="front",
	dominant_side="left",
	timestamp=timestamp,
)

event = segmenter.update(timestamp, side_features, front_features)

if event == "finished":
	repetition = segmenter.get_repetition()

	result = evaluate_repetition(
		exercise_type=repetition["exercise_type"],
		side_sequence=repetition["side_sequence"],
		front_sequence=repetition["front_sequence"],
		dominant_side=repetition["dominant_side"],
		reference_template=reference_template,
	)

	session.add_repetition_result(result)
	segmenter.reset()

	if session.is_complete():
		summary = session.get_summary()
		print(summary["average_score"])
		print(summary["main_feedback"])
```

---

## 15. Format wyniku pojedynczego powtórzenia

Każde ocenione powtórzenie zwraca wynik w takim formacie:

```python
{
	"exercise_type": "arms_only",
	"score": 87.5,
	"component_scores": {
		"front_elbow_extension": 90.0,
		"no_windup": 100.0,
		"torso_stability": 80.0,
		"dtw_reference": 85.0
	},
	"errors": [
		{
			"code": "TORSO_LEANS_FORWARD",
			"severity": 0.3,
			"camera": "side",
			"message": "Utrzymaj bardziej wyprostowany tułów.",
			"priority": 0.9
		}
	],
	"main_feedback": {
		"code": "TORSO_LEANS_FORWARD",
		"message": "Utrzymaj bardziej wyprostowany tułów.",
		"camera": "side",
		"score": 0.27
	},
	"confidence": 0.91
}
```

---

## 16. Format podsumowania serii

Po wykonaniu zadanej liczby powtórzeń sesja zwraca:

```python
{
	"exercise_type": "full",
	"target_repetitions": 10,
	"repetitions_done": 10,
	"average_score": 81.2,
	"best_score": 93.0,
	"worst_score": 64.0,
	"main_feedback": {
		"code": "HAND_FOOT_NOT_SYNCED",
		"message": "Spróbuj zakończyć wyprost ręki i krok w tym samym momencie.",
		"camera": "side"
	},
	"results": [...]
}
```

---

## 17. Zasady projektowe

1. `backend.py` nie zawiera logiki oceniania.
2. Wszystkie progi i wagi są trzymane w `exercise_config.py`.
3. Funkcje matematyczne są niezależne od ćwiczeń.
4. Segmentacja nie liczy wyniku, tylko wykrywa powtórzenie.
5. Scoring ocenia jedno powtórzenie.
6. Session zarządza serią powtórzeń.
7. Feedback pokazuje jeden najważniejszy błąd.
8. DTW wpływa na wynik, ale nie jest jedynym źródłem oceny.
9. Confidence keypointów wpływa na wiarygodność wyniku.
10. Kamera boczna i frontalna mają osobne odpowiedzialności.

---

## 18. Najważniejsze ograniczenia systemu

- Ocena zależy od jakości detekcji YOLO Pose.
- Jeżeli użytkownik stoi poza kadrem, wynik może być niewiarygodny.
- Metryki kamery bocznej zakładają, że oś ruchu jest zgodna z osią obrazu albo wcześniej wyznaczono `forward_axis`.
- Zapadanie kolana do środka powinno być oceniane tylko z kamery frontalnej.
- Długość kroku powinna być oceniana tylko z kamery bocznej.
- DTW porównuje wykonanie do konkretnego wzorca, więc jakość nagrania referencyjnego bezpośrednio wpływa na wynik.

---

## 19. Docelowe zachowanie aplikacji

Po rozpoczęciu treningu użytkownik wybiera:

- typ ćwiczenia,
- liczbę powtórzeń,
- dominującą stronę.

Aplikacja następnie:

1. analizuje keypointy z kamer,
2. wykrywa pojedyncze powtórzenia,
3. ocenia każde powtórzenie w skali `0-100`,
4. zapisuje błędy i ich severity,
5. po zakończeniu serii pokazuje średni wynik,
6. wybiera jeden najważniejszy komunikat do poprawy.

Przykład komunikatu końcowego:

```text
Wynik: 78/100
Najważniejsza rzecz do poprawy: Nie prostuj całkowicie tylnej nogi.
```
