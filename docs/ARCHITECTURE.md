# Устройство сервиса (для разработки)

Неочевидные места кода, которые нужно знать перед доработками. Правила, которые нельзя ломать, — в `CLAUDE.md`,
история решений и планы — в `docs/PROJECT_MEMORY.md`.

## Чек-листы: база и шаблон

- Рабочий чек-лист каждой клиники хранится в базе (`scenarios`, `criteria`, `clinics.weights_json`,
  `clinics.ai_context`), читает его `load_clinic_checklists()`. `config/checklists.yaml` — только шаблон. Он
  применяется в трёх случаях:
  - при запуске к клинике, у которой ещё нет разделов (`seed_clinic_checklist`);
  - при создании клиники «из шаблона» (`/admin/clinics`);
  - по кнопке «Сбросить к шаблону» (`/settings/checklist/reset`). Пункты обновляются по коду, пункты клиники,
    которых нет в шаблоне, отключаются; оценки и сведения для ИИ остаются.
- Поэтому правка подсказок в YAML **не меняет** чек-лист уже работающей клиники. Изменение вносится в редакторе
  критериев или сбросом к шаблону (правки клиники при этом пропадут).
- `clinic_context` из шаблона попадает только в первую клинику (`replace_context=not clinic.ai_context`).
  Клиника, созданная в админке, получает пустые «сведения для ИИ».

## Промпты и кэширование (`evaluation.py`, `llm.py`)

- `_CLASSIFY_RULES` и `_EVAL_RULES` заполняются через `.format()`: фигурные скобки в тексте промпта удваиваются
  (`{{ }}`, как в примерах JSON).
- Оценка: `system_parts = [правила + сведения о клинике, «Чек-лист для оценки» (общие правила + сценарий)]`,
  в `user` — название сценария, данные звонка (`_call_facts`: направление, дата и время, длительность, вид записи)
  и расшифровка. Классификация — одна system-часть, `max_tokens=2000`, модель `LLM_CLASSIFY_MODEL`. Пункты с
  `evaluable=False` в промпт не попадают. Аудио, номер телефона и имя сотрудника в ИИ не уходят.
- `OpenRouterClient` ставит `cache_control: ephemeral` на **последнюю** system-часть (отключается
  `LLM_PROMPT_CACHE=false`). Всё, что меняется от звонка к звонку, — в `user`; постоянное для клиники (например,
  будущие примеры исправлений) — в system, иначе кэш не сработает.
- Разбор ответа (`Evaluator._build`): пропущенный пункт — «na» с предупреждением; зоны роста только по пунктам
  «no», сортировка по весу, не больше 3; сильные стороны по пунктам «no» отбрасываются, не больше 5; при ответе не в
  JSON — один повтор (`_ask_json`).
- В тексте `_EVAL_RULES` упомянуты коды шаблона (G06 в правиле про сильные стороны, G01/G07/IN11 в примере JSON).
  У клиники с другими кодами эти упоминания не совпадут с её пунктами.
- `temperature` не передавать: `test_llm` проверяет, что его нет.

## Где лежит результат оценки (важно для «исправить оценку»)

- Оценка хранится в двух местах: `Call.evaluation_json` (вердикты, summary, strengths, improvements, warnings) и
  строки `CriterionResult` (по одной на оцениваемый пункт, со снимком номера, текста, раздела и веса). Баллы — в
  `Call.score_total / score_general / score_scenario`.
- Карточка (`card_sections`) берёт пункты из `CriterionResult`, обратную связь — из `evaluation_json`. Сводка,
  «частые ошибки» и Excel читают только `CriterionResult` и `Call.score_*` звонков со `status=done`.
- Баллы считает `weighted_score` в `Processor._save_evaluation` по весам на момент оценки. При ручной правке
  вердикта пересчитывать по `CriterionResult.weight` (текущий чек-лист мог измениться). Зоны роста в
  `evaluation_json` отфильтрованы под вердикты «no» — после правки могут противоречить пункту.
- Любая переоценка, смена типа или повторная расшифровка (`requeue` → `_save_evaluation`) делает
  `call.results.clear()` и перезаписывает `evaluation_json`. Ручные исправления хранить отдельно (своя таблица).

## Миграции базы

При каждом запуске `make_session_factory` выполняет по порядку: `create_all` (только новые таблицы) →
`_add_missing_columns` → `_ensure_calls_autoincrement` (однократная перестройка `calls`) → `_add_unique_indexes`.

- **Новая таблица** создаётся сама.
- **Новая колонка**: достаточно добавить в модель — `ALTER TABLE ADD COLUMN` выполнится сам. Значение в старых
  строках появится только при скалярном `default` (bool, число, строка); при `default=datetime.now` или без
  default будет NULL, NOT NULL в старой базе не ставится. Код должен выдерживать None.
- **Новый индекс** (`Index` в `__table_args__`) в существующей таблице `create_all` **не создаёт** — нужен явный
  `CREATE INDEX IF NOT EXISTS` по образцу `_add_unique_indexes` (исключение — `calls`: при перестройке её индексы
  создаются заново).
- Миграцию закрепляйте тестом: база старой схемы через sqlite3 (или `dialect_options` модели), затем
  `make_session_factory` (пример — `test_old_database_gets_autoincrement`).

## Данные и настройки

- `data/` (меняется через `DATA_DIR`):
  - `ocenka.db` и рядом `ocenka.db-wal`, `ocenka.db-shm` (режим WAL);
  - `audio/ГГГГММ/<uuid>.<расширение>` — записи; `Call.audio_path` — путь относительно `data/audio`;
  - `models/` — GigaAM, Silero VAD, TitaNet (`MODEL_FILES` в `transcription.py`);
  - `secret.key` — ключ подписи cookie входа (потеря = все пользователи выйдут).
  Ключ одноразовых сообщений (`_FLASH_KEY`) — только в памяти процесса. Ключи Mango — в таблице `clinics`
  открытым текстом.
- Резервную копию при работающем сервисе нельзя делать простым копированием `ocenka.db` (свежие данные в `-wal`):
  `sqlite3.Connection.backup()` или `VACUUM INTO`.
- Новая переменная окружения добавляется в четыре места: поле `Settings`, строка в `Settings.from_env()`,
  `.env.example` и таблица в README. `load_dotenv(override=False)`: системные переменные важнее `.env`.
- Переменные, которых нет в `.env.example`: `DATA_DIR`, `CHECKLISTS_PATH`, `HOST` (по умолчанию `0.0.0.0` —
  сервис открыт для локальной сети), `OPENROUTER_BASE_URL`, `LLM_MAX_TOKENS` (12000), `LLM_PROMPT_CACHE`,
  `LLM_TIMEOUT_SEC` (240), `MODELS_BASE_URL`, `WORKER_ENABLED`. `OPEN_BROWSER=1` выставляет `start.bat`.
- `MANGO_TIME_SHIFT_HOURS` действует только при создании первой клиники (`utc_offset = 3 + сдвиг`); дальше
  часовые пояса — в настройках клиники (`utc_offset`, `mango_utc_offset`).

## Фоновые потоки и время

- Потоков два (`daemon=True`), создаются при `WORKER_ENABLED` (по умолчанию true) и запускаются в lifespan после
  `requeue_interrupted` (возврат прерванных звонков в очередь):
  - `Worker` — обрабатывает по одному звонку; клиники по очереди, внутри клиники сначала ручные загрузки, потом
    Mango; звонки отключённых клиник ждут;
  - `MangoScheduler` — раз в 60 с проверяет клиники (`due_clinics`); «Загрузить сейчас» запускает `run_clinic`
    в отдельном потоке. Пауза 2 с между `stats/request` соблюдается внутри одного `MangoClient` (одного прогона
    клиники); между клиниками с общей АТС и параллельными ручными запусками не согласована — выручают повторы при 429.
- Итог обработки записывает `Processor._finish` (короткая транзакция с `_claim`; звонок изменили — `Superseded`).
  Оповещения (будущий Telegram) отправлять после commit, вне транзакции и в try/except.
- Все `datetime` наивные: `started_at` — местное время клиники (из Mango: UTC + `clinic.utc_offset`; из имени
  файла: сдвиг `utc_offset − mango_utc_offset`); `created_at`, `processed_at`, `mango_last_run` — локальное время
  компьютера (`datetime.now()`). Фильтр периода сравнивает `datetime.now()` с `coalesce(started_at, created_at)`.

## Тесты ИИ и фоновых потоков

- `FakeLLM` (`tests/conftest.py`) узнаёт классификацию по фразе «определи его тип» в system (так же их отделяют
  подмены в `test_queue_and_card.py` и `test_sync_and_races.py`). Коды пунктов ищет регуляркой
  `^([A-Z]{1,2}\d{2}(?:_\d)?) \[` по строкам `_format_checklist` («G01 [значимость: …]»). Меняете фразу или формат
  строки чек-листа в промпте — правьте FakeLLM и подмены. Коды из 3+ букв, кириллицы или 3+ цифр FakeLLM не видит.
- Тесты ожидают ровно 2 system-части (`system_parts[1]` — чек-лист) и строку «Направление звонка: исходящий» в
  `user` (`test_mango.py`).
- В фикстуре `settings` `worker_enabled=False`, `TestClient` без `with`: lifespan, Worker, MangoScheduler и
  `requeue_interrupted` не запускаются, звонки обрабатывает `process_all(app)`. Mango проверяется через
  `MangoScheduler(settings, sf, http=http_for(FakeMango(...)))` и `run_clinic(id, now=NOW)`; у `MangoClient` в тестах
  паузы обнулены (`poll_interval=0, request_gap=0, retry_delays=(0, 0)`).
- Здесь разработка на Python 3.13, у заказчика — 3.12: возможности новее 3.12 не использовать.

## Windows

- Файлы читать и писать с явным `encoding="utf-8"` (по умолчанию в Windows — cp1251); текст звонков из `.txt`
  читается с запасным вариантом cp1251.
- `start.bat` — CRLF (`.gitattributes`), UTF-8, `chcp 65001`, использует `py -3`, если он есть.
