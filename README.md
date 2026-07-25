# lloc — LLM Localization

Консольная утилита локализации приложений через **локальную LLM (Ollama)**.
Читает базовые строки и контекст из TOML-конфигов, переводит только дельту,
валидирует результат, следит за консистентностью терминологии и раскладывает
переводы по форматам платформ: Android `strings.xml`, iOS `.strings`,
плоский JSON для веба (i18next-совместимый).

- **Zero-install**: только стандартная библиотека Python, без pip и venv.
- **Только дельта**: переводятся отсутствующие и устаревшие строки, удалённые
  ключи вычищаются отовсюду автоматически.
- **Git-friendly**: все состояния — текстовые файлы с детерминированным
  порядком ключей, коммитятся.
- **Монорепо**: workspace-режим с наследованием настроек, группами продуктов
  и общим пулом воркеров.

Детали дизайна и форматов — в [SPEC.md](SPEC.md).

## Требования

- **Python 3.11+** (нужен `tomllib`; других зависимостей нет);
- **[Ollama](https://ollama.com)** локально — только для команд, которые
  переводят (`translate`) или строят терм-карту (`verify --terms`);
  `status`, `build`, `verify`, `import` работают без LLM. Модель по умолчанию
  задана в конфиге (`[ollama] model`), при первом запуске подтягивается
  автоматически.

## Quickstart: одиночный проект

Каталог локализации (обычно `myapp/l10n/`) содержит два файла, которые пишет
человек:

```toml
# l10n.toml
[app]
description = "My TV app: IPTV player for Android TV"
tone = "neutral, concise UI language; formal address"
base_language = "en"
languages = ["ru", "de", "ja"]

[output]
adapter = "android-xml"
[output.android-xml]
res_dir = "../app/src/main/res"   # пути — относительно l10n.toml
```

```toml
# strings.toml — источник правды для всех строк
[menu_search]
text = "Search"
context = "Main menu item"
type = "button"
```

Дальше:

```bash
python lloc.py --dir myapp/l10n status       # что нужно перевести (без LLM)
python lloc.py --dir myapp/l10n translate    # перевод дельты + build
```

Если строки уже живут в Android-проекте, `strings.toml` генерируется из
базового `values/strings.xml`:

```bash
python lloc.py --dir myapp/l10n import --bootstrap-strings
```

Хранилище переводов — `locales/*.json` (пишет утилита, коммитится); XML в
`res/` — генерируемый артефакт `build`.

## Quickstart: workspace (монорепо)

Всё workspace-уровневое живёт в одном каталоге `l10n/` в корне репозитория:

```
monorepo/
├── l10n/
│   ├── config.toml        # members + общие дефолты
│   └── player.toml        # группа "player" (любой *.toml кроме config.toml)
├── apps/tv/l10n.toml      # модульные конфиги — в каталогах модулей
├── apps/mobile/l10n.toml
└── libs/core/l10n.toml
```

```toml
# l10n/config.toml
members = ["apps/*", "libs/*"]   # явные глобы, без автосканирования

[app]
tone = "neutral, concise"
base_language = "en"
languages = ["ru", "de"]

[ollama]
workers = 3                      # общий пул на весь прогон
```

```toml
# l10n/player.toml — контекст продуктовой группы (редактирует команда продукта)
context = "TinyPlayer product family: IPTV player apps and a shared core."
```

```toml
# apps/tv/l10n.toml — модуль: наследует дефолты, добавляет своё
group = "player"

[app]
description = "TV variant; remote-control navigation."

[output]
adapter = "android-xml"
[output.android-xml]
res_dir = "res"
```

Запуск из корня обходит все модули агрегированно; `--dir <модуль>` ограничивает
одним модулем (наследование и группа подхватываются автоматически):

```bash
python lloc.py --dir monorepo status                # все модули
python lloc.py --dir monorepo translate             # общий пул воркеров
python lloc.py --dir monorepo/apps/tv status        # один модуль
```

Группа — граница консистентности: few-shot примеры и терминология не
пересекают её; `verify` предупреждает о дублях ключей внутри группы.
Рабочий пример: [`examples/demo-workspace`](examples/demo-workspace)
(одиночный: [`examples/demo`](examples/demo)).

## Команды

Общие опции: `--dir PATH` (каталог с `l10n.toml` или корень workspace,
по умолчанию `.`), `--lang xx[,yy]` (фильтр языков), `--verbose`.

| Команда | Действие | LLM |
|---|---|---|
| `status` | Таблица ok / missing / stale / orphan по языкам. Exit code 1, если есть работа. | нет |
| `translate` | Перевод дельты (missing + stale), чистка удалённых ключей, обновление `locales/` и lock, затем автоматически `build`. Флаги: `--force` (перевести всё заново), `--limit N`, `--dry-run` (план без запросов), `--no-build`, `--model NAME` (A/B-прогон), `--fix-terms` (авто-исправление расхождений терминологии), `--retranslate-model PATTERN` (пере-перевести строки, чьё сохранённое поле `model` матчит fnmatch-паттерн, например `gemma3*` или `imported`). | да |
| `build` | Сгенерировать файлы платформы из `locales/`: `values/strings.xml` (база) + `values-b+xx/`. Особые ключи в существующих XML сохраняются. | нет |
| `verify` | Валидации существующих переводов, консистентность lock/locales, эвристика расхождений терминов, дубли ключей в группе. `--terms` — точная LLM-терм-карта. | только `--terms` |
| `import` | Импорт существующих переводов платформы (`values-b+xx`) в `locales/` и lock; импортированное считается актуальным (`--assume-stale` — пометить устаревшим, `--overwrite` — перезаписать имеющиеся записи в locales), `model` таких записей — `"imported"`. `--bootstrap-strings` — генерация `strings.toml` из базового `values/strings.xml`. В workspace — только per-модуль (`--dir <member>`). | нет |
| `report` | Ревью свежепереведённого: строки с `translated_at` за последнее окно, по языкам/модулям. `--since 90m/24h/7d` (default 24h), `--last` — синоним режима по умолчанию. | нет |
| `set` | Ручное исправление одного перевода: `set <key> --lang de "текст"` — запись в locales с `model="manual"` + инкрементальный `build` языка. Валидация как у translate; `--force-set` — принять несмотря на ошибки. В workspace — только per-модуль (`--dir <member>`). | нет |

Примеры:

```bash
python lloc.py --dir l10n translate --dry-run          # оценить объём работы
python lloc.py --dir l10n translate --lang ru,de       # только два языка
python lloc.py --dir l10n translate --force --lang ja  # пере-перевести язык
python lloc.py --dir l10n verify --terms               # аудит терминологии
```

`python -m lloc` эквивалентен `python lloc.py`.

## Адаптеры

Выбираются `[output] adapter = "..."`, у всех одна дисциплина: управляемые
ключи (из `strings.toml`) перегенерируются при `build`, всё постороннее в
существующих файлах сохраняется; `import` читает существующие переводы;
запись атомарная.

```toml
[output]
adapter = "android-xml"           # values/ + values-b+xx/strings.xml
[output.android-xml]
res_dir = "../app/src/main/res"   # пути — относительно l10n.toml
legacy_codes = { id = "in", he = "iw" }
```

```toml
[output]
adapter = "ios"                   # <код>.lproj/Localizable.strings (UTF-8)
[output.ios]
lproj_dir = "../App/Resources"    # каталог с *.lproj
# filename = "Localizable.strings"  # default
```

```toml
[output]
adapter = "json"                  # плоский <код>.json на язык, ключи сортированы
[output.json]
dir = "../web/src/i18n"
```

Особенности `ios`: каталоги — BCP-47 как есть (`es-419.lproj`, `pt-BR.lproj`,
`zh-Hant.lproj`), базовый язык — `<base>.lproj` (например `en.lproj`;
`Base.lproj` — механизм сторибордов, не строк); `context` строки пишется
комментарием `/* ... */` над записью; плейсхолдеры `%@`/`%1$@` валидируются
наравне с `%s`; `.stringsdict` не поддерживается: плюрали отвергнуты дизайном
(SPEC §12) — норма: числонезависимые базовые тексты («People: %d», а не
«1 person/5 people»). Особенности `json`: без вложенности, точки в ключах —
литеральные. Легаси-коды Android (`in`/`iw`) в именах файлов ios/json
не используются — конвертируются в `id`/`he`.

## Manual fixes — ручные исправления переводов

Канонический ответ на «перевод X на язык Y кривой»:

```bash
python lloc.py --dir myapp/l10n set menu_search --lang de "Suchen"
```

`set` пишет запись в `locales/de.json` с `model = "manual"`, `src` — хеш
текущего базового текста, и сразу инкрементально перестраивает файлы этого
языка. Плейсхолдеры валидируются как при переводе (несовпадение — ошибка;
`--force-set` — принять осознанно).

Правила игры:

- **`--force` значит force**: переписывает всё, включая manual-записи.
  `--retranslate-model` — честный fnmatch по полю `model`: паттерны моделей
  (`gemma*`) manual-записи естественно не матчат, а явный
  `--retranslate-model manual` — матчит (осознанный запрос).
- Если **базовый текст изменился**, manual-запись становится stale и уходит
  на пере-перевод как обычная (исправленный текст относился к старой базе);
  `translate` явно сообщает: «N manual translation(s) were stale and
  retranslated: ...» — стоит глянуть глазами.
- Manual-переводы — **золотые few-shot примеры**: при отборе примеров для
  промпта они приоритетнее обычных, т.е. одно ручное исправление подтягивает
  стиль соседних строк при следующих переводах.
- Прямая правка `locales/<xx>.json` руками тоже валидна (формат записи —
  ниже; поставьте `"model": "manual"`, иначе запись не защищена) — но `set`
  удобнее: валидация + build + метка одним вызовом.
- Сгенерированные файлы (`res/values*/strings.xml`, `.lproj/*.strings`,
  `<lang>.json` адаптера) по-прежнему НЕ редактируются. Если правки уже
  сделаны в платформенных файлах — затащить их: `import --overwrite --lang xx`
  (такие записи получают `model = "imported"`; паттерны вроде
  `--retranslate-model "gemma*"` их не матчат — поле просто не совпадает,
  отдельной защиты у imported нет).

Каждая запись в `locales/*.json` хранит `model` (кем переведена строка;
у импортированных — `"imported"`) и `translated_at` (UTC ISO-8601). На них
опираются `report` (ревью свежего) и `translate --retranslate-model`
(пере-перевод строк конкретной модели после её смены). Записи, сделанные
старыми версиями утилиты без этих полей, остаются валидными.

## Идемпотентность и прерывание

Повторный запуск ничего не ломает и не делает лишней работы: переводится
только дельта, хеши базовых текстов отслеживают устаревание. `translate`
**пишет locale после каждого батча** — процесс можно прерывать (Ctrl+C,
падение сети, перезагрузка) в любой момент: сделанное сохранено, следующий
запуск продолжит с места остановки. Ctrl+C обрабатывается явно: воркеры
останавливаются, зависшие HTTP-запросы не удерживают процесс, выводится
«interrupted — progress saved, rerun to continue», exit code 130, лок
снимается; второй Ctrl+C завершает немедленно. Удаление ключа из `strings.toml`
вычищает его при следующем `translate` отовсюду: из `locales/*.json`, lock,
терм-кешей и (через build) из всех XML; особые ключи XML при этом не
трогаются.

Все хранилища (locales, lock, терм-кеши, XML) пишутся атомарно (temp-файл +
`os.replace`): прерванная запись не оставляет полуфайлов. Параллельные
запуски на одном модуле исключены OS-локом (`.lloc.lock` в каталоге модуля,
`flock`/`msvcrt.locking`): второй процесс сразу получает «another lloc run
is active for this module». Лок умирает вместе с процессом — после kill или
падения НИКАКОЙ ручной чистки не нужно, файл на диске может оставаться
(добавьте `.lloc.lock` в `.gitignore` проекта).

## Правила для CLAUDE.md

Готовый блок для `CLAUDE.md` репозитория, где используется lloc
(путь в первой строке — плейсхолдер, поправьте под свой репо):

```markdown
## Локализация (lloc)

Утилита: `../lloc/lloc.py` относительно корня репо.
Запуск: `python <путь>/lloc.py --dir <каталог l10n> <команда>`.

- **Источник правды — `strings.toml`** (+ `l10n.toml` конфиг). Сгенерированные
  файлы руками НЕ редактировать: ни `locales/*.json`, ни `res/values*/strings.xml`,
  ни `l10n.lock.json` — их пишет утилита, ручные правки будут перезаписаны.
- Пользователь недоволен конкретным переводом («X на немецком звучит криво,
  надо ...») → `set <key> --lang de "текст"`. Это ЕДИНСТВЕННЫЙ правильный
  способ ручной правки: запись помечается `model="manual"`, валидируется,
  сразу попадает в build и дальше служит золотым few-shot примером для
  соседних строк. Модельные паттерны `--retranslate-model` её не матчат;
  `--force` переписывает всё, включая manual.
- Добавил/изменил строку в `strings.toml` → сразу заполни `context`
  (и `type`: button / title / message) — от контекста напрямую зависит
  качество перевода. Затем `status` — мгновенная оценка дельты без LLM.
- `translate` инкрементален: обычный прогон после мелких правок — секунды или
  минуты. Начальный перевод проекта или массовые операции (`--force`, новый
  язык) — долгие: запускай их В ФОНЕ (фоновый bash / сабагент), не блокируя
  основную работу. Это безопасно: утилита идемпотентна, сохраняет прогресс
  после каждого батча и переживает прерывание — повторный запуск продолжит
  с места остановки.
- Оценить объём перед долгой операцией: `translate --dry-run`.
- Перед релизом: `verify` (быстро, без LLM); глубокий аудит терминологии —
  `verify --terms` (нужна Ollama).
- Удаление ключа из `strings.toml` — это ВСЁ, что нужно для удаления строки:
  следующий `translate` сам вычистит её из locales, lock, терм-кешей и XML.
- Особые ключи Android (`translatable="false"`, `<string-array>`, `<plurals>`)
  живут прямо в XML, утилитой не управляются и переживают `build` нетронутыми —
  их редактируй в XML как обычно.
- В монорепо с workspace-режимом (`l10n/config.toml` с `members`): команды из
  корня обходят все модули; `--dir <модуль>` — один модуль. Терминология и
  примеры переводов общие внутри группы модулей (`group = "..."`).
```

## Тесты

```bash
python -m unittest discover -s tests -t .
```
