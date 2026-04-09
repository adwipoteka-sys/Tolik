# Tolik v3.140 — что добавлено

Этот релиз реализует следующий сильный шаг после v3.139:

**shadow consensus scoring для более сложных LLM-задач**.

Теперь Tolik сравнивает live-ответы с shadow-reference не только по поверхностному текстовому agreement, но и через task-aware consensus нескольких shadow-провайдеров.

## Что именно внедрено

### 1. Новый task-aware consensus scorer

Добавлен модуль:

- `interfaces/shadow_consensus.py`

Он выбирает профиль сравнения в зависимости от задачи:

- `classification`
- `json`
- `qa`
- `summary`
- `freeform`
- `quantum`

То есть теперь для классификации сравниваются метки, для JSON — структура и значения, для summary — более подходящие агрегированные признаки, а не один общий text-overlap для всего.

### 2. Multi-shadow consensus вместо доверия одному shadow

Если доступно несколько shadow-провайдеров, Tolik теперь:

- считает pairwise agreement между shadow-ответами,
- выбирает consensus provider,
- оценивает `consensus_support`,
- оценивает `consensus_strength`,
- и только потом сравнивает live-ответ с этим consensus.

Это существенно надёжнее для сложных LLM-задач и разных форматов ответа.

### 3. Post-promotion drift guard стал consensus-aware

`interfaces/post_promotion_monitor.py` переподключен к новому consensus-слою.

Теперь post-promotion drift monitoring умеет ловить сценарии вида:

- формально похожий текст,
- но уже неправильная label/структура/смысл,
- и автоматически переводит live на safe fallback.

### 4. Canary rollout / re-qualification тоже стал consensus-aware

`interfaces/canary_rollout.py` теперь использует не только fallback, но и дополнительный reference pool из qualified провайдеров.

В canary samples добавлены:

- `comparison_profile`
- `consensus_provider`
- `consensus_support`
- `consensus_strength`

### 5. Ledger расширен новым audit trail

`memory/goal_ledger.py` теперь хранит:

- `interface_shadow_consensus/`

Значит теперь audit trail покрывает qualification → rollout → canary → drift → demotion → shadow consensus.

### 6. Усилены mock-провайдеры для harder LLM cases

`interfaces/provider_registry.py` расширен сценариями:

- `classify_risk`
- `structured_extract`
- `json_extract`

Также добавлен `mock_live_safe_alt`, который форматирует ответы иначе, но сохраняет тот же смысл. Это нужно, чтобы проверить именно semantic consensus, а не банальный string equality.

### 7. Новый demo и новые config examples

Добавлены:

- `provider_shadow_consensus_demo.py`
- `configs/provider_catalog.shadow_consensus.example.json`
- `configs/operator_charter.shadow_consensus.example.json`

## Что подтверждено

- `pytest`: **116 passed**
- `tests/test_v3140_shadow_consensus_scoring.py`: passed
- `provider_shadow_consensus_demo.py`: passed
- `provider_canary_demo.py`: passed
- `provider_shadow_demo.py`: passed
- `provider_rollout_demo.py`: passed
- `autonomous_agi.py` с canary-конфигом: passed

## Ключевой подтвержденный сценарий

Регрессирующий live LLM на задаче `classify_risk` сначала проходит qualification и уходит в live, но затем на реальном запросе начинает выдавать неправильную метку риска.

При этом два safe shadow-провайдера продолжают согласованно выдавать правильный label.

Consensus-слой фиксирует:

- `comparison_profile = classification`
- `consensus_support = 2`
- `live_agreement_score = 0.0`

После этого drift guard автоматически снимает регрессирующего live-провайдера и промотирует safe fallback.

## Следующий сильнейший шаг

После v3.140 следующий сильный шаг:

**cost-aware provider routing между fallback-кандидатами**.

## v3.141 — cost-aware provider routing между fallback-кандидатами

### Что добавлено

#### 1. Новый router для fallback-провайдеров

Добавлен новый модуль:

- `interfaces/provider_routing.py`

Он выбирает fallback не просто по следующему `qualification score`, а по комбинации:

- качества провайдера,
- допустимого score-gap относительно primary,
- оценки стоимости одного вызова.

#### 2. Стоимость провайдера теперь задается в runtime spec

`interfaces/adapter_schema.py` расширен полем:

- `estimated_cost_per_call_usd`

Если стоимость не задана, поведение остаётся backward-compatible: router не ломает старую qualification-first логику.

#### 3. Charter получил настройки маршрутизации

`motivation/operator_charter.py` расширен параметрами:

- `enable_cost_aware_fallback_routing`
- `provider_routing_quality_weight`
- `provider_routing_cost_weight`
- `provider_routing_max_score_gap`

Это позволяет ограничивать cost-оптимизацию сверху: fallback не должен быть слишком слабее quality-anchor.

#### 4. Rollout gate теперь выбирает canary fallback через cost-aware selector

`interfaces/provider_qualification.py` теперь:

- по-прежнему промотирует лучший live provider как primary,
- но fallback для canary выбирает через cost-aware routing,
- и сохраняет выбранную стратегию в rollout decision.

#### 5. Drift demotion теперь тоже использует cost-aware routing

`interfaces/post_promotion_monitor.py` больше не берёт fallback только как «первый в shadow-списке».

Теперь при regression/demotion fallback также выбирается через cost-aware selector.

#### 6. Новый audit trail в ledger

`memory/goal_ledger.py` теперь хранит:

- `interface_provider_routing/`

Там фиксируются:

- роль маршрутизации (`fallback`, `drift_fallback`),
- выбранный провайдер,
- стратегия,
- ранжирование кандидатов,
- routing score.

#### 7. Новый demo и обновленные config examples

Добавлены/обновлены:

- `provider_cost_routing_demo.py`
- `configs/provider_catalog.canary_rollout.example.json`
- `configs/provider_catalog.shadow_consensus.example.json`
- `configs/operator_charter.canary_rollout.example.json`
- `configs/operator_charter.shadow_consensus.example.json`

Теперь demo-конфиги содержат и стоимость провайдеров, и routing knobs.

### Что подтверждено

- `pytest`: **120 passed**
- `tests/test_v3141_cost_aware_routing.py`: passed
- `provider_cost_routing_demo.py`: passed
- `provider_canary_demo.py`: passed
- `provider_shadow_consensus_demo.py`: passed
- `provider_shadow_demo.py`: passed
- `provider_rollout_demo.py`: passed
- `autonomous_agi.py` с canary-конфигом: passed

### Ключевой подтвержденный сценарий

Если есть несколько валидных fallback-провайдеров, Tolik теперь:

- сохраняет более сильного primary,
- но из fallback-кандидатов выбирает более дешёвый вариант,
  если его quality-gap укладывается в charter-boundary.

На cloud-LLM demo это приводит к выбору:

- primary: `mock_live_fast`
- fallback: `mock_live_safe_alt`
- strategy: `cost_aware_balance`

### Что осталось

После v3.141 в текущей Stage 5 очереди остаётся 1 крупный интерфейсный шаг:

**charter-aware rollback cooldown + anti-flap защита**.

## v3.142 — rollback cooldown + anti-flap защита для provider rollout

### Что добавлено

#### 1. Новый protection-layer для rollout stability

Добавлен новый модуль:

- `interfaces/rollout_protection.py`

Он хранит историю защитных событий и вычисляет:

- cooldown после rollback/demotion,
- anti-flap freeze при повторных сбоях,
- canary suppression, пока нестабильность ещё слишком свежая.

#### 2. Charter получил rollout-protection knobs

`motivation/operator_charter.py` расширен параметрами:

- `rollback_cooldown_rollouts`
- `anti_flap_window_rollouts`
- `anti_flap_repeat_failures`
- `anti_flap_freeze_rollouts`

Теперь политика rollout-stability настраивается явно через charter.

#### 3. Canary rollback теперь пишет защитную историю

`interfaces/canary_rollout.py` после `rolled_back_to_fallback` теперь сохраняет protective record для candidate-провайдера.

Это блокирует его немедленное повторное продвижение при следующем rollout.

#### 4. Drift demotion теперь тоже пишет protective history

`interfaces/post_promotion_monitor.py` после drift-based demotion теперь создаёт такой же protective record.

Таким образом, и canary rollback, и post-promotion demotion используют единый protection mechanism.

#### 5. Rollout gate теперь учитывает cooldown и anti-flap

`interfaces/provider_qualification.py` теперь:

- фильтрует провайдеров, находящихся в cooldown,
- suppress-ит canary rollout, если активен anti-flap freeze,
- старается сохранить текущего safe live provider во время protection window (`protective_stickiness`).

#### 6. Новый audit trail в ledger

`memory/goal_ledger.py` теперь хранит:

- `interface_rollout_protections/`

Там фиксируются:

- trigger type,
- affected provider,
- fallback provider,
- rollout index,
- cooldown horizon,
- anti-flap activation state.

#### 7. Новый demo

Добавлен:

- `provider_anti_flap_demo.py`

Он показывает:

- первый rollback,
- immediate cooldown block,
- повторный rollback,
- anti-flap freeze.

### Что подтверждено

- `pytest`: **123 passed**
- `tests/test_v3142_rollout_protection.py`: passed
- `provider_anti_flap_demo.py`: passed
- `provider_canary_demo.py`: passed
- `provider_shadow_consensus_demo.py`: passed
- `provider_shadow_demo.py`: passed
- `provider_rollout_demo.py`: passed
- `provider_cost_routing_demo.py`: passed
- `autonomous_agi.py` с canary-конфигом: passed

### Ключевые подтвержденные сценарии

1. После canary rollback регрессирующий provider не может быть сразу же повторно promoted.
2. После повторного rollback включается anti-flap freeze и canary временно suppress-ится.
3. После drift demotion rollout gate не возвращает проблемный provider немедленно обратно.
4. Во время protection window Tolik старается держать уже-live safe provider, уменьшая churn.

### Что осталось

После v3.142 обязательных крупных интерфейсных шагов в текущей Stage 5 очереди не осталось.
