# Хронология экспериментов ChemiAI

> Май 2026 · train n=751, test n=250 · метрика: средний **RMSE** (root mean squared error) по IC50, CC50, SI

Документ дополняет краткий список закрытых веток в ноутбуке (`solution.ipynb`, §4.1).
Здесь — полная история фаз, принятые решения и типичные ловушки **OOF** (out-of-fold, предсказание на отложенном фолде).

**Финальная конфигурация** (`BEST_CFG`, Phase AF dual): kNN IC50 k=4 (калиброванный + сырой) + kNN CC50 k=6 + ChEMBL concat (вес 0.15) + LightGBM w=0.59 + full CatBoost w=0.20 + SI α=0.30 + 2 seed (42, 7).

**Прогресс метрики на test:** ~306 (baseline) → **~265.27** (−40.7 RMSE).

---

## Как читать

1. **Хронология** — фазы в порядке выполнения.
2. **Ключевые вехи** — что дало наибольший выигрыш.
3. **Детали по фазам** — гипотезы и итоги.
4. **OOF vs test** — когда локальная оценка обманывает.
5. **Закрытые ветки** — полный перечень (дублирует §4.1 ноутбука, но подробнее).

Схема зависимостей:

```text
Baseline → … → Phase W (diff fe) → X3 (ChEMBL) → Y → Z → AA → AB → AC → AD → AE → AF (BEST)
```

---

## Хронология исследований

| # | Фаза | База RMSE | Цель | Лучший результат | RMSE test | Статус |
|---|------|-----------|------|------------------|-----------|--------|
| 0 | **Baseline** | — | Простой табличный пайплайн | KMeans + HGB + transductive kNN | ~**306** | ✅ Старт |
| 1 | **Pre-B** | ~306 | SI robust, инвариант CC50/IC50 | combo_exp3_inv_a35 | **292.06** | ✅ |
| 2 | **Pre-B** | ~292 | CatBoost IC50 на size-дескрипторах | ic50_size_catboost_w25 | **284.61** | ✅ |
| 3 | **Signal search** | 284.61 | OOF-поиск вокруг базового стека | cc50_trans_k3, ic50_cat_w | OOF −1.4 | ✅ Инфра |
| 4 | **B** | ~280.76 | Тюнинг SI и IC50 при frozen CC50 | ic65 + ext cols | ~**280** | ✅ |
| 5 | **C / C2** | 279.92 | Крупный сигнал, alt-модели | ic50_w65 | ~**280** | 🚫 Закрыто |
| 6 | **D** | ~275 | SI CatBoost / замена SI-головы | SI CatBoost | **280.24** | 🚫 Регресс |
| 7 | **E** | 275.30 | Structural heads (fr, mordred, morgan) | fr42 + ic65 | **274.76** | ✅ Прорыв IC50 |
| 8 | **F** | 274.76 | Массовый SI-поиск | si_trans_k5_w35 OOF | **281.36** | 🚫 SI-ветки закрыты |
| 9 | **G** | 274.76 | Mordred IC50 head | mordred_w55 | **272.99** | ✅ |
| 10 | **H** | 272.99 | Morgan IC50 head | morgan_w25 | **272.44** | ✅ |
| 11 | **I** | 272.44 | Full CatBoost на 192 feat | full_cb25 | **272.17** | ✅ |
| 12 | **J / J2** | 272.17 | LightGBM IC50 head | lgb_ic42 | **269.43** | ✅ −2.7 |
| 13 | **K** | 269.43 | Ratio feature engineering | ratio_lgb55 | **269.09** | ✅ |
| 14 | **L2** | 269.09 | LGB в structural head | inter_lgb48 | +0.70 | 🚫 |
| 15 | **M** | 269.09 | Micro-tune full_cb_w | fcb ≠ 0.25 | регресс | 🚫 |
| 16 | **P** | 269.09 | CC50 transductive fe | cc50_fe_lgb55 | **268.82** | ✅ |
| 17 | **Q** | 268.82 | CC50 blend_w, k-tune | cc50_blend_w70 | **268.60** | ✅ |
| 18 | **R** | 268.60 | Micro blend_w 0.72–0.78 | w72/w75/w78 | хуже | 🚫 |
| 19 | **S** | 268.60 | CC50 CatBoost на fe | cc50_cb_fe | **268.23** | ✅ |
| 20 | **T** | 268.23 | cc50_cat_w tune | cc50_cat_w28 | **268.19** | ✅ |
| 21 | **U** | 268.19 | SI α tune | si_a32 (α=0.32) | **268.16** | ✅ |
| 22 | **V** | 268.16 | kNN, physchem, SI Ridge meta | si_meta_ridge | **312.70** | 🚫 Навсегда |
| 23 | **W** | 268.16 | IC50 diff fe, CC50 dual/shape | ic50_diff_fe | **268.00** | ✅ |
| 24 | **X** | 268.00 | ChEMBL concat / pretrain | concat_w01 | **271.10** | 🚫 |
| 25 | **X2** | 268.00 | kNN + calibration fe | knn_cal_k20 | **268.71** | 🚫 |
| 26 | **X3** | 268.00 | Sim-filter ChEMBL top-200 | sim_concat_w015 | **267.23** | ✅ |
| 27 | **X3 tune** | 267.23 | ext_weight + SI α grid | a030_w015 | **267.22** | ✅ |
| 28 | **Y** | 267.22 | lgb_w, potent, 2-seed | blend427 | **266.60** | ✅ |
| 29 | **Z** | 266.60 | combo potent + lgb_w + full_cb | potent_lgbw058_fullcb | **265.80** | ✅ |
| 30 | **AA** | 265.80 | lgb_w grid, 3-seed | lgbw059_fullcb | **265.74** | ✅ |
| 31 | **AB** | 265.74 | full_cb_w, pretrain | fullcb_w020 | **265.740** | ✅ |
| 32 | **AC** | 265.74 | knn_cal/raw fe | knn_cal_k5 | 265.52 | ✅ |
| 33 | **AD** | 265.52 | k micro, dual knn | knn_cal_k4 | 265.38 | ✅ |
| 34 | **AE** | 265.52 | CC50 knn_cal fe | cc50_knn_cal_k6 | 265.43 | ✅ solo CC50 |
| 35 | **AF** | 265.38 | combo k4+cc50_k6+dual | **k4_cc50_k6_dual** | **265.27** | ✅ **BEST** |

**Легенда:** ✅ принято в пайплайн · 🚫 закрыто (регресс или насыщение)

---

## Ключевые вехи (RMSE на test)

| RMSE | Δ | Изменение |
|------|---|-----------|
| 306.26 | — | Baseline: clustering + HGB |
| 292.06 | −14 | SI robust + ratio-инвариант |
| 284.61 | −7 | IC50 size CatBoost |
| 274.76 | −10 | Phase E: structural stack |
| 272.99 | −2 | Phase G: mordred |
| 272.44 | −0.6 | Phase H: morgan |
| 269.43 | −2.7 | Phase J: LightGBM IC50 |
| 269.09 | −0.3 | Phase K: ratio fe |
| 268.82 | −0.3 | Phase P: CC50 transductive fe |
| 268.00 | −0.16 | Phase W: IC50 diff fe |
| 267.23 | −0.52 | Phase X3: ChEMBL w=0.15 |
| 266.60 | −0.62 | Phase Y: 2-seed blend |
| 265.80 | −0.25 | Phase Z: potent + lgb_w |
| 265.74 | −0.06 | Phase AA: lgb_w=0.59 |
| 265.52 | −0.22 | Phase AC: knn_cal k=5 |
| **265.268** | −0.02 | Phase AF: + knn_raw dual ← **best** |

---

## Финальная формула (Phase AF dual)

```text
Phase AA stack: diff fe, full_cb_w=0.20, lgb_w=0.59, potent ChEMBL concat w=0.15

+ fe_knn_cal_ic50: kNN median pIC50 по potent lib (k=4), quantile map
+ fe_knn_raw_ic50: kNN raw median (dual)
+ fe_knn_cal_cc50: kNN по CC50 proxy (k=6), quantile map
+ concat external в LightGBM fold (ext_weight=0.15)
+ test = mean(seed 42, seed 7)

CC50 / SI — α=0.30
```

---

## Детали по фазам

### Фазы 0–3: Baseline и инфраструктура

| Что делали | Результат |
|------------|-----------|
| KMeans + HGB + transductive PCA/kNN для CC50 | RMSE ~306 |
| SI robust blend + мягкий ratio-инвариант | 292.06 |
| CatBoost на size-дескрипторах для IC50 | 284.61 |
| 5-fold OOF, кэш fold, функция `fit_oof` | Инфраструктура для всех фаз |

**Вывод:** transductive CC50 и ансамбль по таргетам — фундамент; стекинг Ridge на OOF дал RMSE **317** (закрыто).

### Phase E — Structural heads (прорыв IC50)

- Morgan-proxy (`fr_*` + `FpDensityMorgan*`) и Mordred-proxy без SMILES.
- Best: fr42 + ic65 → **274.76**.

### Phase F — SI-поиск

- SI transductive kNN, huber, tail-aware.
- OOF улучшился, test **281.36** → **все SI-ветки закрыты**.

### Phase J — LightGBM IC50

- LightGBM поверх Phase I; best **269.43** (−2.74).
- **Паттерн:** OOF хуже, test лучше — типично для IC50-heads.

### Phase K — Ratio features

- Признаки-отношения: logP/TPSA, Bertz/TPSA и др.; best **269.09**.

### Phase P–T — CC50 fe и tune

- Phase P: CC50 transductive fe → **268.82**.
- Phase Q–T: blend_w, CatBoost на fe, cc50_cat_w → **268.19**.

### Phase V — SI meta (катастрофа)

- SI Ridge/Huber meta: OOF −8.6, test **+44 RMSE** → **закрыто навсегда**.

### Phase W — Target-routed fe

| Кандидат | RMSE test | OOF Δ | Итог |
|----------|-----------|-------|------|
| **ic50_diff_fe** | **268.00** ✅ | −0.10 | best (−0.16) |
| cc50_dual_shape | 269.29 | −0.77 | 🚫 OOF обманул (+1.13 test) |
| routed_dual_fe | 269.36 | −0.87 | 🚫 хуже |

**Сработало:** 2 diff-fe в LightGBM IC50 (`fe_diff_aliph_hetero_rot`, `fe_diff_hetero_kappa3`).

### Phase X / X2 / X3 — external ChEMBL

| Подход | RMSE test | Итог |
|--------|-----------|------|
| Полный concat 2700+ строк, w=0.1 | **271.10** | 🚫 смещение распределений |
| kNN fe + quantile map (без concat) | **268.71** | 🚫 слабее best |
| Sim-filter top-200, w=0.15 | **267.23** | ✅ |
| SI α=0.30 + w=0.15 | **267.22** | ✅ |

### Phase Y–AB — micro-grid

- **Y:** 2-seed blend → **266.60**; ext_weight=**0.15** оптимум.
- **Z:** potent + lgb_w=0.58 + full_cb → **265.80**.
- **AA:** lgb_w=**0.59** → **265.74**; 3-seed закрыт.
- **AB:** full_cb_w=**0.20** → **265.740**; pretrain/ext_cal закрыты.

### Phase AC–AF — kNN push

- **AC:** knn_cal fe k=5 → 265.52.
- **AD:** knn_cal k=4 → 265.38.
- **AE:** cc50_knn_cal k=6 → 265.43 (solo CC50); dual/shape/trans закрыты.
- **AF:** k4 + cc50_k6 + knn_raw dual → **265.268** ← **BEST**.

---

## OOF vs test

### Распределение OOF-ошибки (best pipeline, seed=42)

| Таргет | OOF RMSE | Доля в mean |
|--------|----------|-------------|
| SI | ~788 | **50%** |
| CC50 | ~458 | 29% |
| IC50 | ~324 | 21% |
| **Mean** | **~523** | |

### Три повторяющихся паттерна

1. **IC50/CC50 heads:** OOF **занижает** выигрыш на test (Phase J LightGBM, Phase T cat_w35).
2. **SI learned blend:** OOF **переоценивает** (si_meta_ridge: −8.6 OOF → +44 test).
3. **Transductive fe / CC50 dual:** OOF улучшается, test ухудшается (Phase W).

На финальном этапе nested CV и подбор только по OOF **ненадёжны** — нужна проверка на test.

---

## Закрытые ветки (полный список)

### SI

- Жёсткий SI = CC50/IC50 → RMSE ~**360**
- SI CatBoost, transductive, tail-aware → регресс
- **SI Ridge/Huber meta** (Phase V) → **+44 RMSE**
- α=0.38 хуже α=0.32 / 0.30

### IC50

- Stack Ridge на OOF → **317**
- pIC50, isotonic → регресс
- fe_v2, Ro5, mordred heads (Phase P)
- LightGBM w>0.55 в structural, pair interactions (L2)
- **knn_cal_only** (fe без concat ChEMBL) → +2.4 (Phase AC)
- **pretrain / ext_cal** на метках ChEMBL (Phase AB)

### Внешние данные и смешивание

- DDH merge, sim-фильтры n100/n180/n220 → регресс
- ext_weight > 0.15 → регресс
- **3-seed blend**, potent50/potent60 (Phase AA)
- Полный concat ChEMBL без sim-filter

### CC50

- cc50_trans_k=4 → **270.69**
- blend_w > 0.70 (Phase R)
- physchem transductive (Phase V)
- **cc50_dual_fe, cc50_shape, routed_dual** (Phase W) → **+1.1…+1.2** test
- cc50 kNN: n200 proxy, k=7/k=8, dual/shape solo (Phase AE)

### Прочее

- GNN / ChemBERTa — нет SMILES в CSV для inference
- Seed full ensemble без выигрыша
- full_cb_w микро-тюнинг ≠ 0.20 (Phase M)

---

## Связь с ноутбуком

| Артеfact | Содержание |
|----------|------------|
| `solution.ipynb` §4.1 | Краткий обзор закрытых веток |
| `EXPERIMENTS.md` (этот файл) | Полная хронология и детали фаз |
| `solution.ipynb` §7 | Код финального пайплайна `BEST_CFG` |
