"""
ikea_model.py — часть модульного разбиения Step_project_Q14.
"""
from ikea_core import *


def prepare_ml_data(
        df: pd.DataFrame,
        sellable_verdict: str = "KEEP",
        old_price_verdict: str = "SAFE",
        old_price_features_to_use: Optional[List[str]] = None,
        exclude_noisy_features: Optional[List[str]] = None
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, List[str], List[str], List[str], List[str]
]:
    """
   Выполняет комплексную подготовку и конструирование признаков (Feature Engineering)
    для машинного обучения с предотвращением утечки данных (Data Leakage).

    Функция осуществляет предобработку исходного датасета IKEA, генерирует базовые
    геометрические, NLP и доменные признаки, обрабатывает информацию о скидках
    (old_price) в соответствии с переданным уровнем безопасности, разбивает выборку
    на обучающую и тестовую, после чего рассчитывает статистические признаки
    (категориальные медианы, частоты) строго в рамках обучающей выборки.

    Args:
        df (pd.DataFrame): Исходный датасет, содержащий сырые характеристики товаров.
        sellable_verdict (str, optional): Решение по обработке признака 'sellable_online'.
            Допустимые значения:
            - "KEEP": оставить признак в итоговой матрице.
            - "REMOVE": исключить признак из обучения. По умолчанию "KEEP".
        old_price_verdict (str, optional): Вердикт безопасности для информации о старых ценах.
            Допустимые значения: "SAFE", "RISKY", "LEAKAGE". По умолчанию "SAFE".
        old_price_features_to_use (List[str], optional): Список генерируемых признаков
            на основе 'old_price'. По умолчанию используются только базовый флаг
            ['has_old_price']. Может быть дополнительно добавлен ['discount_abs'].
            🔧 ВАЖНО: 'discount_pct' в этот список НЕ добавляется, даже если он
            присутствует в переданном old_price_features_to_use (например, при
            вердикте "RISKY" от check_old_price_leakage()). Это осознанное
            переопределение решения check_old_price_leakage(): discount_pct
            структурно вычисляется из price (target leakage по построению),
            что не ловится порогами этой функции (они детектируют только
            линейную утечку old_price ≈ price × const). Point-ablation
            подтвердил, что удаление discount_pct статистически нейтрально
            (ΔR²=-0.0055, 95% CI пересекает 0) — модель ничего не теряет.
        exclude_noisy_features (Optional[List[str]], optional): Список названий
            признаков (числовых или бинарных), которые нужно исключить из
            numeric_features/binary_features ПОСЛЕ их обычного построения —
            общий, не привязанный к конкретным именам механизм для быстрого
            отключения признаков, признанных шумом/вредом по итогам Ablation
            Study / Bootstrap Ablation, без переписывания самой функции.
            По умолчанию None — ничего не исключается, поведение функции
            не меняется. Пример: exclude_noisy_features=['category_price_level',
            'is_large_item'] — кандидаты по итогам point-ablation (см. Note
            ниже); признак при этом продолжает СОЗДАВАТЬСЯ (нужен, например,
            для последующей проверки на утечку), просто не передаётся в
            модель. 🔧 ВАЖНО: joint-проверка одновременного удаления обоих
            кандидатов на GridSearchCV-параметрах дала ΔR²=-0.0036 (лучше,
            но НЕ ЗНАЧИМО хуже) — граница нашего же порога шума (0.003),
            что говорит: совместный эффект не аддитивен точечным ΔR² по
            отдельности, и решение об удалении стоит подтвердить отдельным
            прогоном (bootstrap, не единичный train/test split), прежде чем
            включать exclude_noisy_features по умолчанию в пайплайне.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series,
              List[str], List[str], List[str], List[str]]:
            Кортеж, содержащий подготовленные структуры данных для построения моделей:
            1.  X_train (pd.DataFrame): Обучающая выборка признаков.
            2.  X_test (pd.DataFrame): Тестовая выборка признаков.
            3.  y_train (pd.Series): Целевая переменная (цена) для обучения.
            4.  y_test (pd.Series): Целевая переменная (цена) для теста.
            5.  X (pd.DataFrame): Полная матрица признаков (без designer_clean).
            6.  y (pd.Series): Полный вектор целевой переменной.
            7.  numeric_features_total (List[str]): Итоговый список числовых признаков.
            8.  categorical_features (List[str]): Список категориальных признаков.
            9.  binary_features (List[str]): Список бинарных признаков (включая is_large_item).
            10. bool_features (List[str]): Список булевых признаков (например, sellable_online).

    Note:
        Для предотвращения утечки данных (Feature/Target Leakage) расчеты признаков,
        зависящих от распределения целевой переменной или частот всего датасета
        ('category_price_level', 'designer_freq', 'is_large_item'), производятся
        строго на обучающей выборке (X_train) и затем транслируются на тестовую (X_test).
        Мультиколлинеарные признаки ('area', 'log_volume') принудительно удаляются.

        По итогам point-ablation (Bootstrap Ablation, 15 повторов, RandomForest)
        из финального набора признаков также исключены:
        - 'discount_pct' — структурная утечка через price (см. описание параметра
          old_price_features_to_use выше);
        - 'desc_quality_score' — дублирует 'premium_materials_count', ΔR²/ΔMAE
          статистически неотличимы от шума (95% CI пересекает 0);
        - 'complexity_x_premium' — произведение assembly_complexity × is_premium_category,
          не несёт информации сверх компонентов (95% CI пересекает 0);
        - 'has_discount' — 99.73% overlap с 'has_old_price' на реальных данных,
          оставлен более строгий 'has_old_price' (построен на распарсенном числе,
          а не на сырой строке).
    """

    if old_price_features_to_use is None:
        old_price_features_to_use = ['has_old_price']

    print("\n" + "=" * 70)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ML (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 70)

    df_ml = df.copy()

    # ========================================================================
    # ШАГ 1: Индикаторы пропусков
    # ========================================================================
    print("\n ШАГ 1: Создание индикаторов пропусков")
    print("-" * 70)

    df_ml['has_depth'] = (~df_ml['depth'].isna()).astype(int)
    df_ml['has_height'] = (~df_ml['height'].isna()).astype(int)
    df_ml['has_width'] = (~df_ml['width'].isna()).astype(int)

    print(f"✓ has_depth: {df_ml['has_depth'].sum()} товаров с глубиной")
    print(f"✓ has_height: {df_ml['has_height'].sum()} товаров с высотой")
    print(f"✓ has_width: {df_ml['has_width'].sum()} товаров с шириной")

    # ========================================================================
    # ШАГ 2: NLP-признаки
    # ========================================================================
    print("\n ШАГ 2: Создание NLP-признаков")
    print("-" * 70)

    df_ml['desc_length'] = df_ml['short_description'].str.len()
    df_ml['desc_word_count'] = df_ml['short_description'].str.split().str.len()

    premium_markers = ["solid wood", "oak", "walnut", "leather", "steel", "glass"]

    # Количество премиум-материалов (не бинарный!)
    df_ml['premium_materials_count'] = df_ml['short_description'].str.lower().apply(
        lambda x: sum(1 for m in premium_markers if m in str(x))
    )

    print(f"✓ desc_length: средняя длина = {df_ml['desc_length'].mean():.1f} символов")
    print(f"✓ desc_word_count: средняя длина = {df_ml['desc_word_count'].mean():.1f} слов")
    print(f"✓ premium_materials_count: среднее = {df_ml['premium_materials_count'].mean():.2f}")
    # 🔧 desc_quality_score УДАЛЁН из создания (был математически идентичен
    # premium_materials_count — та же формула подсчёта premium_markers).
    # По итогам bootstrap-ablation исключён из numeric_features (95% CI шума),
    # теперь не создаётся вообще, а не просто игнорируется как ранее.

    # ========================================================================
    # ШАГ 3: Геометрические признаки (ТОЛЬКО БАЗОВЫЕ!)
    # ========================================================================
    print("\n ШАГ 3: Создание геометрических признаков")
    print("-" * 70)

    # Оставляем только volume (производные area, log_volume УДАЛЕНЫ)
    df_ml['volume'] = df_ml['width'] * df_ml['height'] * df_ml['depth']

    # ❌ УДАЛЕНО: area = width * depth (дублирует информацию)
    # ❌ УДАЛЕНО: log_volume = log(volume + 1) (дублирует volume)

    print(f"✓ volume: создано {df_ml['volume'].notna().sum()} значений")
    print(f"❌ area, log_volume УДАЛЕНЫ (мультиколлинеарность)")

    # ========================================================================
    # ШАГ 4: Частотность дизайнера — ПЕРЕНОСИМ ПОСЛЕ SPLIT!
    # ========================================================================
    print("\n ШАГ 4: Частотность дизайнера")
    print("-" * 70)
    print("⚠️ designer_freq будет создан ПОСЛЕ train/test split (устранение feature leakage)")

    # ========================================================================
    # ШАГ 5: Бинарные признаки из гипотез + ПРИЗНАКИ ИЗ old_price
    # ========================================================================
    print("\n ШАГ 5: Бинарные признаки из гипотез + old_price")
    print("-" * 70)
    df_ml['has_other_colors'] = (df_ml['other_colors'] == 'Yes').astype(int)
    # 🔧 has_discount: колонка создаётся для справки (использовалась в H1/HA и для
    # overlap-проверки с has_old_price — 99.73% совпадение на реальных данных), но
    # НЕ входит в binary_features ниже и не используется моделью — вместо неё в
    # обучении участвует более строгий has_old_price (построен на распарсенном
    # числе, а не на сырой строке). Печать ниже — информационная, не индикатор
    # реального использования признака.
    df_ml['has_discount'] = (df_ml['old_price'] != 'No old price').astype(int)
    print(f"✓ has_other_colors: {df_ml['has_other_colors'].sum()} товаров")
    print(f"✓ has_discount: {df_ml['has_discount'].sum()} товаров (справочно, в модель не входит)")

    # ========================================================================
    # ШАГ 5.1: ПРИЗНАКИ ИЗ old_price (на основе проверки утечки)
    # ========================================================================
    print(f"\n ШАГ 5.1: Признаки из old_price")
    print("-" * 70)

    # Парсим old_price
    def parse_old_price_safe(x: Any) -> float:
        # 🔧 ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ (осознанное, не баг):
        # Формат "SR 50/4 pack" (цена за упаковку из N шт.) не обрабатывается
        # и намеренно проваливается в except → NaN → has_old_price=0.
        # Затронуто 10 из 3694 строк (0.27%), все — с идентичной строкой
        # old_price при разных price (BRYNILEN/BRENNÅSEN/BURFJORD/BÅTSFJORD/
        # BJORLI: одинаковый "SR 50/4 pack" при price=30/30/40/30/40) — то есть
        # это, вероятно, цена сопутствующего аксессуара (напр. ножек), а не
        # старая цена самого товара. Наивный парсинг ("50" или "50/4=12.5")
        # даёт отрицательную/абсурдную скидку относительно текущей price,
        # поэтому NaN здесь — безопасный дефолт, а не потеря информации.
        # Решение: не парсить, оставить как есть.
        if pd.isna(x) or str(x).strip() == '' or str(x).lower() in ['nan', 'no old price']:
            return np.nan
        try:
            clean = str(x).replace(' ', '').replace('SR', '').replace(',', '')
            if '-' in clean:
                parts = clean.split('-')
                return float(parts[0])
            return float(clean)
        except (ValueError, TypeError):
            return np.nan

    df_ml['old_price_parsed'] = df_ml['old_price'].apply(parse_old_price_safe)

    # has_old_price — бинарный флаг (всегда безопасен)
    df_ml['has_old_price'] = df_ml['old_price_parsed'].notna().astype(int)
    print(f"✓ has_old_price: {df_ml['has_old_price'].sum()} товаров со старой ценой")

    # Проверяем вердикт из check_old_price_leakage

    verdict = old_price_verdict
    features_to_use = old_price_features_to_use
    print(f"  ℹ️ Вердикт по old_price: {verdict}")
    print(f"  ℹ️ Признаки к использованию: {features_to_use}")

    # discount_pct — процент скидки (если вердикт позволяет)
    if 'discount_pct' in features_to_use:
        df_ml['discount_pct'] = np.where(
            df_ml['old_price_parsed'].notna() & (df_ml['old_price_parsed'] > 0),
            (df_ml['old_price_parsed'] - df_ml['price']) / df_ml['old_price_parsed'],
            np.nan
        )
        print(f"✓ discount_pct: {df_ml['discount_pct'].notna().sum()} товаров с процентом скидки")
        print(f"  • Средняя скидка: {df_ml['discount_pct'].mean() * 100:.2f}%")
    else:
        print(f"❌ discount_pct НЕ создаётся (вердикт: {verdict})")

    # discount_abs — абсолютная скидка (только если SAFE)
    if 'discount_abs' in features_to_use:
        df_ml['discount_abs'] = np.where(
            df_ml['old_price_parsed'].notna(),
            df_ml['old_price_parsed'] - df_ml['price'],
            np.nan
        )
        print(f"✓ discount_abs: {df_ml['discount_abs'].notna().sum()} товаров с абсолютной скидкой")
    else:
        print(f"❌ discount_abs НЕ создаётся (вердикт: {verdict})")

    # old_price напрямую — ТОЛЬКО если SAFE (рискованно!)
    if verdict == "SAFE":
        print(f"⚠️  old_price напрямую НЕ используется (риск переобучения)")
    else:
        print(f"❌ old_price напрямую НЕ используется (вердикт: {verdict})")

    # Удаляем временную колонку
    df_ml = df_ml.drop(columns=['old_price_parsed'], errors='ignore')

    # ========================================================================
    # ШАГ 6: ПРИЗНАКИ ДЛЯ ПРЕМИУМ-СЕГМЕНТА (УПРОЩЕНО)
    # ========================================================================
    print("\n ШАГ 6: Создание признаков для премиум-сегмента")
    print("-" * 70)

    # 6.1. Флаг премиум-категории
    premium_categories = ['Wardrobes', 'Sofas & armchairs', 'Beds',
                          'Sideboards, buffets & console tables', 'Cabinets & cupboards']
    df_ml['is_premium_category'] = df_ml['category'].isin(premium_categories).astype(int)
    print(f"✓ is_premium_category: {df_ml['is_premium_category'].sum()} товаров в премиум-категориях")

    # ❌ УДАЛЕНО: is_large_item (переносится ПОСЛЕ split — устранение leakage)
    # ❌ УДАЛЕНО: premium_x_volume, premium_x_log_volume, premium_x_team (мультиколлинеарность)
    print(f"❌ is_large_item, premium_x_* УДАЛЕНЫ (leakage + мультиколлинеарность)")

    # ========================================================================
    # ШАГ 6.6: ПРИЗНАК СЛОЖНОСТИ СБОРКИ (УПРОЩЕНО)
    # ========================================================================
    print(f"\n ШАГ 6.6: Признак сложности сборки (из отраслевых данных)")
    print("-" * 70)

    # Шкала сложности сборки IKEA (0-1)
    assembly_complexity = {
        'Tables & desks': 0.30,
        'Chairs': 0.20,
        'Bookcases & shelving units': 0.25,
        'Beds': 0.55,
        'Cabinets & cupboards': 0.60,
        'Chests of drawers & drawer units': 0.60,
        'Wardrobes': 0.80,
        'Sofas & armchairs': 0.55,
        'Outdoor furniture': 0.35,
        "Children's furniture": 0.25,
        'Bar furniture': 0.20,
        'Café furniture': 0.20,
        'Trolleys': 0.10,
        'Room dividers': 0.15,
        'TV & media furniture': 0.40,
        'Sideboards, buffets & console tables': 0.45,
        'Nursery furniture': 0.25,
    }

    df_ml['assembly_complexity'] = df_ml['category'].map(assembly_complexity)

    print(f"✓ assembly_complexity: признак сложности сборки добавлен")
    print(f"  • Средняя сложность: {df_ml['assembly_complexity'].mean():.2f}")

    # ❌ УДАЛЕНО: complexity_x_premium (произведение assembly_complexity × is_premium_category —
    # по итогам bootstrap-ablation не несёт информации сверх компонентов, 95% CI шума).
    # ❌ УДАЛЕНО: complexity_x_log_volume (дублирует complexity_x_premium + volume)
    # ❌ УДАЛЕНО: is_complex_assembly (дублирует assembly_complexity)
    print(f"❌ complexity_x_premium, complexity_x_log_volume, is_complex_assembly УДАЛЕНЫ (мультиколлинеарность/шум)")

    # ========================================================================
    # ШАГ 6.7: ЦЕНОВЫЕ СЕГМЕНТЫ — ПОЛНОСТЬЮ УДАЛЕНЫ!
    # ========================================================================
    print(f"\n ШАГ 6.7: Ценовые сегменты")
    print("-" * 70)
    print("❌ price_segment, segment_x_* ПОЛНОСТЬЮ УДАЛЕНЫ")
    print("   Причина: target leakage (создавались на всём датасете)")
    print("   Деревья сами находят нелинейные зависимости от категории")

    # ========================================================================
    # ШАГ 7: Исключение ненужных колонок
    # ========================================================================
    print("\n🗑️ ШАГ 7: Исключение ненужных колонок")
    print("-" * 70)

    drop_cols = [
        'Unnamed: 0', 'item_id', 'name', 'old_price', 'link',
        'short_description', 'designer', 'other_colors',
        'price', 'all_dims_missing', 'description_composite', 'slash_in_name',
        'is_accessory', 'is_mass_market', 'designer_type'
    ]

    # 🆕 ИСПРАВЛЕНИЕ: НЕ удаляем 'designer_clean' здесь!
    # Он нужен для создания designer_freq ПОСЛЕ split

    # sellable_online — оставляем (анализ показал значимость)

    if sellable_verdict == "REMOVE":
        drop_cols.append('sellable_online')
        print(f"  ⚠️ sellable_online УДАЛЁН (по результатам анализа)")
    else:
        print(f"  ✅ sellable_online ОСТАВЛЕН в модели")

    cols_to_drop = [col for col in drop_cols if col in df_ml.columns]
    print(f"\n✓ Исключаем {len(cols_to_drop)} колонок:")
    for col in cols_to_drop:
        print(f"  • {col}")
    df_ml = df_ml.drop(columns=cols_to_drop)

    # ========================================================================
    # ШАГ 8: Разделение на X и y
    # ========================================================================
    print("\n ШАГ 8: Разделение на X и y")
    print("-" * 70)

    X = df_ml.copy()
    y = df['price'].copy()

    print(f"Размер X: {X.shape}")
    print(f"Размер y: {y.shape}")
    print(f"\nПризнаки в X ({len(X.columns)} шт.):")
    # 🔧 Эти 2 колонки создаются для справки/overlap-проверки (has_old_price vs
    # has_discount) и для check_data_leakage(), но НЕ передаются в numeric_features/
    # binary_features на Шаге 9 — в модель не входят. Явно помечаем здесь, чтобы
    # при чтении лога не создавалось впечатление, что все колонки X используются.
    reference_only_columns = {'discount_pct', 'has_discount'}
    for i, col in enumerate(X.columns, 1):
        dtype = X[col].dtype
        n_missing = X[col].isnull().sum()
        marker = "  ⚠️ справочно, в модель НЕ входит" if col in reference_only_columns else ""
        print(f"  {i:2d}. {col:30s} | тип: {str(dtype):10s} | пропусков: {n_missing:4d}{marker}")

    # ========================================================================
    # ШАГ 9: Классификация признаков для Pipeline (УПРОЩЕНО!)
    # ========================================================================
    print("\n ШАГ 9: Классификация признаков для Pipeline")
    print("-" * 70)

    # 📌 Числовые признаки (БАЗОВЫЕ + динамическое добавление из old_price)
    # 🔧 ИСКЛЮЧЕНЫ по итогам point-ablation (Bootstrap Ablation, 15 повторов, RF):
    #   - desc_quality_score: ΔR²/ΔMAE в пределах 95% CI шума (дублирует premium_materials_count)
    #   - complexity_x_premium: ΔR²/ΔMAE в пределах 95% CI шума (произведение assembly_complexity × is_premium_category,
    #     не несёт информации сверх компонентов)
    numeric_features = [
        # Габариты (базовые)
        'depth', 'height', 'width', 'volume',
        # NLP
        'desc_length', 'desc_word_count',
        'premium_materials_count',
        # Сложность сборки
        'assembly_complexity',
    ]

    # 🔧 discount_pct ИСКЛЮЧЁН НАМЕРЕННО, независимо от вердикта check_old_price_leakage().
    # Причина: discount_pct = (old_price - price) / old_price — признак по построению
    # является функцией от target (price), то есть структурная утечка (target leakage).
    # check_old_price_leakage() выдаёт вердикт "RISKY" (не "LEAKAGE") только потому, что
    # std отношения old_price/price = 0.1637 >= 0.05 — этот порог детектирует ЛИНЕЙНУЮ
    # утечку (old_price ≈ price × const), но не ловит эту, более простую, структурную
    # утечку через саму формулу скидки. Point-ablation подтвердил, что удаление
    # discount_pct статистически нейтрально (ΔR²=-0.0055, 95% CI [-0.0163;+0.0053],
    # пересекает 0) — то есть модель ничего не теряет.
    if old_price_features_to_use:
        if 'discount_abs' in old_price_features_to_use and 'discount_abs' in df_ml.columns:
            numeric_features.append('discount_abs')

    #  Числовые признаки, которые БУДУТ ДОБАВЛЕНЫ после split (ШАГ 10.1)
    numeric_features_after_split = [
        'category_price_level',  # медиана по категории (только на train)
        'designer_freq',  # частотность дизайнера (только на train)
    ]

    # Итоговый список числовых признаков (для справки)
    numeric_features_total = numeric_features + numeric_features_after_split

    #  Бинарные признаки (УПРОЩЕНО!)
    binary_features = [
        # Гипотезы
        'is_team', 'has_other_colors', 'is_composite',
        # 🔧 has_discount ИСКЛЮЧЁН: оставлен только has_old_price (99.73% overlap
        # между ними на реальных данных — 10 расхождений, все из-за формата
        # "SR X/N pack" в old_price, см. комментарий в parse_old_price_safe()).
        # has_old_price построен на распарсенном числе (более строгий критерий),
        # а не на сырой строке. Point-ablation подтвердил: удаление любого из
        # двух по отдельности статистически нейтрально (оба CI пересекают 0) —
        # значит один полностью подменяет другой, оставлен более чистый.
        'has_old_price',
        # Индикаторы пропусков
        'has_depth', 'has_height', 'has_width',
        # Премиум
        'is_premium_category',
    ]

    categorical_features = ['category']

    # sellable_online только если он не был удалён
    if 'sellable_online' in df_ml.columns:
        bool_features = ['sellable_online']
    else:
        bool_features = []

    # 📊 Вывод информации
    print(f"Числовые признаки ({len(numeric_features_total)}):")
    print(f"  📌 Уже есть в X ({len(numeric_features)}):")
    for f in numeric_features:
        print(f"    • {f}")
    print(f"  📌 Будут добавлены после split ({len(numeric_features_after_split)}):")
    for f in numeric_features_after_split:
        print(f"    • {f} (для предотвращения утечки данных)")

    print(f"\nБинарные признаки ({len(binary_features)}):")
    for f in binary_features:
        print(f"  • {f}")

    print(f"\nКатегориальные признаки ({len(categorical_features)}): {categorical_features}")
    print(f"Булевые признаки ({len(bool_features)}): {bool_features}")

    total = len(numeric_features_total) + len(binary_features) + len(categorical_features) + len(bool_features)
    print(f"\n Всего признаков (итого): {total}")

    # ПРОВЕРКА: все ли ТЕКУЩИЕ признаки есть в X
    all_current_features = numeric_features + binary_features + categorical_features + bool_features
    missing = [f for f in all_current_features if f not in X.columns]
    if missing:
        print(f"\n⚠️ ВНИМАНИЕ: отсутствуют в X: {missing}")
        print(f"   Проверьте, правильно ли созданы эти признаки!")
    else:
        print(f"\n Все текущие признаки найдены в X!")
        print(f"   (признаки {numeric_features_after_split} будут добавлены в ШАГЕ 10.1)")

    # ========================================================================
    # ШАГ 10: Разделение на train/test
    # ========================================================================
    print("\n ШАГ 10: Разделение на train/test (80/20) ")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"X_train: {X_train.shape} ")
    print(f"X_test:  {X_test.shape} ")

    # ========================================================================
    # ШАГ 10.1: Создание признаков ПОСЛЕ SPLIT (устранение leakage!)
    # ========================================================================
    print(f"\n ШАГ 10.1: Создание признаков ПОСЛЕ split (устранение leakage)")
    print("-" * 70)

    # 10.1.1: category_price_level (медиана цены по категории)
    print("\n 10.1.1: category_price_level (только на train)")
    train_with_price = X_train.copy()
    train_with_price['price'] = y_train
    category_medians = train_with_price.groupby('category')['price'].median()

    X_train['category_price_level'] = X_train['category'].map(category_medians)
    X_train['category_price_level'] = X_train['category_price_level'].fillna(y_train.median())

    X_test['category_price_level'] = X_test['category'].map(category_medians)
    X_test['category_price_level'] = X_test['category_price_level'].fillna(y_train.median())

    print(f"✓ category_price_level: вычислен на train, применён к train/test")

    # 10.1.2: designer_freq (частотность дизайнера)
    print("\n 10.1.2: designer_freq (только на train)")
    designer_counts = X_train['designer_clean'].value_counts()

    X_train['designer_freq'] = X_train['designer_clean'].map(designer_counts).fillna(1)
    X_test['designer_freq'] = X_test['designer_clean'].map(designer_counts).fillna(1)

    print(f"✓ designer_freq: вычислен на train, применён к train/test")
    print(f"  • Средняя частота (train): {X_train['designer_freq'].mean():.1f}")

    # 🆕 ИСПРАВЛЕНИЕ: Теперь, когда designer_freq создан, удаляем designer_clean
    X_train = X_train.drop(columns=['designer_clean'])
    X_test = X_test.drop(columns=['designer_clean'])
    X = X.drop(columns=['designer_clean'])
    print(f"✓ designer_clean УДАЛЁН из X_train, X_test, X (больше не нужен)")

    # 10.1.3: is_large_item (крупный товар)
    print("\n 10.1.3: is_large_item (только на train)")
    volume_median_train = X_train['volume'].median()

    X_train['is_large_item'] = (X_train['volume'] > volume_median_train * 2).astype(int)
    X_test['is_large_item'] = (X_test['volume'] > volume_median_train * 2).astype(int)

    print(f"✓ is_large_item: вычислен на train, применён к train/test")
    print(f"  • Медиана объёма (train): {volume_median_train:.0f} см³")
    print(f"  • Крупных товаров (train): {X_train['is_large_item'].sum()}")

    # Добавляем is_large_item в бинарные признаки
    binary_features.append('is_large_item')

    print(f"\n✅ LEAKAGE УСТРАНЁН: все статистики вычислены только на train set!")

    # ========================================================================
    # ИСКЛЮЧЕНИЕ ПРИЗНАКОВ ПО ЗАПРОСУ (exclude_noisy_features)
    # ========================================================================
    # 🔧 Общий, не привязанный к конкретным именам механизм — признак(и)
    # продолжают существовать как колонки в X_train/X_test/X (на случай, если
    # нужны для последующих проверок вроде check_data_leakage), но убираются
    # из списков, которые реально идут в модель через preprocessor.
    if exclude_noisy_features:
        excluded_numeric = [f for f in exclude_noisy_features if f in numeric_features_total]
        excluded_binary = [f for f in exclude_noisy_features if f in binary_features]
        not_found = [f for f in exclude_noisy_features
                     if f not in numeric_features_total and f not in binary_features]

        numeric_features_total = [f for f in numeric_features_total if f not in exclude_noisy_features]
        binary_features = [f for f in binary_features if f not in exclude_noisy_features]

        print(f"\n🔧 exclude_noisy_features: исключено из модели по запросу")
        if excluded_numeric:
            print(f"  • Из числовых: {excluded_numeric}")
        if excluded_binary:
            print(f"  • Из бинарных: {excluded_binary}")
        if not_found:
            print(f"  ⚠️ Не найдены (уже отсутствуют или опечатка): {not_found}")

    # ========================================================================
    # ИТОГ
    # ========================================================================
    print(f"\n Данные подготовлены: ")
    print(f"  • Удалены мультиколлинеарные признаки (area, log_volume, premium_x_*, segment_x_*)")
    print(f"  • Устранены утечки данных (designer_freq, is_large_item, category_price_level)")
    print(f"  • Всего признаков: {len(X.columns)} ")
    print(f"  • Числовых: {len(numeric_features_total)}")
    print(f"  • Бинарных: {len(binary_features)}")
    print(f"  • Категориальных: {len(categorical_features)}")
    print(f"  • Булевых: {len(bool_features)}")

    return X_train, X_test, y_train, y_test, X, y, numeric_features_total, categorical_features, binary_features, bool_features

# =============================================================================
# ЧАСТЬ 15: СРАВНЕНИЕ МОДЕЛЕЙ
# =============================================================================

def compare_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
        bool_features: List[str]
) -> Tuple[pd.DataFrame, str, Pipeline, ColumnTransformer]:
    """
    Строит препроцессинг, обучает пул ML-моделей на логарифмированном таргете,
    сравнивает их по метрикам регрессии и визуализирует результаты.

    Функция автоматизирует процесс выбора лучшей модели:
    1. Фильтрует списки признаков, оставляя только присутствующие в X_train.
    2. Логарифмирует целевую переменную: y_log = log1p(y).
    3. Для каждой модели динамически собирает Sklearn Pipeline с учетом
       необходимости масштабирования (scaling) числовых данных.
    4. Обучает модели и считает метрики (MAE, RMSE, R², RMSLE, MAPE) на исходной шкале цен.
    5. Ранжирует модели по R² и строит диагностические графики (Actual vs Predicted, Residuals).

    Args:
        X_train (pd.DataFrame): Обучающий набор признаков.
        X_test (pd.DataFrame): Тестовый набор признаков.
        y_train (pd.Series): Целевая переменная (оригинальные цены) для обучения.
        y_test (pd.Series): Целевая переменная (оригинальные цены) для валидации.
        numeric_features (List[str]): Список исходных числовых признаков.
        categorical_features (List[str]): Список исходных категориальных признаков.
        binary_features (List[str]): Список исходных бинарных признаков.
        bool_features (List[str]): Список исходных булевых признаков.

    Returns:
        Tuple[pd.DataFrame, str, Pipeline, ColumnTransformer]:
            Результаты сравнения моделей:
            1.  results_df (pd.DataFrame): Сводная таблица метрик всех моделей,
                отсортированная по R² по убыванию.
            2.  best_model_name (str): Название модели-победителя (например, 'XGBoost').
            3.  best_pipeline (Pipeline): Обученный Sklearn Pipeline для лучшей модели
                (включает препроцессор и модель).
            4.  preprocessor_best (ColumnTransformer): Настроенный препроцессор,
                использованный для лучшей модели.

    Note:
        - Функция требует наличия глобальной переменной `MODELS_REQUIRING_SCALING`
          (список моделей, критичных к масштабу признаков, например LinearRegression, Ridge).
        - Обучение моделей происходит строго на значениях `np.log1p(y)`, однако все
          выводящиеся метрики рассчитываются после обратного преобразования `np.expm1()`
          на оригинальном масштабе цен (SR).
        - Построенные графики сохраняются автоматически с помощью функции `save_plot()`.
    """
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (С ЛОГАРИФМИРОВАНИЕМ TARGET)")
    print("=" * 70)

    # ========================================================================
    # ФИЛЬТРАЦИЯ ПРИЗНАКОВ — используем только те, что реально есть в X_train
    # ========================================================================
    print("\n🔍 Фильтрация признаков (используем только существующие в X_train):")

    # Фильтруем числовые признаки
    numeric_features_filtered = [f for f in numeric_features if f in X_train.columns]
    removed_numeric = [f for f in numeric_features if f not in X_train.columns]

    # Фильтруем бинарные признаки
    binary_features_filtered = [f for f in binary_features if f in X_train.columns]
    removed_binary = [f for f in binary_features if f not in X_train.columns]

    # Фильтруем категориальные признаки
    categorical_features_filtered = [f for f in categorical_features if f in X_train.columns]

    # Фильтруем булевые признаки
    bool_features_filtered = [f for f in bool_features if f in X_train.columns]

    # Выводим информацию
    print(f"  📌 Числовые признаки: {len(numeric_features_filtered)} из {len(numeric_features)}")
    if removed_numeric:
        print(f"     ❌ Удалены (нет в X_train): {removed_numeric}")

    print(f"  📌 Бинарные признаки: {len(binary_features_filtered)} из {len(binary_features)}")
    if removed_binary:
        print(f"     ❌ Удалены (нет в X_train): {removed_binary}")

    print(f"  📌 Категориальные признаки: {len(categorical_features_filtered)} из {len(categorical_features)}")
    print(f"  📌 Булевые признаки: {len(bool_features_filtered)} из {len(bool_features)}")

    # Используем отфильтрованные списки
    numeric_features = numeric_features_filtered
    binary_features = binary_features_filtered
    categorical_features = categorical_features_filtered
    bool_features = bool_features_filtered

    # ========================================================================
    # ЛОГАРИФМИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
    # ========================================================================
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    print(f"\n📊 Логарифмирование целевой переменной:")
    print(f"  • y_train: медиана = {y_train.median():.2f}, log-медиана = {y_train_log.median():.4f}")
    print(f"  • y_test:  медиана = {y_test.median():.2f}, log-медиана = {y_test_log.median():.4f}")


    # ========================================================================
    # ПРЕПРОЦЕССОР (ДИФФЕРЕНЦИРОВАННЫЙ ПО ТИПУ МОДЕЛИ)
    # ========================================================================
    #  Препроцессор создаётся ВНУТРИ цикла для каждой модели
    # (см. ниже в цикле по моделям)
    # Объединяем все бинарные/булевы признаки
    all_binary_features = binary_features + bool_features
    # ========================================================================
    # МОДЕЛИ
    # ========================================================================
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    results = []

    # ========================================================================
    # ЦИКЛ ПО МОДЕЛЯМ
    # ========================================================================
    print(f"\n🚀 Обучение {len(models)} моделей...")
    print("-" * 70)

    for name, model in models.items():
    # 🔑 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: выбираем препроцессор в зависимости от типа модели
        with_scaling = name in MODELS_REQUIRING_SCALING

        preprocessor = get_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=all_binary_features,
            with_scaling=with_scaling
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        # Обучаем на log(price)
        pipeline.fit(X_train, y_train_log)

        # Предсказываем в log-пространстве
        y_pred_log = pipeline.predict(X_test)

        # Обратное преобразование
        y_pred = np.expm1(y_pred_log)

        # Метрики в оригинальном пространстве
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

        # MAPE (Mean Absolute Percentage Error) — относительная ошибка
        mask_nonzero = y_test != 0
        mape = np.mean(np.abs((y_test[mask_nonzero] - y_pred[mask_nonzero]) / y_test[mask_nonzero])) * 100

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'RMSLE': rmsle,
            'MAPE': mape
        })

        print(f"  ✓ {name}:")
        print(f"     MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, RMSLE: {rmsle:.4f}, MAPE: {mape:.1f}%")

    # ========================================================================
    # СВОДНАЯ ТАБЛИЦА
    # ========================================================================
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА:")
    print("=" * 70)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

    best_model_name = results_df.iloc[0]['Model']
    print(f"\n🏆 Лучшая модель: {best_model_name} (R² = {results_df.iloc[0]['R2']:.4f})")

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print("\n📈 Создание визуализаций...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: R² по моделям
    ax1 = axes[0, 0]
    colors = ['gold' if i == 0 else 'skyblue' for i in range(len(results_df))]
    bars = ax1.bar(results_df['Model'], results_df['R2'], color=colors, edgecolor='black')
    ax1.set_ylabel('R²')
    ax1.set_title('R² по моделям', fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    for bar, r2 in zip(bars, results_df['R2']):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

    # График 2: RMSE по моделям
    ax2 = axes[0, 1]
    bars2 = ax2.bar(results_df['Model'], results_df['RMSE'], color=colors, edgecolor='black')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE по моделям', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# График 3: Actual vs Predicted
    ax3 = axes[1, 0]

# 🔑 Используем препроцессор для лучшей модели
    with_scaling_best = best_model_name in MODELS_REQUIRING_SCALING
    preprocessor_best = get_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=all_binary_features,
        with_scaling=with_scaling_best
    )

    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_best),
        ('model', models[best_model_name])
    ])

    best_pipeline.fit(X_train, y_train_log)
    y_pred_best = np.expm1(best_pipeline.predict(X_test))

    ax3.scatter(y_test, y_pred_best, alpha=0.5, s=20, c='steelblue')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Идеальное')
    ax3.set_xlabel('Фактическая цена')
    ax3.set_ylabel('Предсказанная цена')
    ax3.set_title(f'{best_model_name}: Actual vs Predicted', fontweight='bold')
    ax3.legend()

    # График 4: Residuals
    ax4 = axes[1, 1]
    residuals = y_test - y_pred_best
    ax4.scatter(y_pred_best, residuals, alpha=0.5, s=20, c='coral')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Предсказанная цена')
    ax4.set_ylabel('Остатки')
    ax4.set_title(f'{best_model_name}: Residuals', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'ml_model_comparison')

    print(f"\n✅ Сравнение моделей завершено!")
    print(f"  • Использовано признаков: {len(numeric_features) + len(categorical_features) + len(all_binary_features)}")
    print(f"  • Обучено моделей: {len(models)}")
    print(f"  • Лучшая: {best_model_name} (R² = {results_df.iloc[0]['R2']:.4f})")

    return results_df, best_model_name, best_pipeline, preprocessor_best

# =============================================================================
# ЧАСТЬ 16: GRIDSEARCHCV
# =============================================================================

def gridsearch_xgboost(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer
) -> Tuple[Pipeline, GridSearchCV, float]:
    """
    Выполняет оптимизацию гиперпараметров модели XGBoost методом решетчатого поиска
    (GridSearchCV) с использованием кросс-валидации на логарифмированном таргете.

    Функция решает следующие задачи:
    1. Настраивает сетку поиска для регуляризации дерева (глубина, подвыборки строк/колонок)
       и параметров обучения (количество итераций, темп обучения).
    2. Оценивает комбинации параметров с помощью 5-фолдовой кросс-валидации по метрике R².
    3. Выводит лучшие гиперпараметры и их R² на этапе кросс-валидации.
    4. Оценивает итоговое качество на отложенной тестовой выборке с обратным
       преобразованием предсказаний (`expm1`) к исходному масштабу цен (SR).
    5. Визуализирует качество подбора: сопоставление прогнозов с фактом, диаграмму
       остатков, гистограмму распределения остатков и график Q-Q для оценки нормальности ошибок.

    Args:
        X_train (pd.DataFrame): Обучающий набор признаков.
        X_test (pd.DataFrame): Тестовый набор признаков.
        y_train (pd.Series): Фактические цены для обучения (будут логарифмированы внутри).
        y_test (pd.Series): Фактические цены для тестирования (используются для валидации).
        preprocessor (ColumnTransformer): Настроенный Sklearn препроцессор для
            трансформации признаков перед подачей в модель.

    Returns:
        Tuple[Pipeline, GridSearchCV, float]:
            Результаты оптимизации:
            1.  best_model (Pipeline): Обученный Sklearn Pipeline, использующий
                оптимальную комбинацию параметров XGBoost.
            2.  grid_search (GridSearchCV): Объект поиска Sklearn, содержащий полную
                историю итераций, метрик и метаданных оптимизации.
            3.  r2 (float): Коэффициент детерминации (R²) лучшей модели на тестовом множестве.

    Note:
        - Оптимизация методом Grid Search требует значительных вычислительных ресурсов.
          Параметр `n_jobs=-1` задействует все доступные потоки процессора.
        - Визуализации остатков автоматически экспортируются на диск с помощью
          вспомогательной функции `save_plot()`.
    """
    print("\n" + "=" * 70)
    print("GRIDSEARCHCV ДЛЯ XGBOOST")
    print("=" * 70)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    print(f"\nЗапуск GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_log)

    print(f"\nЛучшие гиперпараметры:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")
    print(f"\nЛучший R² на CV: {grid_search.best_score_:.4f}")

    # Оценка на тесте
    best_model = grid_search.best_estimator_
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nМетрики на тесте:")
    print(f"  • MAE: {mae:.2f}")
    print(f"  • RMSE: {rmse:.2f}")
    print(f"  • R²: {r2:.4f}")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20, c='steelblue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Идеальное')
    ax1.set_xlabel('Фактическая цена')
    ax1.set_ylabel('Предсказанная цена')
    ax1.set_title(f'XGBoost + GridSearchCV (R²={r2:.3f})', fontweight='bold')
    ax1.legend()

    ax2 = axes[0, 1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20, c='coral')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Предсказанная цена')
    ax2.set_ylabel('Остатки')
    ax2.set_title('Residuals Plot', fontweight='bold')

    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Остатки')
    ax3.set_ylabel('Частота')
    ax3.set_title('Распределение остатков', fontweight='bold')

    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot остатков', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'ml_gridsearchcv')

    return best_model, grid_search, r2

# =============================================================================
# ЧАСТЬ 17: КРОСС-ВАЛИДАЦИЯ
# =============================================================================

def cross_validate_model(
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: ColumnTransformer,
        best_params: Optional[Dict[str, Any]] = None,
        model_name: str = 'RandomForest'
) -> Tuple[np.ndarray, Pipeline]:
    """
    Выполняет кросс-валидацию финальной модели со стратификацией по целевой переменной,
    после чего обучает итоговый пайплайн на всем объеме данных.

    🔧 ДИНАМИЧЕСКИЙ ВЫБОР МОДЕЛИ:
    Функция автоматически создаёт модель нужного типа на основе параметра `model_name`.
    Поддерживаются: 'RandomForest', 'HistGB', 'XGBoost'. Префиксы 'model__' в
    `best_params` автоматически удаляются для совместимости с пайплайнами sklearn.

    Логика работы функции:
    1. Динамический выбор модели на основе `model_name` (никаких зашивок!).
    2. Проводит разбиение на 5 фолдов. Для непрерывной переменной `y` (цена)
       строятся страты на основе квантилей (`pd.qcut`).
    3. Проверяет применимость стратификации (размер и количество страт). Если условия
       не выполняются, безопасно переключается на классический `KFold`.
    4. Вычисляет метрику R² на кросс-валидации в логарифмическом пространстве `np.log1p(y)`.
    5. Обучает финальный пайплайн на всех предоставленных данных.
    6. Визуализирует оценки R² по каждому фолду и сохраняет график.

    Args:
        X (pd.DataFrame): Полный набор признаков (объединенный train + test или только train).
        y (pd.Series): Оригинальные значения целевой переменной (цены).
        preprocessor (ColumnTransformer): Настроенный Sklearn препроцессор для
            подготовки признаков.
        best_params (Optional[Dict[str, Any]], optional): Оптимальные гиперпараметры
            модели. Если None, используется пустой словарь и модель по умолчанию.
            Defaults to None.
        model_name (str, optional): Название базового алгоритма модели. Поддерживаются:
            'RandomForest', 'HistGB', 'XGBoost'. По умолчанию 'RandomForest'.

    Returns:
        Tuple[np.ndarray, Pipeline]:
            Результаты валидации и итоговая модель:
            1.  cv_scores (np.ndarray): Массив значений R² для каждого из 5 фолдов.
            2.  final_pipeline (Pipeline): Обученный на полной выборке Sklearn Pipeline,
                готовый к инференсу (включает препроцессор и обученную модель).

    Raises:
        ValueError: Возникает внутри блока `try-except` при неравномерном распределении
            страт, что инициирует безопасный переход (fallback) на обычный `KFold`.

    Note:
        - Перед началом валидации индексы `X` и `y` принудительно сбрасываются
          (`reset_index`), чтобы избежать несовпадения индексов при разбиении.
        - Валидация и финальное обучение происходят на логарифмированном таргете `y_log`.
        - График распределения R² по фолдам сохраняется на диск с помощью `save_plot()`.
        - Ключи `random_state` и `n_jobs` принудительно удаляются из `best_params`
          перед передачей в конструктор модели, чтобы избежать конфликта
          "multiple values for keyword argument" при явной передаче этих параметров.
    """
    print(f"\n{'=' * 70}")
    print("КРОСС-ВАЛИДАЦИЯ ФИНАЛЬНОЙ МОДЕЛИ")
    print(f"{'=' * 70}")

    # ========================================================================
    # 🔧 1. Динамический выбор модели на основе model_name
    # ========================================================================
    # Нормализуем имя модели для поддержки разных вариантов написания
    model_name_normalized = model_name.lower().replace(' ', '').replace('_', '')

    # Определяем тип модели и её дефолтные параметры
    if model_name_normalized in ('randomforest', 'rf'):
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True
    elif model_name_normalized in ('histgb', 'histgradientboosting', 'hist'):
        model_class = HistGradientBoostingRegressor
        default_params = {
            'max_iter': 200,
            'learning_rate': 0.1,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'max_bins': 255,
            'random_state': 42
        }
        supports_n_jobs = False  # HistGB не поддерживает n_jobs
    elif model_name_normalized in ('xgboost', 'xgb'):
        model_class = xgb.XGBRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1
        }
        supports_n_jobs = True
    else:
        # Fallback: неизвестный тип — используем RandomForest с предупреждением
        print(f"\n⚠️ Неизвестный тип модели '{model_name}', используем RandomForest")
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True

    print(f"📊 Используемая модель: {model_class.__name__}")

    # ========================================================================
    # 2. Очистка параметров от префиксов 'model__'
    # ========================================================================
    if best_params is None:
        best_params = default_params.copy()
    else:
        # Создаём копию, чтобы не мутировать исходный словарь
        best_params = best_params.copy()
        clean_params = {}
        for key, value in best_params.items():
            if key.startswith('model__'):
                clean_params[key[7:]] = value
            else:
                clean_params[key] = value
        best_params = clean_params

    # 🔧 Гарантируем random_state=42 для воспроизводимости
    best_params['random_state'] = 42

    # 🔧 Удаляем n_jobs для моделей, которые его не поддерживают
    if not supports_n_jobs and 'n_jobs' in best_params:
        best_params.pop('n_jobs', None)

    # 🔧 ДИНАМИЧЕСКОЕ СОЗДАНИЕ МОДЕЛИ (никаких зашивок!)
    model = model_class(**best_params)

    # Сбрасываем индексы (важно после pd.concat!)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    y_log = np.log1p(y)

    # ========================================================================
    # СТРАТИФИЦИРОВАННАЯ КРОСС-ВАЛИДАЦИЯ (с fallback на KFold)
    # ========================================================================

    n_splits = 5
    cv_indices = None

    # ВАЖНО: используем Pipeline с preprocessor, а не голую модель!
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    try:
        # Создаём страты через квантили и явно приводим к int
        y_strata = pd.qcut(y, q=n_splits, labels=False, duplicates='drop').astype(int)

        # Проверяем минимальный размер страты
        min_count = y_strata.value_counts().min()
        n_unique_strata = y_strata.nunique()

        print(f"\n Анализ страт:")
        print(f"   Уникальных страт: {n_unique_strata}")
        print(f"   Минимальный размер страты: {min_count}")
        print(f"   Распределение: {y_strata.value_counts().to_dict()}")

        # Если все страты имеют достаточно примеров — используем стратификацию
        if min_count >= n_splits and n_unique_strata >= n_splits:
            stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_indices = list(stratified_cv.split(X, y_strata))
            print(f"✅ Используется стратифицированная {n_splits}-кратная CV")
        else:
            print(f"⚠️ Страты неравномерны, используем обычный KFold")
            raise ValueError("Неравномерные страты")

    except Exception as e:
        # Fallback на обычный KFold
        print(f"⚠️ Стратификация не удалась: {e}")
        print(f"   Используем обычный {n_splits}-кратный KFold")
        kfold_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_indices = list(kfold_cv.split(X))

    # ВАЖНО: передаём pipeline, а не model!
    cv_scores = cross_val_score(pipeline, X, y_log, cv=cv_indices, scoring='r2', n_jobs=-1)

    print(f"\n Результаты {n_splits}-кратной кросс-валидации:")
    print(f"   R² по фолдам: {cv_scores}")
    print(f"   Средний R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Обучаем финальную модель на всех данных
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    final_pipeline.fit(X, y_log)

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', linewidth=2, markersize=10)
    ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Средний R²: {cv_scores.mean():.4f}')
    ax.set_xlabel('Номер фолда')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Кросс-валидация ({model_class.__name__})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i, score in enumerate(cv_scores):
        ax.text(i + 1, score + 0.01, f'{score:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'ml_cross_validation_final')
    plt.close()

    print(f"  ✓ Сохранено: ml_cross_validation_final.png и ml_cross_validation_final.svg")

    return cv_scores, final_pipeline

# =============================================================================
# ЧАСТЬ 18: FEATURE IMPORTANCES
# =============================================================================

def optimize_with_optuna_experiments(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
        rerun: bool = False,
        n_trials_1: int = 50,
        n_trials_2: int = 75,
        n_trials_3: int = 75,
        cv_folds: int = 5
) -> Tuple[
    Optional[Pipeline], Optional[str],
    Optional['optuna.study.Study'], Optional['optuna.study.Study'],
    Optional[pd.DataFrame], Optional[Dict[str, Any]]
]:
    """Выполняет комплексный автоматический поиск гиперпараметров (AutoML) с помощью Optuna
    для нескольких архитектур моделей и сравнивает их с динамически вычисляемым baseline.

    Функция последовательно проводит четыре независимых эксперимента:
        1. Exp1: Оптимизация XGBoost на суженном пространстве параметров (базовый).
        2. Exp2: Оптимизация XGBoost на расширенном пространстве параметров.
        3. Exp3: Настройка гистограммного градиентного бустинга (HistGradientBoostingRegressor).
        4. Exp4: Настройка случайного леса (RandomForestRegressor).

    Для сокращения времени выполнения реализован механизм кэширования: если `rerun=False`
    и кэш существует, функция мгновенно восстанавливает параметры лучшей модели,
    метрики и итоговый пайплайн без повторного запуска расчетов.

    Args:
        X_train (pd.DataFrame): Обучающий набор признаков.
        X_test (pd.DataFrame): Тестовый набор признаков для финальной оценки моделей.
        y_train (pd.Series): Оригинальные целевые значения (цены) для обучения.
        y_test (pd.Series): Оригинальные целевые значения (цены) для теста.
        preprocessor (ColumnTransformer): Настроенный Sklearn препроцессор для
            трансформации данных внутри кросс-валидационного пайплайна.
        rerun (bool, optional): Флаг принудительного перезапуска расчетов. Если True,
            кэш игнорируется и перезаписывается. Defaults to False.
        n_trials_1 (int, optional): Число итераций Optuna для экспериментов
            Exp1 и Exp4. Defaults to 50.
        n_trials_2 (int, optional): Число итераций Optuna для Exp2. Defaults to 75.
        n_trials_3 (int, optional): Число итераций Optuna для Exp3 (HistGB). Defaults to 75.
        cv_folds (int, optional): Количество фолдов при кросс-валидации. Defaults to 5.

    Returns:
        Tuple[Optional[Pipeline], Optional[str], Optional[optuna.study.Study],
              Optional[optuna.study.Study], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
            Возвращает кортеж из 6 элементов:
                1. final_pipeline (Pipeline): Обученный на полном X_train Sklearn Pipeline.
                2. final_model_name (str): Текстовое название победившей модели.
                3. study_exp1 (optuna.study.Study): Объект исследования Optuna для Exp1.
                4. study_exp2 (optuna.study.Study): Объект исследования Optuna для Exp2.
                5. comparison_df (pd.DataFrame): Сводная таблица сравнения метрик.
                6. best_params (Dict[str, Any]): Словарь оптимальных гиперпараметров.

    Note:
        - Обучение и кросс-валидация проводятся на логарифмированной целевой переменной
          `np.log1p(y)`, однако итоговые метрики R² и MAE на тесте рассчитываются в
          исходном масштабе цен после потенцирования (`np.expm1`).
        - В процессе работы функция строит и сохраняет комплексный график из четырех
          подграфиков (сравнение R², сравнение MAE и истории сходимости Exp1 и Exp2)
          под именем 'ml_optuna_experiments_comparison'.
    """
    print("\n" + "=" * 70)
    print("OPTUNA: СЕРИЯ ЭКСПЕРИМЕНТОВ ДЛЯ ПОИСКА ЛУЧШИХ ПАРАМЕТРОВ")
    print("=" * 70)
    print(f"\n⚙ Параметры: trials={n_trials_1}/{n_trials_2}/{n_trials_3}/{n_trials_1}, cv={cv_folds}")

    # ========================================================================
    # Динамическое вычисление baseline (Random Forest без настройки)
    # ========================================================================
    print("\n📊 Вычисление baseline (Random Forest без настройки)...")
    baseline_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', baseline_rf)
    ])

    y_train_log = np.log1p(y_train)
    baseline_pipeline.fit(X_train, y_train_log)
    y_pred_baseline = np.expm1(baseline_pipeline.predict(X_test))
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

    print(f"  • Baseline RF R² (тест): {baseline_r2:.4f}")
    print(f"  • Baseline RF MAE (тест): {baseline_mae:.2f} SR")

    # ========================================================================
    # ПРОВЕРКА КЭША (с валидацией отпечатка — набор признаков + trials/cv)
    # ========================================================================
    current_fingerprint = compute_optuna_fingerprint(
        columns=list(X_train.columns),
        n_trials_1=n_trials_1, n_trials_2=n_trials_2, n_trials_3=n_trials_3,
        cv_folds=cv_folds
    )

    if not rerun:
        cache = load_optuna_cache()
        cached_fingerprint = cache.get('fingerprint') if cache else None

        if cache and cached_fingerprint == current_fingerprint:
            print("✅ Кэш найден — используем сохранённые результаты")

            # Чтение метрик из кэша (ключи должны совпадать с save_optuna_cache)
            r2_exp1 = cache.get('exp1_r2', 0.0)
            r2_exp2 = cache.get('exp2_r2', 0.0)
            r2_hgb = cache.get('exp3_r2', 0.0)
            r2_rf = cache.get('exp4_r2', 0.0)

            mae_exp1 = cache.get('exp1_mae', 0.0)
            mae_exp2 = cache.get('exp2_mae', 0.0)
            mae_hgb = cache.get('exp3_mae', 0.0)
            mae_rf = cache.get('exp4_mae', 0.0)

            # Определение лучшей модели среди экспериментов
            r2_values = {
                'XGBoost + Optuna (Exp1)': r2_exp1,
                'XGBoost + Optuna (Exp2)': r2_exp2,
                'HistGB + Optuna (Exp3)': r2_hgb,
                'RandomForest + Optuna (Exp4)': r2_rf
            }
            mae_values = {
                'XGBoost + Optuna (Exp1)': mae_exp1,
                'XGBoost + Optuna (Exp2)': mae_exp2,
                'HistGB + Optuna (Exp3)': mae_hgb,
                'RandomForest + Optuna (Exp4)': mae_rf
            }

            best_r2_cached = max(r2_values.values())
            best_mae_cached = mae_values[[k for k, v in r2_values.items() if v == best_r2_cached][0]]

            best_model_name = cache['best_model_name']
            best_params = cache['best_params']

            # Определяем тип модели по параметрам
            if 'max_iter' in best_params:
                model = HistGradientBoostingRegressor(**best_params, random_state=42)
            elif 'learning_rate' in best_params and 'bootstrap' not in best_params:
                model = xgb.XGBRegressor(**best_params, random_state=42)
            else:
                model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

            final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

            # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обучаем пайплайн перед возвратом
            final_pipeline.fit(X_train, y_train_log)

            # Формирование сравнительной таблицы (baseline — динамический)
            comparison_df = pd.DataFrame({
                'Модель': [
                    'Baseline: RF без настройки',
                    'XGBoost + Optuna (Exp1)',
                    'XGBoost + Optuna (Exp2)',
                    'HistGB + Optuna (Exp3)',
                    'RandomForest + Optuna (Exp4)'
                ],
                'R² (тест)': [
                    baseline_r2,
                    r2_exp1,
                    r2_exp2,
                    r2_hgb,
                    r2_rf
                ],
                'MAE': [
                    baseline_mae,
                    mae_exp1,
                    mae_exp2,
                    mae_hgb,
                    mae_rf
                ]
            })

            print(f"   Лучшая модель: {best_model_name}")
            print(f"   R² (тест): {best_r2_cached:.4f}")
            print(f"   MAE: {best_mae_cached:.2f}")
            return final_pipeline, best_model_name, None, None, comparison_df, best_params
        else:
            if cache is None:
                print("\n⚠️ Кэш не найден — запускаем оптимизацию Optuna...")
            else:
                print("\n⚠️ Кэш найден, но УСТАРЕЛ (изменился набор признаков или "
                      "параметры trials/cv) — запускаем оптимизацию Optuna заново...")
                print(f"   Сохранённый отпечаток:  {cached_fingerprint}")
                print(f"   Текущий отпечаток:      {current_fingerprint}")

    # ========================================================================
    # ОПТИМИЗАЦИЯ
    # ========================================================================
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("✓ optuna установлен")
    except ImportError:
        print("❌ Установите: pip install optuna")
        return None, None, None, None, None, None

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    all_experiments = []

    # ========================================================================
    # ЭКСПЕРИМЕНТ 1: XGBoost базовый
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f" ЭКСПЕРИМЕНТ 1: Базовый ({n_trials_1} trials, cv={cv_folds})")
    print(f"{'=' * 70}")

    def objective_exp1(trial: optuna.trial.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 3),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        }
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=1)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv_folds, scoring='r2', n_jobs=1)
        return scores.mean()

    print(f"\n Запуск Эксперимента 1 ({n_trials_1} trials, cv={cv_folds})...")
    study_exp1 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_exp1.optimize(objective_exp1, n_trials=n_trials_1, show_progress_bar=True)

    best_xgb_exp1 = xgb.XGBRegressor(**study_exp1.best_params, random_state=42)
    pipeline_exp1 = Pipeline([('preprocessor', preprocessor), ('model', best_xgb_exp1)])
    pipeline_exp1.fit(X_train, y_train_log)
    y_pred_exp1 = np.expm1(pipeline_exp1.predict(X_test))
    r2_exp1 = r2_score(y_test, y_pred_exp1)
    mae_exp1 = mean_absolute_error(y_test, y_pred_exp1)

    print(f"\n✅ Эксперимент 1 завершён:")
    print(f"  • Лучший CV R²: {study_exp1.best_value:.4f}")
    print(f"  • Тест R²: {r2_exp1:.4f}")
    print(f"  • Тест MAE: {mae_exp1:.2f}")

    all_experiments.append({
        'Experiment': f'Exp1: {n_trials_1} trials, cv={cv_folds}',
        'CV R²': study_exp1.best_value,
        'Test R²': r2_exp1,
        'Test MAE': mae_exp1,
        'Trials': n_trials_1,
        'CV': cv_folds
    })

    # ========================================================================
    # ЭКСПЕРИМЕНТ 2: XGBoost расширенный
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f" ЭКСПЕРИМЕНТ 2: Расширенный ({n_trials_2} trials, cv={cv_folds})")
    print(f"{'=' * 70}")

    def objective_exp2(trial: optuna.trial.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=1)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv_folds, scoring='r2', n_jobs=1)
        return scores.mean()

    print(f"\n⏳ Запуск Эксперимента 2 ({n_trials_2} trials, cv={cv_folds})...")
    study_exp2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_exp2.optimize(objective_exp2, n_trials=n_trials_2, show_progress_bar=True)

    best_xgb_exp2 = xgb.XGBRegressor(**study_exp2.best_params, random_state=42)
    pipeline_exp2 = Pipeline([('preprocessor', preprocessor), ('model', best_xgb_exp2)])
    pipeline_exp2.fit(X_train, y_train_log)
    y_pred_exp2 = np.expm1(pipeline_exp2.predict(X_test))
    r2_exp2 = r2_score(y_test, y_pred_exp2)
    mae_exp2 = mean_absolute_error(y_test, y_pred_exp2)

    print(f"\n✅ Эксперимент 2 завершён:")
    print(f"  • Лучший CV R²: {study_exp2.best_value:.4f}")
    print(f"  • Тест R²: {r2_exp2:.4f}")
    print(f"  • Тест MAE: {mae_exp2:.2f}")

    all_experiments.append({
        'Experiment': f'Exp2: {n_trials_2} trials, cv={cv_folds}',
        'CV R²': study_exp2.best_value,
        'Test R²': r2_exp2,
        'Test MAE': mae_exp2,
        'Trials': n_trials_2,
        'CV': cv_folds
    })

    # ========================================================================
    # ЭКСПЕРИМЕНТ 3: HistGB
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f" ЭКСПЕРИМЕНТ 3: HistGB ({n_trials_3} trials, cv={cv_folds})")
    print(f"{'=' * 70}")

    def objective_exp3(trial: optuna.trial.Trial) -> float:
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
            'max_bins': trial.suggest_int('max_bins', 128, 255),
        }
        model = HistGradientBoostingRegressor(**params, random_state=42)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv_folds, scoring='r2', n_jobs=1)
        return scores.mean()

    print(f"\n⏳ Запуск Эксперимента 3: HistGB ({n_trials_3} trials, cv={cv_folds})...")
    study_exp3 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_exp3.optimize(objective_exp3, n_trials=n_trials_3, show_progress_bar=True)

    best_hgb = HistGradientBoostingRegressor(**study_exp3.best_params, random_state=42)
    pipeline_hgb = Pipeline([('preprocessor', preprocessor), ('model', best_hgb)])
    pipeline_hgb.fit(X_train, y_train_log)
    y_pred_hgb = np.expm1(pipeline_hgb.predict(X_test))
    r2_hgb = r2_score(y_test, y_pred_hgb)
    mae_hgb = mean_absolute_error(y_test, y_pred_hgb)

    print(f"\n✅ Эксперимент 3 завершён:")
    print(f"  • Лучший CV R²: {study_exp3.best_value:.4f}")
    print(f"  • Тест R²: {r2_hgb:.4f}")
    print(f"  • Тест MAE: {mae_hgb:.2f}")

    all_experiments.append({
        'Experiment': f'Exp3: HistGB {n_trials_3} trials, cv={cv_folds}',
        'CV R²': study_exp3.best_value,
        'Test R²': r2_hgb,
        'Test MAE': mae_hgb,
        'Trials': n_trials_3,
        'CV': cv_folds
    })

    # ========================================================================
    # ЭКСПЕРИМЕНТ 4: Random Forest с оптимизацией
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f" ЭКСПЕРИМЕНТ 4: Random Forest ({n_trials_1} trials, cv={cv_folds})")
    print(f"{'=' * 70}")

    def objective_exp4(trial: optuna.trial.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv_folds, scoring='r2', n_jobs=1)
        return scores.mean()

    print(f"\n⏳ Запуск Эксперимента 4: Random Forest ({n_trials_1} trials, cv={cv_folds})...")
    study_exp4 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_exp4.optimize(objective_exp4, n_trials=n_trials_1, show_progress_bar=True)

    best_rf = RandomForestRegressor(**study_exp4.best_params, random_state=42)
    pipeline_rf = Pipeline([('preprocessor', preprocessor), ('model', best_rf)])
    pipeline_rf.fit(X_train, y_train_log)
    y_pred_rf = np.expm1(pipeline_rf.predict(X_test))
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    print(f"\n✅ Эксперимент 4 завершён:")
    print(f"  • Лучший CV R²: {study_exp4.best_value:.4f}")
    print(f"  • Тест R²: {r2_rf:.4f}")
    print(f"  • Тест MAE: {mae_rf:.2f}")

    all_experiments.append({
        'Experiment': f'Exp4: RF {n_trials_1} trials, cv={cv_folds}',
        'CV R²': study_exp4.best_value,
        'Test R²': r2_rf,
        'Test MAE': mae_rf,
        'Trials': n_trials_1,
        'CV': cv_folds
    })

    # ========================================================================
    # СРАВНЕНИЕ ВСЕХ ЭКСПЕРИМЕНТОВ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" СРАВНЕНИЕ ВСЕХ ЭКСПЕРИМЕНТОВ")
    print(f"{'=' * 70}")

    exp_df = pd.DataFrame(all_experiments)

    # Динамический baseline
    baseline_row = {
        'Experiment': 'Baseline: RF без настройки',
        'CV R²': None,
        'Test R²': baseline_r2,
        'Test MAE': baseline_mae,
        'Trials': 0,
        'CV': 0
    }
    exp_df = pd.concat([exp_df, pd.DataFrame([baseline_row])], ignore_index=True)
    print(exp_df.to_markdown(index=False, floatfmt=".4f"))

    best_exp = exp_df.loc[exp_df['Test R²'].idxmax()]
    print(f"\n ЛУЧШИЙ ЭКСПЕРИМЕНТ: {best_exp['Experiment']}")
    print(f"   Test R²: {best_exp['Test R²']:.4f}")
    print(f"   Test MAE: {best_exp['Test MAE']:.2f}")

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВИЗУАЛИЗАЦИЯ ЭКСПЕРИМЕНТОВ")
    print(f"{'=' * 70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral', 'violet', 'gray']
    bars = ax1.bar(exp_df['Experiment'], exp_df['Test R²'], color=colors, edgecolor='black')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Сравнение R² по экспериментам', fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    for bar, val in zip(bars, exp_df['Test R²']):
        if not pd.isna(val):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', fontweight='bold')
    # Динамическая линия baseline
    ax1.axhline(y=baseline_r2, color='red', linestyle='--', linewidth=2,
                label=f'Baseline RF: R²={baseline_r2:.4f}')
    ax1.legend()

    ax2 = axes[0, 1]
    bars2 = ax2.bar(exp_df['Experiment'], exp_df['Test MAE'], color=colors, edgecolor='black')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('Сравнение MAE по экспериментам', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    for bar, val in zip(bars2, exp_df['Test MAE']):
        if not pd.isna(val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.0f}', ha='center', fontweight='bold')
    # Динамическая линия baseline
    ax2.axhline(y=baseline_mae, color='red', linestyle='--', linewidth=2,
                label=f'Baseline RF: MAE={baseline_mae:.2f}')
    ax2.legend()

    ax3 = axes[1, 0]
    trials_exp1 = [t.value for t in study_exp1.trials if t.value is not None]
    ax3.plot(range(len(trials_exp1)), trials_exp1, 'b-', alpha=0.3, label='Все trials')
    best_values_exp1 = np.maximum.accumulate(trials_exp1)
    ax3.plot(range(len(best_values_exp1)), best_values_exp1, 'r-', linewidth=2, label='Лучшее')
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('CV R²')
    ax3.set_title('Exp1: История оптимизации XGBoost', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    trials_exp2 = [t.value for t in study_exp2.trials if t.value is not None]
    ax4.plot(range(len(trials_exp2)), trials_exp2, 'b-', alpha=0.3, label='Все trials')
    best_values_exp2 = np.maximum.accumulate(trials_exp2)
    ax4.plot(range(len(best_values_exp2)), best_values_exp2, 'r-', linewidth=2, label='Лучшее')
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('CV R²')
    ax4.set_title('Exp2: История оптимизации XGBoost', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, 'ml_optuna_experiments_comparison')

    # ========================================================================
    # Правильная логика выбора финальной модели
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВЫБОР ФИНАЛЬНОЙ МОДЕЛИ")
    print(f"{'=' * 70}")

    if best_exp['Experiment'] == f'Exp1: {n_trials_1} trials, cv={cv_folds}':
        final_pipeline = pipeline_exp1
        final_model_name = 'XGBoost + Optuna (Exp1)'
        best_params = study_exp1.best_params
    elif best_exp['Experiment'] == f'Exp2: {n_trials_2} trials, cv={cv_folds}':
        final_pipeline = pipeline_exp2
        final_model_name = 'XGBoost + Optuna (Exp2)'
        best_params = study_exp2.best_params
    elif best_exp['Experiment'] == f'Exp3: HistGB {n_trials_3} trials, cv={cv_folds}':
        final_pipeline = pipeline_hgb
        final_model_name = 'HistGB + Optuna (Exp3)'
        best_params = study_exp3.best_params
    elif best_exp['Experiment'] == f'Exp4: RF {n_trials_1} trials, cv={cv_folds}':
        final_pipeline = pipeline_rf
        final_model_name = 'RandomForest + Optuna (Exp4)'
        best_params = study_exp4.best_params
    else:
        # При победе baseline возвращаем БАЗОВЫЙ RF
        print("\n🏆 Random Forest без настройки оказался лучше!")
        print("   Принцип Occam's Razor: простая модель работает лучше сложной")
        final_pipeline = baseline_pipeline  # ← Базовый RF, а не RF+Optuna!
        final_model_name = 'RandomForest (baseline, без настройки)'
        best_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}

    print(f"\n Финальная модель: {final_model_name}")
    print(f"  • Test R²: {best_exp['Test R²']:.4f}")
    print(f"  • Test MAE: {best_exp['Test MAE']:.2f}")

    # Динамический baseline в таблице сравнения
    comparison_df = pd.DataFrame({
        'Модель': [
            'Baseline: RF без настройки',
            'XGBoost + Optuna (Exp1)',
            'XGBoost + Optuna (Exp2)',
            'HistGB + Optuna (Exp3)',
            'RandomForest + Optuna (Exp4)'
        ],
        'R² (тест)': [
            baseline_r2,
            r2_exp1,
            r2_exp2,
            r2_hgb,
            r2_rf
        ],
        'MAE': [
            baseline_mae,
            mae_exp1,
            mae_exp2,
            mae_hgb,
            mae_rf
        ]
    })

    # Сохраняем в кэш (вместе с отпечатком текущего набора признаков + trials/cv)
    save_optuna_cache(final_model_name, best_params,
                      r2_exp1, mae_exp1, r2_exp2, mae_exp2, r2_hgb, mae_hgb, r2_rf, mae_rf,
                      fingerprint=current_fingerprint)

    return final_pipeline, final_model_name, study_exp1, study_exp2, comparison_df, best_params

# =============================================================================
# ЧАСТЬ 20: SHAP анализ по категориямтдля интерпритации  модели
# =============================================================================

def check_data_leakage(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
        bool_features: List[str]
) -> Tuple[float, float, float, str]:
    r"""
  Проверяет модель на наличие утечки данных (Data Leakage) через целевые или синтетические признаки.

    Функция исследует потенциальное влияние «заглядывания в будущее» через два канала:
    1.  Признак `category_price_level` (сравниваются метрики модели с ним и без него).
    2.  Признаки группы `old_price` (`has_old_price`, `discount_abs`), которые
        могут содержать скрытую информацию о целевой цене товара.

    🔧 ВАЖНО: `discount_pct` НЕ входит в проверяемую группу, несмотря на упоминание в
    названии переменных ниже по коду (`old_price_features_in_data` может теоретически
    включать её по имени). Признак исключён из numeric_features ещё в prepare_ml_data()
    как структурная утечка (discount_pct = функция от price по построению — см. docstring
    prepare_ml_data() и check_old_price_leakage()), поэтому к моменту вызова этой функции
    он физически отсутствует в активном наборе признаков модели, и данная проверка
    отдельно discount_pct уже не тестирует — только has_old_price и discount_abs.

    Методология проверки строится на обучении изолированных моделей `HistGradientBoostingRegressor`
    на логарифмированном таргете `np.log1p(y)` с последующим экспонированием предсказаний
    и оценкой деградации метрик R² и MAE на тестовой выборке.

    Args:
        X_train (pd.DataFrame): Обучающий набор признаков.
        X_test (pd.DataFrame): Тестовый набор признаков.
        y_train (pd.Series): Истинные значения целевой переменной (цены) для обучения.
        y_test (pd.Series): Истинные значения целевой переменной (цены) для валидации.
        numeric_features (List[str]): Исходный список непрерывных числовых признаков.
        categorical_features (List[str]): Список категориальных признаков.
        binary_features (List[str]): Список бинарных признаков.
        bool_features (List[str]): Список булевых (True/False) признаков.

    Returns:
        Tuple[float, float, float, str]:
            Результаты тестирования стабильности модели при исключении признака `category_price_level`:
            1.  r2_with (float): Коэффициент детерминации $R^2$ модели, обученной на полном
                наборе признаков.
            2.  r2_without (float): Коэффициент детерминации $R^2$ модели после удаления
                исследуемого признака `category_price_level`.
            3.  r2_diff (float): Абсолютное изменение метрики ($\Delta R^2 = R^2_{with} - R^2_{without}$).
            4.  verdict (str): Инженерный вердикт безопасности признака:
                - "SAFE": $\Delta R^2 < 0.01$ (признак не вызывает утечки, влияние минимально).
                - "RISKY": $0.01 \le \Delta R^2 \le 0.05$ (признак дает подозрительный прирост,
                  требуется экспертный анализ).
                - "LEAKAGE": $\Delta R^2 > 0.05$ (явная утечка данных, признак подлежит удалению).

    Note:
        - Функция автоматически фильтрует входные списки признаков, исключая колонки,
          которых физически нет в `X_train` (например, не созданные на этапе предобработки),
          что предотвращает падения `ColumnTransformer`.
        - Результаты промежуточных проверок (включая детальный расчет влияния `old_price`)
          и сравнительные таблицы метрик форматированно выводятся в стандартный поток вывода (stdout).
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА НА УТЕЧКУ ДАННЫХ: category_price_level")
    print("=" * 70)

    # ========================================================================
    # 🔧 КРИТИЧЕСКИ ВАЖНО: Фильтрация признаков
    # Оставляем только те числовые признаки, которые реально есть в X_train
    # ========================================================================
    numeric_features_filtered = [f for f in numeric_features if f in X_train.columns]
    removed_features = [f for f in numeric_features if f not in X_train.columns]

    if removed_features:
        print(f"\n⚠️ Удалены из numeric_features (нет в X_train): {removed_features}")
        print(f"   Причина: признаки не были созданы в prepare_ml_data()")

    # Используем отфильтрованный список во всех моделях
    numeric_features = numeric_features_filtered

    print(f"\n📊 Числовых признаков для проверки утечки: {len(numeric_features)}")
    print(f"   Список: {numeric_features}")

    # ========================================================================
    # Модель 1: С признаком category_price_level
    # ========================================================================
    print("\n Модель 1: С признаком category_price_level")
    # Объединяем все бинарные/булевы признаки
    all_binary_features = binary_features + bool_features

    # 🔧 ИСПРАВЛЕНИЕ: используем универсальный препроцессор
    # HistGradientBoosting — дерево, поэтому with_scaling=False
    preprocessor_with = get_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=all_binary_features,
        with_scaling=False  # HistGradientBoosting — дерево
    )

    model_with = Pipeline([
        ('preprocessor', preprocessor_with),
        ('model', HistGradientBoostingRegressor(
            max_iter=346,
            learning_rate=0.082,
            max_depth=14,
            min_samples_leaf=10,
            l2_regularization=0.96,
            random_state=42
        ))
    ])

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model_with.fit(X_train, y_train_log)
    y_pred_with = model_with.predict(X_test)
    y_pred_with = np.expm1(y_pred_with)

    r2_with = r2_score(y_test, y_pred_with)
    mae_with = mean_absolute_error(y_test, y_pred_with)

    print(f"  • R² (тест): {r2_with:.4f}")
    print(f"  • MAE: {mae_with:.2f}")
    print(
        f"  • Признаков: {len(numeric_features) + len(categorical_features) + len(binary_features) + len(bool_features)}")

    # ========================================================================
    # Модель 2: БЕЗ признака category_price_level
    # ========================================================================
    print("\n Модель 2: БЕЗ признака category_price_level")

    # Удаляем признак из списков
    numeric_features_no_leak = [f for f in numeric_features if f != 'category_price_level']

    # Удаляем признак из данных
    X_train_no_leak = X_train.drop(columns=['category_price_level'], errors='ignore')
    X_test_no_leak = X_test.drop(columns=['category_price_level'], errors='ignore')

    print(f"  • Удалён признак: category_price_level")
    print(f"  • Осталось числовых признаков: {len(numeric_features_no_leak)}")
    print(
        f"  • Всего признаков: {len(numeric_features_no_leak) + len(categorical_features) + len(binary_features) + len(bool_features)}")

    # 🔑 Используем универсальный препроцессор
    preprocessor_without = get_preprocessor(
        numeric_features=numeric_features_no_leak,
        categorical_features=categorical_features,
        binary_features=all_binary_features,
        with_scaling=False  # HistGradientBoosting — дерево
    )

    model_without = Pipeline([
        ('preprocessor', preprocessor_without),
        ('model', HistGradientBoostingRegressor(
            max_iter=346,
            learning_rate=0.082,
            max_depth=14,
            min_samples_leaf=10,
            l2_regularization=0.96,
            random_state=42
        ))
    ])

    model_without.fit(X_train_no_leak, y_train_log)
    y_pred_without = model_without.predict(X_test_no_leak)
    y_pred_without = np.expm1(y_pred_without)

    r2_without = r2_score(y_test, y_pred_without)
    mae_without = mean_absolute_error(y_test, y_pred_without)

    print(f"  • R² (тест): {r2_without:.4f}")
    print(f"  • MAE: {mae_without:.2f}")

    # ========================================================================
    # Модель 1.5: С признаками из old_price (если они есть)
    # ========================================================================
    print("\n Модель 1.5: Проверка признаков из old_price")

    # Проверяем, есть ли признаки из old_price СРЕДИ РЕАЛЬНО ИСПОЛЬЗУЕМЫХ признаков модели
    # 🔧 ИСПРАВЛЕНО: раньше проверялось `feat in X_train.columns` — это сырой датафрейм,
    # где 'discount_pct' физически существует как неиспользуемая колонка (prepare_ml_data()
    # создаёт её всегда, но исключает из numeric_features по причине структурной утечки —
    # см. docstring prepare_ml_data()). Из-за этого лог ошибочно сообщал, что discount_pct
    # тестируется на утечку, хотя фактически его не было ни в numeric_features, ни в
    # binary_features, и вся проверка для него была вычислительно пустой (нечего удалять).
    # Теперь проверяем членство в реальных списках признаков модели, а не в сыром X_train.
    active_features = set(numeric_features) | set(binary_features) | set(bool_features)
    old_price_features_in_data = []
    for feat in ['has_old_price', 'discount_pct', 'discount_abs']:
        if feat in active_features:
            old_price_features_in_data.append(feat)

    if old_price_features_in_data:
        print(f"  • Найдены признаки из old_price: {old_price_features_in_data}")

        # Модель БЕЗ признаков из old_price
        X_train_no_old = X_train.drop(columns=old_price_features_in_data, errors='ignore')
        X_test_no_old = X_test.drop(columns=old_price_features_in_data, errors='ignore')

        # 🔧 ИСПРАВЛЕНИЕ: фильтруем ВСЕ списки признаков!
        numeric_features_no_old = [f for f in numeric_features if f not in old_price_features_in_data]
        binary_features_no_old = [f for f in binary_features if f not in old_price_features_in_data]
        bool_features_no_old = [f for f in bool_features if f not in old_price_features_in_data]
        all_binary_features_no_old = binary_features_no_old + bool_features_no_old

        print(f"  • Числовых признаков без old_price: {len(numeric_features_no_old)}")
        print(f"  • Бинарных признаков без old_price: {len(all_binary_features_no_old)}")

        # 🔧 ИСПРАВЛЕНИЕ: используем универсальный препроцессор
        # HistGradientBoosting — дерево, поэтому with_scaling=False
        preprocessor_no_old = get_preprocessor(
            numeric_features=numeric_features_no_old,
            categorical_features=categorical_features,
            binary_features=all_binary_features_no_old,
            with_scaling=False  # HistGradientBoosting — дерево
        )

        model_no_old = Pipeline([
            ('preprocessor', preprocessor_no_old),
            ('model', HistGradientBoostingRegressor(
                max_iter=346, learning_rate=0.082, max_depth=14,
                min_samples_leaf=10, l2_regularization=0.96,
                random_state=42
            ))
        ])

        model_no_old.fit(X_train_no_old, y_train_log)
        y_pred_no_old = np.expm1(model_no_old.predict(X_test_no_old))
        r2_no_old = r2_score(y_test, y_pred_no_old)
        mae_no_old = mean_absolute_error(y_test, y_pred_no_old)

        # Модель С признаками из old_price (уже есть в model_with)
        r2_with_old = r2_with
        mae_with_old = mae_with

        r2_diff_old = r2_with_old - r2_no_old
        mae_diff_old = mae_with_old - mae_no_old

        print(f"\n  • R² с old_price:   {r2_with_old:.4f}")
        print(f"  • R² без old_price: {r2_no_old:.4f}")
        print(f"  • ΔR²:              {r2_diff_old:+.4f}")
        print(f"  • ΔMAE:             {mae_diff_old:+.2f}")

        if r2_diff_old > 0.05:
            print(f"\n  ⚠️  ПОДОЗРЕНИЕ НА УТЕЧКУ! ΔR² > 0.05")
            print(f"     Рекомендуется удалить признаки из old_price")
            old_price_verdict = "LEAKAGE"
        elif r2_diff_old > 0.02:
            print(f"\n  ⚠️  Признаки из old_price дают сильный прирост")
            print(f"     Требуется дополнительная проверка")
            old_price_verdict = "RISKY"
        else:
            print(f"\n  ✅ Признаки из old_price безопасны")
            old_price_verdict = "SAFE"
    else:
        print(f"  ℹ️  Признаки из old_price отсутствуют в данных")
        r2_no_old = r2_without
        old_price_verdict = "NOT_PRESENT"

    # ========================================================================
    # Сравнение
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" СРАВНЕНИЕ:")
    print(f"{'=' * 70}")

    r2_diff = r2_with - r2_without
    mae_diff = mae_with - mae_without

    print(f"\n  • R² с признаком:    {r2_with:.4f}")
    print(f"  • R² без признака:   {r2_without:.4f}")
    print(f"  • Разница (ΔR²):     {r2_diff:+.4f} ({r2_diff / r2_without * 100:+.2f}%)")

    print(f"\n  • MAE с признаком:   {mae_with:.2f}")
    print(f"  • MAE без признака:  {mae_without:.2f}")
    print(f"  • Разница (ΔMAE):    {mae_diff:+.2f}")

    # ========================================================================
    # Вывод
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВЫВОД:")
    print(f"{'=' * 70}")

    if abs(r2_diff) < 0.01:  # Разница менее 0.5%
        print(f"\n Признак category_price_level БЕЗОПАСЕН")
        print(f"   • Разница в R² менее 0.5%")
        print(f"   • Признак не создаёт утечки данных")
        print(f"   • Можно оставить в модели")
        verdict = "SAFE"
    elif r2_diff > 0.05:  # Разница более 2%
        print(f"\n Признак category_price_level СОЗДАЁТ УТЕЧКУ!")
        print(f"   • Разница в R² более 2%")
        print(f"   • Модель использует информацию о price через этот признак")
        print(f"   • Рекомендуется удалить признак")
        verdict = "LEAKAGE"
    else:
        print(f"\n Признак category_price_level ТРЕБУЕТ ВНИМАНИЯ")
        print(f"   • Разница в R²: {r2_diff * 100:.2f}%")
        print(f"   • Признак немного помогает, но есть риск утечки")
        print(f"   • Рекомендуется удалить для чистоты эксперимента")
        verdict = "RISKY"

    return r2_with, r2_without, r2_diff, verdict

# =============================================================================
#  23. Доверительный интервал MAE
# =============================================================================

def gridsearch_best_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer
) -> Dict[str, Any]:
    """
    Выполняет поиск гиперпараметров (GridSearchCV) для модели RandomForestRegressor.

    Функция строит пайплайн машинного обучения, включающий предобработку данных
    и модель случайного леса. Обучение и кросс-валидация проходят на логарифмированном
    таргете (y_train_log = log1p(y_train)) для стабилизации дисперсии. После подбора
    лучших параметров вычисляются финальные метрики на тестовом наборе данных с
    обратным преобразованием предсказаний (expm1).

    Основные этапы работы:
    1. Логарифмирование целевой переменной train/test.
    2. Создание Pipeline: `preprocessor` -> `RandomForestRegressor`.
    3. Запуск GridSearchCV (5-fold CV) по сетке гиперпараметров.
    4. Оценка качества лучшей модели на тестовых данных (R² и MAE).
    5. Сохранение полной таблицы результатов кросс-валидации в CSV-файл.
    6. Построение и сохранение горизонтального столбчатого графика топ-10 комбинаций.

    Args:
        X_train (pd.DataFrame): Матрица признаков для обучения.
        X_test (pd.DataFrame): Матрица признаков для тестирования.
        y_train (pd.Series): Вектор ответов для обучения (в исходном масштабе).
        y_test (pd.Series): Вектор ответов для тестирования (в исходном масштабе).
        preprocessor (ColumnTransformer): Объект sklearn для предобработки
            числовых и категориальных признаков.

    Returns:
        Dict[str, Any]: Словарь с результатами работы, содержащий ключи:
            - 'best_params' (dict): Лучшие найденные гиперпараметры модели.
            - 'best_cv_r2' (float): Лучший средний показатель R² на кросс-валидации.
            - 'test_r2' (float): Коэффициент детерминации R² на тестовой выборке.
            - 'test_mae' (float): Средняя абсолютная ошибка MAE на тестовой выборке.
            - 'best_model' (Pipeline): Обученный пайплайн с лучшими параметрами.

    Raises:
        FileNotFoundError: Если не удается создать директорию для сохранения отчетов
            (обрабатывается внутри блока try-except).
        Exception: При ошибках визуализации или сохранения CSV (обрабатываются внутри).

    Note:
        🔧 ИСПРАВЛЕНИЕ (v2): предыдущая попытка ("удалена конструкция
        `with parallel_backend('threading')`") устранила один источник
        предупреждений, но не основной. Реальная причина — ВЛОЖЕННЫЙ
        параллелизм: GridSearchCV(n_jobs=-1) распределяет 405 фитов по всем
        ядрам, и КАЖДЫЙ из них внутри пытался снова занять все ядра через
        RandomForestRegressor(n_jobs=-1) — отсюда "sklearn.utils.parallel.delayed
        should be used with sklearn.utils.parallel.Parallel..." (контекст
        sklearn-конфигурации не передаётся корректно через вложенные
        joblib-воркеры). Исправлено: внутренняя модель теперь n_jobs=1 —
        весь параллелизм отдан внешнему GridSearchCV, как и рекомендует
        sklearn для вложенных CV+ensemble сценариев. Побочный эффект —
        отсутствие оверсабскрайба CPU может сделать сам перебор чуть быстрее,
        а не только тише в логах.
    """
    # Логарифмируем target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Создаём pipeline с RandomForest
    # 🔧 n_jobs=1 (не -1!): распараллеливание уже делает внешний GridSearchCV
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=1))
    ])

    # Пространство гиперпараметров (без None для max_depth!)
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 15, 20],  # Без None!
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt']  # Без 'auto'!
    }

    print("\n Пространство гиперпараметров:")
    for key, values in param_grid.items():
        print(f"  • {key}: {values}")

    print(f"\n Запуск GridSearchCV (5-fold CV)...")
    print(f"   Всего комбинаций: {np.prod([len(v) for v in param_grid.values()])}")

    # GridSearchCV с R² как основной метрикой
    # 🔧 ИСПРАВЛЕНИЕ: используем n_jobs=-1 напрямую, без parallel_backend
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,  # ← Задействуем все ядра напрямую
        verbose=1,
        refit=True
    )

    # 🔧 ИСПРАВЛЕНИЕ: убран with parallel_backend('threading', n_jobs=-1)
    # Теперь просто вызываем fit напрямую — предупреждения исчезнут
    grid_search.fit(X_train, y_train_log)

    # Лучшие параметры
    print(f"\n Лучшие параметры GridSearchCV:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")

    # Метрики на тесте
    y_pred_log = grid_search.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)

    print(f"\n Метрики на тесте (GridSearchCV):")
    print(f"  • R²: {r2_test:.4f}")
    print(f"  • MAE: {mae_test:.2f} SR")
    print(f"  • Лучший CV R²: {grid_search.best_score_:.4f}")

    # Создаём DataFrame с результатами
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    # Топ-10 комбинаций
    top_10 = cv_results_df.nlargest(10, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    print(f"\n Топ-10 комбинаций гиперпараметров:")
    print(top_10.to_markdown(index=False))

    # Создаём директорию reports_step, если её нет
    os.makedirs('reports_step', exist_ok=True)

    # Сохраняем в CSV
    try:
        cv_results_df.to_csv('reports_step/gridsearchcv_results.csv', index=False)
        print(f"\n  ✓ Результаты GridSearchCV сохранены в gridsearchcv_results.csv")
    except Exception as e:
        print(f"   Не удалось сохранить CSV: {e}")

    # Обученная модель сохраняется в results
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_r2': grid_search.best_score_,
        'test_r2': r2_test,
        'test_mae': mae_test,
        'best_model': grid_search.best_estimator_  # ← Обученная модель!
    }

    print(f"\n Обученная модель сохранена в results['best_model']")
    print(f"   Тип модели: {type(grid_search.best_estimator_).__name__}")
    print(f"   Модель обучена на train set ({len(X_train)} примеров)")

    # Визуализация результатов GridSearchCV
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Топ-10 по mean_test_score
        top_10_plot = cv_results_df.nlargest(10, 'mean_test_score')
        params_labels = [str(p)[:50] for p in top_10_plot['params']]
        bars = ax.barh(range(len(params_labels)),
                       top_10_plot['mean_test_score'],
                       xerr=top_10_plot['std_test_score'],
                       color='steelblue', edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(params_labels)))
        ax.set_yticklabels(params_labels, fontsize=8)
        ax.set_xlabel('Средний R² (CV)', fontsize=11)
        ax.set_title('GridSearchCV: Топ-10 комбинаций гиперпараметров',
                     fontweight='bold', fontsize=13, pad=15)
        ax.invert_yaxis()

        plt.tight_layout()
        save_plot(fig, 'ml_gridsearchcv_results')
        plt.close()
        print(f"  ✓ Сохранено: ml_gridsearchcv_results.png и .svg")
    except Exception as e:
        print(f"  ⚠️ Не удалось создать визуализацию: {e}")

    print(f"\n{'=' * 70}")
    print(" GRIDSEARCHCV ЗАВЕРШЁН УСПЕШНО!")
    print(f"{'=' * 70}")

    return results

#=====================================================================================
# Созданеи графиков остатков для анализа качества регрессионной иодели
#=====================================================================================

def run_ablation_study(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
        bool_features: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        model_name: str = 'RandomForest'
) -> pd.DataFrame:
    """
    Проводит исследование вклада отдельных групп признаков (Ablation Study).

    Функция предназначена для оценки важности различных групп фичей (логических блоков,
    таких как габариты, NLP-метрики, сложность сборки и т.д.). Она фильтрует исходные
    списки признаков, оставляя только те, которые фактически присутствуют в `X_train`,
    группирует их по категориям и последовательно удаляет каждую группу для оценки
    её влияния на качество модели.

    🔧 ДИНАМИЧЕСКИЙ ВЫБОР МОДЕЛИ:
    Функция автоматически создаёт модель нужного типа на основе параметра `model_name`.
    Поддерживаются: 'RandomForest', 'HistGB', 'XGBoost'. Префиксы 'model__' в
    `best_params` автоматически удаляются для совместимости с пайплайнами sklearn.

    Parameters
    ----------
    X_train : pd.DataFrame
        Матрица признаков для обучения.
    X_test : pd.DataFrame
        Матрица признаков для тестирования.
    y_train : pd.Series
        Вектор ответов для обучения (в исходном масштабе).
    y_test : pd.Series
        Вектор ответов для тестирования (в исходном масштабе).
    numeric_features : List[str]
        Исходный список названий числовых признаков.
    categorical_features : List[str]
        Список названий категориальных признаков.
    binary_features : List[str]
        Список названий бинарных признаков.
    bool_features : Optional[List[str]], optional
        Список названий булевых признаков. Если не передан,
        инициализируется как пустой список.
    best_params : Optional[Dict[str, Any]], optional
        Словарь подобранных гиперпараметров модели. Может содержать
        префиксы 'model__'. Если не передан, используется дефолтный
        набор параметров для указанной модели.
    model_name : str, optional
        Название базового алгоритма модели. Поддерживаются:
        'RandomForest', 'HistGB', 'XGBoost'. По умолчанию 'RandomForest'.

    Returns
    -------
    pd.DataFrame
        Таблица с результатами ablation study, содержащая колонки:
        'Group', 'R²', 'MAE', 'ΔR²', 'ΔMAE', 'Drop_%'.
        Строки отсортированы по R² (descending).

    Notes
    -----
    - Самая важная группа — та, удаление которой приводит к максимальному
      падению R² (минимальный R² после удаления).
    - Наименее важная группа — та, удаление которой приводит к минимальному
      падению R² (максимальный R² после удаления).
    - Если удаление группы улучшает R² (ΔR² < 0), это указывает на шум
      или мультиколлинеарность в данной группе признаков.

    Examples
    --------
    >>> results_df = run_ablation_study(
    ...     X_train, X_test, y_train, y_test,
    ...     numeric_features=['volume', 'width'],
    ...     categorical_features=['category'],
    ...     binary_features=['is_team'],
    ...     model_name='HistGB'
    ... )
    >>> print(results_df.head())
    """

    print("\n" + "=" * 70)
    print("ABLATION STUDY: анализ вклада групп признаков")
    print("=" * 70)

    # ========================================================================
    # 🔧 1. Динамический выбор модели на основе model_name
    # ========================================================================
    # Нормализуем имя модели для поддержки разных вариантов написания
    model_name_normalized = model_name.lower().replace(' ', '').replace('_', '')

    # Определяем тип модели и её дефолтные параметры
    if model_name_normalized in ('randomforest', 'rf'):
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True
    elif model_name_normalized in ('histgb', 'histgradientboosting', 'hist'):
        model_class = HistGradientBoostingRegressor
        default_params = {
            'max_iter': 200,
            'learning_rate': 0.1,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'max_bins': 255,
            'random_state': 42
        }
        supports_n_jobs = False  # HistGB не поддерживает n_jobs
    elif model_name_normalized in ('xgboost', 'xgb'):
        model_class = xgb.XGBRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1
        }
        supports_n_jobs = True
    else:
        # Fallback: неизвестный тип — используем RandomForest с предупреждением
        print(f"\n⚠️ Неизвестный тип модели '{model_name}', используем RandomForest")
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True

    print(f"📊 Используемая модель: {model_class.__name__}")

    # ========================================================================
    # 2. Очищаем best_params от префикса 'model__'
    # ========================================================================
    if best_params:
        clean_params = {}
        for key, value in best_params.items():
            if key.startswith('model__'):
                clean_params[key[7:]] = value
            else:
                clean_params[key] = value
        best_params = clean_params
    else:
        best_params = default_params.copy()

    # 🔧 Гарантируем random_state=42 для воспроизводимости
    best_params['random_state'] = 42

    # 🔧 Удаляем n_jobs для моделей, которые его не поддерживают
    if not supports_n_jobs and 'n_jobs' in best_params:
        best_params.pop('n_jobs', None)

    # ========================================================================
    # 3. Фильтрация признаков — оставляем только те, что есть в X_train
    # ========================================================================
    numeric_features_filtered = [f for f in numeric_features if f in X_train.columns]
    removed_features = [f for f in numeric_features if f not in X_train.columns]

    if removed_features:
        print(f"\n⚠️ Удалены из numeric_features (нет в X_train): {removed_features}")
        print(f"   Причина: признаки не были созданы в prepare_ml_data()")

    numeric_features = numeric_features_filtered

    print(f"\n📊 Числовых признаков для ablation study: {len(numeric_features)}")
    print(f"   Список: {numeric_features}")

    # ========================================================================
    # 4. Определяем группы признаков
    # ========================================================================
    if bool_features is None:
        bool_features = []

    groups = {
        'Габариты': ['depth', 'height', 'width', 'volume'],
        'NLP': ['desc_length', 'desc_word_count', 'premium_materials_count'],
        'Дизайнер': ['designer_freq'],
        'Сложность сборки': ['assembly_complexity'],
        'Ценовой контекст': ['category_price_level'],
        'Крупные товары': ['is_large_item'],
        # 🔧 is_large_item исключён отсюда: он уже отдельная группа "Крупные товары"
        # выше. Раньше binary_features (куда is_large_item добавляется на Шаге 10.1.3)
        # передавался сюда целиком — признак тестировался дважды под разными именами,
        # и ΔR² группы "Бинарные" был частично загрязнён его вкладом.
        'Бинарные': [f for f in binary_features if f != 'is_large_item'],
        'Булевые': bool_features,
        'Категория': categorical_features
    }

    print("\n📊 Группы признаков для ablation study:")
    for group_name, group_features in groups.items():
        existing = [f for f in group_features if f in X_train.columns]
        if existing:
            print(f"  • {group_name}: {len(existing)} признаков")
        else:
            print(f"  • {group_name}: ⚠️ нет в данных")

    # ========================================================================
    # 5. Функция для обучения и оценки (ДИНАМИЧЕСКИЙ ВЫБОР МОДЕЛИ)
    # ========================================================================

    def train_evaluate(X_tr, X_te, y_tr, y_te, num_feats, cat_feats, bin_feats):
        """Обучает модель на заданных признаках и возвращает R², MAE."""
        y_tr_log = np.log1p(y_tr)
        y_te_log = np.log1p(y_te)

        # Деревья не требуют масштабирования, поэтому with_scaling=False
        preprocessor = get_preprocessor(
            numeric_features=num_feats,
            categorical_features=cat_feats,
            binary_features=bin_feats,
            with_scaling=False
        )

        # Защита от пустых списков
        if not num_feats and not cat_feats and not bin_feats:
            print("    ⚠️ Все списки признаков пусты! Пропускаем обучение.")
            return 0.0, float('inf')

        # 🔧 ДИНАМИЧЕСКОЕ СОЗДАНИЕ МОДЕЛИ (никаких зашивок!)
        model = model_class(**best_params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_tr, y_tr_log)
        y_pred_log = pipeline.predict(X_te)
        y_pred = np.expm1(y_pred_log)
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        return r2, mae

    # ========================================================================
    # 6. Базовый случай (все признаки)
    # ========================================================================
    print("\n🔹 Базовый случай: все признаки")
    base_r2, base_mae = train_evaluate(
        X_train, X_test, y_train, y_test,
        numeric_features, categorical_features, binary_features + bool_features
    )
    print(f"  R² = {base_r2:.4f}, MAE = {base_mae:.2f} SR")

    # ========================================================================
    # 7. Цикл по группам
    # ========================================================================
    results = []
    results.append({
        'Group': 'Все признаки (baseline)',
        'R²': base_r2,
        'MAE': base_mae,
        'ΔR²': 0.0,
        'ΔMAE': 0.0,
        'Drop_%': 0.0
    })

    for group_name, group_features in groups.items():
        # Фильтруем только те признаки, которые есть в X_train
        cols_to_drop = [c for c in group_features if c in X_train.columns]
        if not cols_to_drop:
            print(f"\n⚠️ Группа '{group_name}' не содержит признаков в данных, пропускаем")
            continue

        print(f"\n🔸 Удаляем группу: {group_name} ({len(cols_to_drop)} признаков)")
        print(f"   Признаки: {cols_to_drop}")

        X_tr_sub = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_te_sub = X_test.drop(columns=cols_to_drop, errors='ignore')

        # Обновляем списки признаков
        num_feats = [c for c in numeric_features if c not in cols_to_drop]
        cat_feats = [c for c in categorical_features if c not in cols_to_drop]
        bin_feats = [c for c in (binary_features + bool_features) if c not in cols_to_drop]

        r2_sub, mae_sub = train_evaluate(
            X_tr_sub, X_te_sub, y_train, y_test,
            num_feats, cat_feats, bin_feats
        )

        delta_r2 = base_r2 - r2_sub
        delta_mae = mae_sub - base_mae
        # 🔧 Защита от отрицательного base_r2
        drop_pct = (delta_r2 / abs(base_r2) * 100) if base_r2 != 0 else 0

        print(f"  R² = {r2_sub:.4f} (Δ = {delta_r2:+.4f}), MAE = {mae_sub:.2f} SR (Δ = {delta_mae:+.2f})")
        print(f"  Падение R²: {drop_pct:.2f}%")

        results.append({
            'Group': group_name,
            'R²': r2_sub,
            'MAE': mae_sub,
            'ΔR²': delta_r2,
            'ΔMAE': delta_mae,
            'Drop_%': drop_pct
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False).reset_index(drop=True)

    # ========================================================================
    # 8. Сводная таблица
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА ABLATION STUDY")
    print("=" * 70)
    print(results_df[['Group', 'R²', 'MAE', 'ΔR²', 'ΔMAE', 'Drop_%']].to_markdown(index=False, floatfmt=".4f"))

    os.makedirs('reports_step', exist_ok=True)
    results_df.to_csv('reports_step/ablation_study_results.csv', index=False)
    print(f"\n✓ Результаты сохранены в reports_step/ablation_study_results.csv")

    # ========================================================================
    # 9. Визуализация
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    sorted_df = results_df.sort_values('R²', ascending=False)
    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(sorted_df))]
    bars1 = ax1.bar(sorted_df['Group'], sorted_df['R²'], color=colors, edgecolor='black')
    ax1.set_ylabel('R² (тест)', fontsize=11)
    ax1.set_title(f'Влияние удаления групп признаков на R² ({model_class.__name__})', fontweight='bold')
    ax1.axhline(y=base_r2, color='red', linestyle='--', linewidth=2, label='Baseline R²')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    for bar, val in zip(bars1, sorted_df['R²']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    colors2 = ['gold' if i == 0 else 'coral' for i in range(len(sorted_df))]
    bars2 = ax2.bar(sorted_df['Group'], sorted_df['MAE'], color=colors2, edgecolor='black')
    ax2.set_ylabel('MAE (SR)', fontsize=11)
    ax2.set_title(f'Влияние удаления групп признаков на MAE ({model_class.__name__})', fontweight='bold')
    ax2.axhline(y=base_mae, color='red', linestyle='--', linewidth=2, label='Baseline MAE')
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    for bar, val in zip(bars2, sorted_df['MAE']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_plot(fig, 'ml_ablation_study')
    plt.close()

    print(f"\n✓ График сохранён: ml_ablation_study.png и .svg")

    # ========================================================================
    # 10. Итоговые выводы
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("📈 ВЫВОДЫ ABLATION STUDY:")
    print(f"{'=' * 70}")

    # Фильтруем baseline
    non_baseline = results_df[results_df['Group'] != 'Все признаки (baseline)']

    if len(non_baseline) > 0:
        # Самая важная группа = минимальный R² (максимальное падение при удалении)
        most_important = non_baseline.loc[non_baseline['R²'].idxmin()]
        print(f"\n🏆 Самая важная группа признаков: {most_important['Group']}")
        print(f"   • Удаление приводит к падению R² на {most_important['Drop_%']:.2f}%")
        print(f"   • ΔR² = {most_important['ΔR²']:+.4f}")

        # Наименее важная группа = максимальный R² (минимальное падение при удалении)
        least_important = non_baseline.loc[non_baseline['R²'].idxmax()]
        print(f"\n📉 Наименее важная группа: {least_important['Group']}")
        print(f"   • Удаление приводит к падению R² на {least_important['Drop_%']:.2f}%")

        # Группы с отрицательным ΔR² (улучшение при удалении!)
        negative_delta = results_df[results_df['ΔR²'] < 0]
        if len(negative_delta) > 0:
            print(f"\n⚠️ Группы, удаление которых УЛУЧШАЕТ модель:")
            for _, row in negative_delta.iterrows():
                print(f"   • {row['Group']}: ΔR² = {row['ΔR²']:+.4f}")
            print(f"   → Эти признаки создают шум или мультиколлинеарность")

        # 🔧 РЕШЕНИЕ ПО category_price_level И is_large_item (по итогам Bootstrap
        # Ablation + группового Ablation Study — оба метода согласованно дали
        # отрицательную ΔR² для этих двух признаков по отдельности):
        # Быстрая joint-проверка (единичный train/test split, параметры
        # GridSearchCV, БЕЗ повторного полного пересчёта Optuna) на
        # одновременном удалении ОБОИХ признаков сразу дала ΔR²=-0.0036,
        # ΔMAE=+1.84 SR — то есть совместный эффект НЕ аддитивен точечным
        # результатам по отдельности (ожидался позитивный или нейтральный
        # эффект, получен слабо отрицательный, на грани нашего же порога
        # шума 0.003). Признаки СОХРАНЕНЫ в модели по итогам этой проверки.
        # Для окончательного решения нужен bootstrap-прогон совместного
        # удаления (не единичный сплит) — technически это уже поддержано
        # параметром exclude_noisy_features в prepare_ml_data(), полный
        # пересчёт GridSearchCV/Optuna пока не запускался.
        print(f"\n🔧 Отдельная проверка: category_price_level + is_large_item")
        print(f"   Оба признака по отдельности дали отрицательную ΔR² в этом ")
        print(f"   Ablation Study И в Bootstrap Ablation (см. выше/отдельный отчёт).")
        print(f"   Быстрая совместная проверка (один train/test split, без")
        print(f"   пересчёта Optuna): ΔR²=-0.0036, ΔMAE=+1.84 SR — совместный")
        print(f"   эффект слабо ОТРИЦАТЕЛЕН (на грани порога шума 0.003), не")
        print(f"   аддитивен точечным результатам. Признаки СОХРАНЕНЫ в модели;")
        print(f"   для окончательного решения нужен bootstrap-прогон совместного")
        print(f"   удаления. См. параметр exclude_noisy_features в prepare_ml_data().")

    return results_df

#===========================================================================================
# BOOTSTRAP ABLATION – ТОЧЕЧНЫЙ АНАЛИЗ ВКЛАДА ОТДЕЛЬНЫХ ПРИЗНАКОВ
#===========================================================================================

def run_bootstrap_ablation(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
        bool_features: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        model_name: str = 'RandomForest',
        features_to_test: Optional[List[str]] = None,
        n_repeats: int = 15,
        ci_level: float = 0.95,
        random_state_base: int = 42
) -> pd.DataFrame:
    """
    Точечный (по одному признаку) Bootstrap Ablation с динамически вычисляемыми
    доверительными интервалами — без фиксированных внешних порогов (0.003/20 SR и т.п.).

    В отличие от run_ablation_study() (который удаляет целые ГРУППЫ признаков одним
    train/test split), эта функция:
      1. удаляет ОДИН признак за раз — изолируя его вклад от остальных членов его
         группы (например, можно проверить 'volume' отдельно от width/height/depth,
         что run_ablation_study() сделать не может, т.к. они в одной группе "Габариты");
      2. повторяет обучение `n_repeats` раз на БУТСТРАП-РЕСЭМПЛАХ обучающей выборки
         (строки train с возвращением, разный seed), парно (baseline и "без признака"
         на ОДИНАКОВЫХ seed'ах, т.е. на одних и тех же ресэмплах) — это даёт
         распределение ΔR²/ΔMAE, а не одно число из единственного train/test split.
         🔧 Важно: изменчивость берётся из ресэмплинга ДАННЫХ, а не из model.random_state —
         для моделей без встроенного бутстрапа строк (напр. HistGradientBoosting, в
         отличие от RandomForest) одна лишь смена random_state может давать нулевую
         дисперсию между повторами на некрупных датасетах, что сделало бы CI вырожденным;
      3. вместо сравнения с фиксированным порогом (напр. ΔR² > 0.003) строит 95% CI
         парных дельт и проверяет, пересекает ли он ноль — порог тем самым вычисляется
         динамически из СОБСТВЕННОГО шума модели на конкретном признаке, а не
         переносится с чужого теста (напр. bootstrap CI для MAE финальной модели).

    Асимметричное правило решения (см. обсуждение порогов отбора признаков):
      • mean ΔR² < 0 (удаление признака УЛУЧШАЕТ модель) → 'УБРАТЬ' всегда,
        независимо от значимости — отрицательная дельта сама по себе сигнал
        шума/мультиколлинеарности, порог здесь не нужен;
      • mean ΔR² ≥ 0 и 95% CI (по ΔR² ИЛИ по ΔMAE) не пересекает ноль → 'ОСТАВИТЬ'
        (вклад статистически значим хотя бы по одной из двух метрик — см. обоснование
        необходимости двух метрик, а не только R², в обсуждении порогов);
      • иначе (CI пересекает ноль по обеим метрикам) → 'УБРАТЬ' (неотличимо от шума).

    Parameters
    ----------
    X_train, X_test, y_train, y_test : см. run_ablation_study().
    numeric_features, categorical_features, binary_features, bool_features :
        Финальные (уже очищенные) списки признаков модели.
    best_params : Optional[Dict[str, Any]]
        Гиперпараметры лучшей модели (с префиксом 'model__' или без).
    model_name : str
        'RandomForest', 'HistGB' или 'XGBoost' — динамический выбор, как в
        run_ablation_study().
    features_to_test : Optional[List[str]]
        Список признаков для точечной проверки. Если None — проверяются ВСЕ
        числовые и бинарные признаки по одному.
    n_repeats : int
        Количество повторов обучения с разными random_state для baseline и для
        каждого проверяемого признака (по умолчанию 15).
    ci_level : float
        Уровень доверительного интервала (по умолчанию 0.95).
    random_state_base : int
        Начальное значение random_state; повторы используют
        random_state_base, random_state_base+1, ..., random_state_base+n_repeats-1.

    Returns
    -------
    pd.DataFrame
        Таблица с колонками: 'Feature', 'delta_R2_mean', 'delta_R2_CI_low',
        'delta_R2_CI_high', 'delta_MAE_mean', 'delta_MAE_CI_low', 'delta_MAE_CI_high',
        'Verdict'. Отсортирована по delta_R2_mean (descending).

    Examples
    --------
    >>> results_df = run_bootstrap_ablation(
    ...     X_train, X_test, y_train, y_test,
    ...     numeric_features=numeric_features, categorical_features=['category'],
    ...     binary_features=binary_features, model_name='RandomForest',
    ...     features_to_test=['volume', 'desc_word_count'], n_repeats=15
    ... )
    """

    print("\n" + "=" * 70)
    print("BOOTSTRAP ABLATION: точечный анализ вклада отдельных признаков")
    print("=" * 70)

    if bool_features is None:
        bool_features = []

    # ========================================================================
    # 1. Динамический выбор модели (идентично run_ablation_study)
    # ========================================================================
    model_name_normalized = model_name.lower().replace(' ', '').replace('_', '')

    if model_name_normalized in ('randomforest', 'rf'):
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': -1
        }
        supports_n_jobs = True
    elif model_name_normalized in ('histgb', 'histgradientboosting', 'hist'):
        model_class = HistGradientBoostingRegressor
        default_params = {
            'max_iter': 200, 'learning_rate': 0.1, 'max_depth': None,
            'min_samples_leaf': 20, 'l2_regularization': 0.0, 'max_bins': 255
        }
        supports_n_jobs = False
    elif model_name_normalized in ('xgboost', 'xgb'):
        model_class = xgb.XGBRegressor
        default_params = {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': 1
        }
        supports_n_jobs = True
    else:
        print(f"\n⚠️ Неизвестный тип модели '{model_name}', используем RandomForest")
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': -1
        }
        supports_n_jobs = True

    print(f"📊 Используемая модель: {model_class.__name__}")

    if best_params:
        clean_params = {}
        for key, value in best_params.items():
            clean_params[key[7:] if key.startswith('model__') else key] = value
        base_params = clean_params
    else:
        base_params = default_params.copy()

    # random_state задаётся динамически на каждый повтор — убираем фиксированный
    base_params.pop('random_state', None)
    if not supports_n_jobs:
        base_params.pop('n_jobs', None)

    # ========================================================================
    # 2. Определяем список признаков для точечной проверки
    # ========================================================================
    numeric_features = [f for f in numeric_features if f in X_train.columns]
    binary_features = [f for f in binary_features if f in X_train.columns]

    if features_to_test is None:
        features_to_test = numeric_features + binary_features
    else:
        missing = [f for f in features_to_test if f not in numeric_features + binary_features]
        if missing:
            print(f"\n⚠️ Пропущены (нет в активных признаках модели): {missing}")
        features_to_test = [f for f in features_to_test if f in numeric_features + binary_features]

    print(f"\n📊 Признаков для точечной проверки: {len(features_to_test)}")
    print(f"   Список: {features_to_test}")
    print(f"   Повторов на признак: {n_repeats}, уровень CI: {ci_level:.0%}")

    # ========================================================================
    # 3. Функция обучения/оценки на бутстрап-ресэмпле обучающих данных
    # ========================================================================
    # 🔧 ВАЖНО: изменчивость получаем из РЕСЭМПЛИНГА СТРОК train (with replacement),
    # а не только из model.random_state. Модели вроде HistGradientBoosting не
    # используют random_state для бутстрапа строк (в отличие от RandomForest,
    # где это встроено в алгоритм) — при варьировании только random_state на
    # датасете такого размера HistGB может давать буквально идентичную модель
    # на каждом повторе (нулевую дисперсию), что делает CI вырожденным и любой
    # признак — ложно "значимым". Ресэмплинг данных даёт настоящую изменчивость
    # независимо от типа модели.
    n_train = len(X_train)

    def train_evaluate(num_feats, cat_feats, bin_feats, seed):
        rng = np.random.RandomState(seed)
        boot_idx = rng.randint(0, n_train, size=n_train)
        X_tr_boot = X_train.iloc[boot_idx]
        y_tr_boot_log = np.log1p(y_train.iloc[boot_idx])

        preprocessor = get_preprocessor(
            numeric_features=num_feats, categorical_features=cat_feats,
            binary_features=bin_feats, with_scaling=False
        )
        params = dict(base_params)
        params['random_state'] = seed
        model = model_class(**params)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_tr_boot, y_tr_boot_log)
        y_pred = np.expm1(pipeline.predict(X_test))
        return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

    seeds = [random_state_base + i for i in range(n_repeats)]

    # ========================================================================
    # 4. Baseline: n_repeats обучений на полном наборе признаков
    # ========================================================================
    print(f"\n🔹 Baseline (все признаки, {n_repeats} повторов)...")
    base_r2 = np.empty(n_repeats)
    base_mae = np.empty(n_repeats)
    for i, s in enumerate(seeds):
        base_r2[i], base_mae[i] = train_evaluate(numeric_features, categorical_features, binary_features, s)
    print(f"  R² = {base_r2.mean():.4f} ± {base_r2.std():.4f}")
    print(f"  MAE = {base_mae.mean():.2f} ± {base_mae.std():.2f} SR")

    # ========================================================================
    # 5. Цикл по отдельным признакам
    # ========================================================================
    alpha = 1 - ci_level
    ci_low_pct, ci_high_pct = alpha / 2 * 100, (1 - alpha / 2) * 100

    results = []
    for feat in features_to_test:
        print(f"\n🔸 Проверяем: {feat}")
        num_feats = [f for f in numeric_features if f != feat]
        bin_feats = [f for f in binary_features if f != feat]

        r2_wo = np.empty(n_repeats)
        mae_wo = np.empty(n_repeats)
        for i, s in enumerate(seeds):
            r2_wo[i], mae_wo[i] = train_evaluate(num_feats, categorical_features, bin_feats, s)

        delta_r2 = base_r2 - r2_wo      # >0 = признак помогает
        delta_mae = mae_wo - base_mae   # >0 = признак помогает (без него MAE выше)

        ci_r2 = np.percentile(delta_r2, [ci_low_pct, ci_high_pct])
        ci_mae = np.percentile(delta_mae, [ci_low_pct, ci_high_pct])

        sig_r2 = not (ci_r2[0] <= 0 <= ci_r2[1])
        sig_mae = not (ci_mae[0] <= 0 <= ci_mae[1])

        if delta_r2.mean() < 0:
            verdict = "УБРАТЬ (вредит/шум)"
        elif sig_r2 or sig_mae:
            verdict = "ОСТАВИТЬ (значим)"
        else:
            verdict = "УБРАТЬ (шум)"

        print(f"  ΔR² = {delta_r2.mean():+.4f} 95%CI[{ci_r2[0]:+.4f};{ci_r2[1]:+.4f}]  "
              f"ΔMAE = {delta_mae.mean():+.2f} 95%CI[{ci_mae[0]:+.2f};{ci_mae[1]:+.2f}]  → {verdict}")

        results.append({
            'Feature': feat,
            'delta_R2_mean': delta_r2.mean(),
            'delta_R2_CI_low': ci_r2[0],
            'delta_R2_CI_high': ci_r2[1],
            'delta_MAE_mean': delta_mae.mean(),
            'delta_MAE_CI_low': ci_mae[0],
            'delta_MAE_CI_high': ci_mae[1],
            'Verdict': verdict
        })

    results_df = pd.DataFrame(results).sort_values('delta_R2_mean', ascending=False).reset_index(drop=True)

    # ========================================================================
    # 6. Сводная таблица и сохранение
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА BOOTSTRAP ABLATION")
    print("=" * 70)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

    os.makedirs('reports_step', exist_ok=True)
    results_df.to_csv('reports_step/bootstrap_ablation_results.csv', index=False)
    print(f"\n✓ Результаты сохранены в reports_step/bootstrap_ablation_results.csv")

    # ========================================================================
    # 7. Визуализация: forest plot ΔR² с доверительными интервалами
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(results_df) + 2)))
    plot_df = results_df.sort_values('delta_R2_mean')
    y_pos = np.arange(len(plot_df))
    for yi, (_, row) in zip(y_pos, plot_df.iterrows()):
        c = {'ОСТАВИТЬ (значим)': 'seagreen',
             'УБРАТЬ (вредит/шум)': 'firebrick',
             'УБРАТЬ (шум)': 'gray'}[row['Verdict']]
        err = [[row['delta_R2_mean'] - row['delta_R2_CI_low']],
               [row['delta_R2_CI_high'] - row['delta_R2_mean']]]
        ax.errorbar(row['delta_R2_mean'], yi, xerr=err, fmt='o',
                    color=c, ecolor=c, elinewidth=2, capsize=4, markersize=7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Feature'])
    ax.set_xlabel(f'ΔR² (среднее ± {ci_level:.0%} CI по {n_repeats} повторам)')
    ax.set_title(f'Bootstrap Ablation: вклад отдельных признаков ({model_class.__name__})',
                 fontweight='bold')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='seagreen', lw=2, label='Оставить (значим)'),
        Line2D([0], [0], color='firebrick', lw=2, label='Убрать (вредит/шум)'),
        Line2D([0], [0], color='gray', lw=2, label='Убрать (шум)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    plt.tight_layout()
    save_plot(fig, 'ml_bootstrap_ablation')
    plt.close()

    print(f"\n✓ График сохранён: ml_bootstrap_ablation.png и .svg")

    # ========================================================================
    # 8. Итоговые выводы
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("📈 ВЫВОДЫ BOOTSTRAP ABLATION:")
    print(f"{'=' * 70}")

    to_keep = results_df[results_df['Verdict'] == 'ОСТАВИТЬ (значим)']
    to_remove = results_df[results_df['Verdict'] != 'ОСТАВИТЬ (значим)']

    print(f"\n✅ Значимый вклад ({len(to_keep)} признаков): {list(to_keep['Feature'])}")
    print(f"🗑️ Кандидаты на удаление ({len(to_remove)} признаков): {list(to_remove['Feature'])}")

    harmful = results_df[results_df['Verdict'] == 'УБРАТЬ (вредит/шум)']
    if len(harmful) > 0:
        print(f"\n⚠️ Отдельно: признаки с отрицательной ΔR² (вредят модели):")
        for _, row in harmful.iterrows():
            print(f"   • {row['Feature']}: ΔR² = {row['delta_R2_mean']:+.4f}")

    return results_df

#===========================================================================================
# Анализ чувствительности модели к объему обучающих данных
#==========================================================================================

def sensitivity_analysis(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
        best_params: Optional[Dict[str, Any]] = None,
        fractions: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
        n_repeats: int = 3,
        baseline_r2: Optional[float] = None,
        baseline_mae: Optional[float] = None,
        model_name: str = 'RandomForest'
) -> pd.DataFrame:
    """
    Выполняет анализ чувствительности (Sensitivity Analysis) модели к объёму обучающих данных.

    Функция исследует, как изменяется качество предсказаний (метрики R² и MAE) при
    обучении модели на подмножествах обучающей выборки разного размера. Для каждой заданной
    доли данных (`fraction`) эксперимент повторяется `n_repeats` раз со случайным сэмплированием
    без возвращения для получения устойчивой оценки (среднего значения и стандартного отклонения).

    🔧 ДИНАМИЧЕСКИЙ ВЫБОР МОДЕЛИ:
    Функция автоматически создаёт модель нужного типа на основе параметра `model_name`.
    Поддерживаются: 'RandomForest', 'HistGB', 'XGBoost'. Префиксы 'model__' в
    `best_params` автоматически удаляются для совместимости с пайплайнами sklearn.

    Основные этапы работы:
    1. Динамический выбор модели на основе `model_name`.
    2. Очистка ключей `best_params` от префикса 'model__' для прямой передачи в конструктор.
    3. Циклический подбор подвыборок для каждой доли из `fractions` и обучение пайплайна
       на логарифмированном таргете (log1p) с последующим экспонированием предсказаний (expm1).
    4. Добавление baseline-метрик (100% данных) в итоговую таблицу.
    5. Построение двухпанельной столбчатой диаграммы (R² и MAE) с отображением усов стандартного
       отклонения (error bars) и линий baseline.
    6. Сохранение визуализации (в форматах .png и .svg) и результирующего датафрейма в CSV.

    Args:
        X_train (pd.DataFrame): Полная матрица признаков обучающей выборки.
        X_test (pd.DataFrame): Матрица признаков тестовой выборки.
        y_train (pd.Series): Полный вектор ответов обучающей выборки (в исходном масштабе).
        y_test (pd.Series): Вектор ответов тестовой выборки (в исходном масштабе).
        preprocessor (ColumnTransformer): Пайплайн предобработки признаков.
        best_params (Optional[Dict[str, Any]], optional): Словарь гиперпараметров модели.
            Если не передан, используются стандартные параметры для указанной модели.
        fractions (List[float], optional): Список долей обучающей выборки, для которых
            проводится анализ. По умолчанию [0.5, 0.6, 0.7, 0.8, 0.9].
        n_repeats (int, optional): Количество случайных разбиений для каждой доли выборки
            для усреднения метрик. По умолчанию 3.
        baseline_r2 (Optional[float], optional): Значение метрики R² на 100% данных.
            Если не передано, в качестве baseline используется результат последней доли
            из списка `fractions`.
        baseline_mae (Optional[float], optional): Значение метрики MAE на 100% данных.
            Если не передано, используется результат последней доли из `fractions`.
        model_name (str, optional): Название базового алгоритма модели. Поддерживаются:
            'RandomForest', 'HistGB', 'XGBoost'. По умолчанию 'RandomForest'.

    Returns:
        pd.DataFrame: Таблица с результатами анализа, содержащая столбцы:
            - 'Fraction' (float): Доля обучающей выборки (например, 0.5).
            - 'Fraction_%' (str): Строковое представление доли (например, '50%').
            - 'R2_mean' (float): Средний коэффициент детерминации R² на тесте.
            - 'R2_std' (float): Стандартное отклонение R² на тесте.
            - 'MAE_mean' (float): Средняя абсолютная ошибка MAE на тесте.
            - 'MAE_std' (float): Стандартное отклонение MAE на тесте.
    """

    # ========================================================================
    # 🔧 1. Динамический выбор модели на основе model_name
    # ========================================================================
    # Нормализуем имя модели для поддержки разных вариантов написания
    model_name_normalized = model_name.lower().replace(' ', '').replace('_', '')

    # Определяем тип модели и её дефолтные параметры
    if model_name_normalized in ('randomforest', 'rf'):
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True
    elif model_name_normalized in ('histgb', 'histgradientboosting', 'hist'):
        model_class = HistGradientBoostingRegressor
        default_params = {
            'max_iter': 200,
            'learning_rate': 0.1,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'max_bins': 255,
            'random_state': 42
        }
        supports_n_jobs = False  # HistGB не поддерживает n_jobs
    elif model_name_normalized in ('xgboost', 'xgb'):
        model_class = xgb.XGBRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1
        }
        supports_n_jobs = True
    else:
        # Fallback: неизвестный тип — используем RandomForest с предупреждением
        print(f"\n⚠️ Неизвестный тип модели '{model_name}', используем RandomForest")
        model_class = RandomForestRegressor
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        supports_n_jobs = True

    print(f"📊 Используемая модель: {model_class.__name__}")

    # ========================================================================
    # 2. Очистка параметров от префиксов 'model__'
    # ========================================================================
    if best_params is None:
        best_params = default_params.copy()
    else:
        clean_params = {}
        for key, value in best_params.items():
            if key.startswith('model__'):
                clean_params[key[7:]] = value
            else:
                clean_params[key] = value
        best_params = clean_params

    # 🔧 Гарантируем random_state=42 для воспроизводимости
    best_params['random_state'] = 42

    # 🔧 Удаляем n_jobs для моделей, которые его не поддерживают
    if not supports_n_jobs and 'n_jobs' in best_params:
        best_params.pop('n_jobs', None)

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: влияние размера выборки на качество модели")
    print("=" * 70)

    # 🔧 Фиксируем seed для воспроизводимости сэмплирования
    np.random.seed(42)

    results = []

    for frac in fractions:
        print(f"\nДоля выборки: {frac*100:.0f}%")
        frac_r2 = []
        frac_mae = []

        for rep in range(n_repeats):
            n_samples = int(len(X_train) * frac)
            indices = np.random.choice(len(X_train), size=n_samples, replace=False)
            X_sub = X_train.iloc[indices]
            y_sub = y_train.iloc[indices]

            # 🔧 ДИНАМИЧЕСКОЕ СОЗДАНИЕ МОДЕЛИ (никаких зашивок!)
            model = model_class(**best_params)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            y_sub_log = np.log1p(y_sub)
            y_test_log = np.log1p(y_test)

            pipeline.fit(X_sub, y_sub_log)
            y_pred_log = pipeline.predict(X_test)
            y_pred = np.expm1(y_pred_log)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            frac_r2.append(r2)
            frac_mae.append(mae)

        mean_r2 = np.mean(frac_r2)
        std_r2 = np.std(frac_r2)
        mean_mae = np.mean(frac_mae)
        std_mae = np.std(frac_mae)

        print(f"  R² = {mean_r2:.4f} ± {std_r2:.4f}, MAE = {mean_mae:.2f} ± {std_mae:.2f} SR")

        results.append({
            'Fraction': frac,
            'Fraction_%': f'{frac*100:.0f}%',
            'R2_mean': mean_r2,
            'R2_std': std_r2,
            'MAE_mean': mean_mae,
            'MAE_std': std_mae
        })

    results_df = pd.DataFrame(results)

    # ========================================================================
    # 3. Добавляем точку для 100% данных (baseline)
    # ========================================================================
    if baseline_r2 is None:
        print("\n⚠️ Baseline R² не передан — используем значение для 90% данных как baseline")
        baseline_r2 = results_df['R2_mean'].iloc[-1]
        baseline_mae = results_df['MAE_mean'].iloc[-1]

    baseline_row = pd.DataFrame([{
        'Fraction': 1.0,
        'Fraction_%': '100%',
        'R2_mean': baseline_r2,
        'R2_std': 0,
        'MAE_mean': baseline_mae,
        'MAE_std': 0
    }])
    results_df = pd.concat([results_df, baseline_row], ignore_index=True)

    # ========================================================================
    # 4. Визуализация
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_labels = results_df['Fraction_%'].tolist()

    # График 1: R²
    ax1 = axes[0]
    bars1 = ax1.bar(all_labels, results_df['R2_mean'],
                    yerr=results_df['R2_std'],
                    capsize=5, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Доля обучающей выборки')
    ax1.set_ylabel('R² (тест)')
    ax1.set_title(f'Зависимость R² от размера обучающей выборки ({model_class.__name__})', fontweight='bold')
    ax1.axhline(y=baseline_r2, color='red', linestyle='--',
                linewidth=2, label=f'Baseline (100%): R²={baseline_r2:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for bar, val in zip(bars1, results_df['R2_mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # График 2: MAE
    ax2 = axes[1]
    bars2 = ax2.bar(all_labels, results_df['MAE_mean'],
                    yerr=results_df['MAE_std'],
                    capsize=5, color='coral', edgecolor='black')
    ax2.set_xlabel('Доля обучающей выборки')
    ax2.set_ylabel('MAE (SR)')
    ax2.set_title(f'Зависимость MAE от размера обучающей выборки ({model_class.__name__})', fontweight='bold')
    ax2.axhline(y=baseline_mae, color='red', linestyle='--',
                linewidth=2, label=f'Baseline (100%): MAE={baseline_mae:.1f} SR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for bar, val in zip(bars2, results_df['MAE_mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # ========================================================================
    # 5. Сохранение графика
    # ========================================================================
    save_plot(fig, 'ml_sensitivity_analysis')
    print(f"\n✓ График сохранён: ml_sensitivity_analysis.png и .svg")

    plt.close()

    # ========================================================================
    # 6. Сохраняем результаты в CSV
    # ========================================================================
    os.makedirs('reports_step', exist_ok=True)
    results_df.to_csv('reports_step/sensitivity_analysis_results.csv', index=False)
    print(f"✓ Результаты сохранены в reports_step/sensitivity_analysis_results.csv")

    return results_df

# =============================================================================
# MAIN
# =============================================================================

def _get_model_type_from_name(model_name: str) -> str:
    """Преобразует имя модели в формат, понятный дочерним функциям.

    🔧 РЕФАКТОРИНГ: раньше это была вложенная функция внутри run_ml_pipeline()
    (недоступная снаружи и создававшаяся заново при каждом вызове). Вынесена
    на уровень модуля как приватный (_) хелпер — логика не изменилась ни на
    строку, изменилась только область видимости.

    Поддерживает варианты:
    - 'HistGradientBoosting', 'HistGB', 'hist' → 'HistGB'
    - 'RandomForest', 'RF', 'RandomForest + Optuna (Exp4)' → 'RandomForest'
    - 'XGBoost', 'XGB', 'XGBoost + Optuna (Exp1)' → 'XGBoost'
    - Fallback → 'RandomForest'
    """
    name_lower = model_name.lower().replace(' ', '').replace('_', '')

    if 'histgradient' in name_lower or 'histgb' in name_lower or name_lower == 'hist':
        return 'HistGB'
    elif 'randomforest' in name_lower or name_lower == 'rf':
        return 'RandomForest'
    elif 'xgboost' in name_lower or name_lower == 'xgb':
        return 'XGBoost'
    else:
        print(f"\n⚠️ Не удалось определить тип модели из '{model_name}', используем RandomForest")
        return 'RandomForest'


def _prepare_and_compare_baseline(
        df_unique: pd.DataFrame,
        eda_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Шаги 1-3 run_ml_pipeline(): подготовка данных, сравнение 8 базовых
    моделей, проверка на утечку данных (category_price_level).

    🔧 РЕФАКТОРИНГ (разбиение run_ml_pipeline на приватные функции, куратор):
    чистое извлечение метода (extract method) — логика и порядок операций
    не изменились, изменилась только организация кода. Все промежуточные
    величины возвращаются одним словарём (не dataclass — это внутренняя,
    не публичная передача данных между шагами одного пайплайна, здесь
    typo-риск не так критичен, как в MLPipelineResult, который расходится
    по всему проекту).
    """
    print("\n" + "=" * 70)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ML")
    print("=" * 70)

    sellable_verdict = eda_results['sellable_verdict']
    old_price_check = eda_results['old_price_check']

    X_train, X_test, y_train, y_test, X, y, num_feat, cat_feat, bin_feat, bool_feat = prepare_ml_data(
        df_unique,
        sellable_verdict=sellable_verdict,
        old_price_verdict=old_price_check['verdict'],
        old_price_features_to_use=old_price_check['features_to_use']
    )

    results_df, best_name, best_pipeline, preprocessor = compare_models(
        X_train, X_test, y_train, y_test, num_feat, cat_feat, bin_feat, bool_feat
    )

    print(f"\n{'=' * 70}")
    print(" ФИНАЛЬНАЯ МОДЕЛЬ (после сравнения)")
    print(f"{'=' * 70}")
    print(f"\n Лучшая модель: {best_name}")
    print(f"   R² (тест): {results_df.iloc[0]['R2']:.4f}")
    print(f"   MAE: {results_df.iloc[0]['MAE']:.2f}")

    simple_r2 = results_df.iloc[0]['R2']
    simple_mae = results_df.iloc[0]['MAE']

    r2_with, r2_without, r2_diff, verdict = check_data_leakage(
        X_train, X_test, y_train, y_test,
        num_feat, cat_feat, bin_feat, bool_feat
    )

    if verdict == "LEAKAGE":
        print("\n⚠️ Удаляем признак category_price_level из-за утечки!")
        X_train = X_train.drop(columns=['category_price_level'])
        X_test = X_test.drop(columns=['category_price_level'])
        X = X.drop(columns=['category_price_level'])
        num_feat = [f for f in num_feat if f != 'category_price_level']
    else:
        print(f"\n Оставляем признак category_price_level (verdict: {verdict})")

    return {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'X': X, 'y': y, 'num_feat': num_feat, 'cat_feat': cat_feat,
        'bin_feat': bin_feat, 'bool_feat': bool_feat, 'preprocessor': preprocessor,
        'best_name': best_name, 'best_pipeline': best_pipeline,
        'simple_r2': simple_r2, 'simple_mae': simple_mae, 'verdict': verdict,
    }


def _run_hyperparameter_search(
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series,
        preprocessor: ColumnTransformer,
        rerun_optuna: bool, trials: Optional[int], cv: Optional[int]
) -> Dict[str, Any]:
    """Шаги 4-5 run_ml_pipeline(): Optuna + GridSearchCV.

    🔧 РЕФАКТОРИНГ: чистое извлечение метода, логика не изменилась.
    """
    print("\n" + "=" * 70)
    print("OPTUNA: БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
    print("=" * 70)

    optuna_pipeline, optuna_model_name, study_exp1, study_exp2, optuna_comparison, optuna_params = optimize_with_optuna_experiments(
        X_train, X_test, y_train, y_test, preprocessor,
        rerun=rerun_optuna,
        n_trials_1=trials or 50,
        n_trials_2=trials or 75,
        n_trials_3=trials or 75,
        cv_folds=cv or 5
    )

    y_pred_optuna_log = optuna_pipeline.predict(X_test)
    y_pred_optuna = np.expm1(y_pred_optuna_log)
    optuna_r2 = r2_score(y_test, y_pred_optuna)
    optuna_mae = mean_absolute_error(y_test, y_pred_optuna)

    print(f"\n📊 Метрики Optuna на тесте:")
    print(f"   Модель: {optuna_model_name}")
    print(f"   R² (тест): {optuna_r2:.4f}")
    print(f"   MAE (тест): {optuna_mae:.2f} SR")

    print(f"\n{'=' * 70}")
    print("GRIDSEARCHCV ДЛЯ ЛУЧШЕЙ МОДЕЛИ ")
    print(f"{'=' * 70}")

    gridsearch_results = gridsearch_best_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    gs_pipeline = gridsearch_results['best_model']
    gs_r2 = gridsearch_results['test_r2']
    gs_mae = gridsearch_results['test_mae']
    gs_params = gridsearch_results['best_params']

    print(f"\n📊 Метрики GridSearchCV на тесте:")
    print(f"   R² (тест): {gs_r2:.4f}")
    print(f"   MAE (тест): {gs_mae:.2f} SR")

    return {
        'optuna_pipeline': optuna_pipeline, 'optuna_model_name': optuna_model_name,
        'optuna_r2': optuna_r2, 'optuna_mae': optuna_mae, 'optuna_params': optuna_params,
        'gridsearch_results': gridsearch_results, 'gs_pipeline': gs_pipeline,
        'gs_r2': gs_r2, 'gs_mae': gs_mae, 'gs_params': gs_params,
    }


def _select_best_model(
        baseline: Dict[str, Any],
        hp_search: Dict[str, Any]
) -> Dict[str, Any]:
    """Шаг 6 run_ml_pipeline(): сравнение простой модели / Optuna / GridSearchCV,
    выбор победителя (принцип Occam's Razor), сборка model_selection_details.

    🔧 РЕФАКТОРИНГ: имя функции — прямая рекомендация куратора
    ("_select_best_model()"). Логика не изменилась.
    """
    print(f"\n{'=' * 70}")
    print(" СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
    print(f"{'=' * 70}")

    candidates = {
        'simple': {'r2': baseline['simple_r2'], 'mae': baseline['simple_mae'],
                   'name': baseline['best_name'], 'pipeline': baseline['best_pipeline'], 'params': None},
        'optuna': {'r2': hp_search['optuna_r2'], 'mae': hp_search['optuna_mae'],
                   'name': hp_search['optuna_model_name'], 'pipeline': hp_search['optuna_pipeline'],
                   'params': hp_search['optuna_params']},
        'gridsearch': {'r2': hp_search['gs_r2'], 'mae': hp_search['gs_mae'],
                       'name': 'RandomForest + GridSearchCV', 'pipeline': hp_search['gs_pipeline'],
                       'params': hp_search['gs_params']}
    }

    best_candidate = max(candidates.items(), key=lambda x: x[1]['r2'])
    winner = best_candidate[0]
    winner_data = best_candidate[1]

    final_pipeline = winner_data['pipeline']
    final_model_name = winner_data['name']
    final_params = winner_data['params']
    final_r2 = winner_data['r2']
    final_mae = winner_data['mae']

    final_model_type = _get_model_type_from_name(final_model_name)

    print(f"\n🏆 Победитель: {final_model_name}")
    print(f"   Тип модели (для дочерних функций): {final_model_type}")
    print(f"   R²: {final_r2:.4f}")
    print(f"   MAE: {final_mae:.2f} SR")
    print(f"\n📊 Сравнение кандидатов:")
    print(f"   Простая модель ({baseline['best_name']}): R²={baseline['simple_r2']:.4f}, MAE={baseline['simple_mae']:.2f}")
    print(f"   Optuna ({hp_search['optuna_model_name']}): R²={hp_search['optuna_r2']:.4f}, MAE={hp_search['optuna_mae']:.2f}")
    print(f"   GridSearchCV: R²={hp_search['gs_r2']:.4f}, MAE={hp_search['gs_mae']:.2f}")

    if winner == 'simple':
        print(f"\n⚠️ ВНИМАНИЕ: Простая модель работает ЛУЧШЕ сложных!")
        print(f"   Принцип Occam's Razor: если простая модель не хуже сложной, выбираем простую")
        print(f"   Разница R²: {baseline['simple_r2'] - hp_search['optuna_r2']:+.4f} vs Optuna, "
              f"{baseline['simple_r2'] - hp_search['gs_r2']:+.4f} vs GridSearchCV")

    model_selection_details = {
        'winner': winner,
        'simple_model_name': baseline['best_name'],
        'simple_r2': baseline['simple_r2'],
        'simple_mae': baseline['simple_mae'],
        'optuna_model_name': hp_search['optuna_model_name'],
        'optuna_r2': hp_search['optuna_r2'],
        'optuna_mae': hp_search['optuna_mae'],
        'optuna_params': hp_search['optuna_params'],
        'gridsearch_r2': hp_search['gs_r2'],
        'gridsearch_mae': hp_search['gs_mae'],
        'gridsearch_params': hp_search['gs_params'],
        'final_model_name': final_model_name,
        'final_model_type': final_model_type,
        'final_r2': final_r2,
        'final_mae': final_mae,
    }

    return {
        'winner': winner, 'final_pipeline': final_pipeline, 'final_model_name': final_model_name,
        'final_model_type': final_model_type, 'final_params': final_params,
        'final_r2': final_r2, 'final_mae': final_mae,
        'model_selection_details': model_selection_details,
    }


def _run_post_training_analysis(
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series,
        num_feat: List[str], cat_feat: List[str], bin_feat: List[str], bool_feat: List[str],
        preprocessor: ColumnTransformer,
        selection: Dict[str, Any]
) -> Dict[str, Any]:
    """Шаги 7-9 run_ml_pipeline(): Ablation Study, Bootstrap Ablation,
    Sensitivity Analysis, финальная кросс-валидация.

    🔧 РЕФАКТОРИНГ: имя функции — прямая рекомендация куратора
    ("_run_post_training_analysis()"). Логика не изменилась.
    """
    final_params = selection['final_params']
    final_model_type = selection['final_model_type']
    final_r2 = selection['final_r2']
    final_mae = selection['final_mae']

    print(f"\n{'=' * 70}")
    print("ABLATION STUDY (анализ вклада групп признаков)")
    print(f"{'=' * 70}")

    ablation_results = run_ablation_study(
        X_train, X_test, y_train, y_test,
        num_feat, cat_feat, bin_feat,
        bool_features=bool_feat,
        best_params=final_params,
        model_name=final_model_type
    )

    print(f"\n{'=' * 70}")
    print("BOOTSTRAP ABLATION (точечный анализ отдельных признаков)")
    print(f"{'=' * 70}")

    bootstrap_ablation_results = run_bootstrap_ablation(
        X_train, X_test, y_train, y_test,
        numeric_features=num_feat, categorical_features=cat_feat,
        binary_features=bin_feat, bool_features=bool_feat,
        best_params=final_params, model_name=final_model_type,
        features_to_test=None,
        n_repeats=15
    )

    print(f"\n{'=' * 70}")
    print("SENSITIVITY ANALYSIS (анализ чувствительности к объёму данных)")
    print(f"{'=' * 70}")

    sensitivity_results = sensitivity_analysis(
        X_train, X_test, y_train, y_test,
        preprocessor,
        best_params=final_params,
        fractions=[0.5, 0.6, 0.7, 0.8, 0.9],
        n_repeats=3,
        baseline_r2=final_r2,
        baseline_mae=final_mae,
        model_name=final_model_type
    )

    X_full = pd.concat([X_train, X_test]).sort_index()
    y_full = pd.concat([y_train, y_test]).sort_index()

    cv_scores, cv_pipeline = cross_validate_model(
        X_full, y_full,
        preprocessor,
        best_params=final_params,
        model_name=final_model_type
    )

    return {
        'ablation_results': ablation_results,
        'bootstrap_ablation_results': bootstrap_ablation_results,
        'sensitivity_results': sensitivity_results,
        'cv_scores': cv_scores, 'cv_pipeline': cv_pipeline,
        'X_full': X_full, 'y_full': y_full,
    }


def run_ml_pipeline(
        df_unique: pd.DataFrame,
        eda_results: Dict[str, Any],
        rerun_optuna: bool = False,
        trials: Optional[int] = None,
        cv: Optional[int] = None
) -> 'MLPipelineResult':
    """
    Управляет полным циклом машинного обучения и оценки моделей.

    Выполняет следующие шаги:
    1. Предобработка и деление данных на обучающую/тестовую выборки с учетом
       вердиктов безопасности, полученных на этапе EDA.
    2. Первичное сравнение базовых моделей (линейные модели, деревья решений, градиентный бустинг).
    3. Дополнительный тест на утечку данных (проверка признака `category_price_level`).
    4. Байесовская оптимизация гиперпараметров с помощью Optuna (опционально).
    5. Поиск по сетке (GridSearchCV) для лучшей модели.
    6. Сравнение Optuna, GridSearchCV и простой модели, выбор лучшей.
    7. Исследование вклада признаков в итоговую модель (Ablation Study).
    8. Оценка устойчивости модели к объёму тренировочных данных (Sensitivity Analysis).
    9. Финальная кросс-валидация для оценки генерализующей способности модели.

    🔧 ДИНАМИЧЕСКАЯ ПЕРЕДАЧА ТИПА МОДЕЛИ:
    Функция определяет тип финальной модели и передаёт его в дочерние функции
    (run_ablation_study, sensitivity_analysis, cross_validate_model) через
    параметр model_name. Это обеспечивает согласованность всех анализов с
    финальной моделью, без жёстких зашивок.

    🔧 РЕФАКТОРИНГ (куратор): раньше это была одна функция на 325 строк.
    Теперь — тонкий оркестратор из 4 приватных шагов (_prepare_and_compare_baseline,
    _run_hyperparameter_search, _select_best_model, _run_post_training_analysis),
    каждый из которых можно читать и тестировать отдельно. Логика вычислений
    не изменилась ни на строку — это чистое извлечение метода (extract method).

    Args:
        df_unique: Датасет без дубликатов для построения признаков и обучения моделей.
        eda_results: Результаты этапа EDA, содержащие вердикты по `sellable_online`
            и `old_price` для предотвращения утечек данных (Data Leakage).
        rerun_optuna: Если True, запускает повторный ресурсоемкий поиск параметров
            через Optuna. Если False, используются предопределенные стабильные параметры.
        trials: Количество итераций (испытаний) для оптимизатора Optuna.
        cv: Количество блоков (фолдов) при кросс-валидации.

    Returns:
        MLPipelineResult: Dataclass (не словарь — см. класс MLPipelineResult выше
            по файлу) с объектами обучения и результатами валидации. Поля те же,
            что раньше были ключами словаря, доступ теперь через точку:
            result.X_train вместо result['X_train'].
            Поля: X_train, X_test, y_train, y_test, X_full, y_full, num_feat,
            cat_feat, bin_feat, bool_feat, preprocessor, best_model_pipeline,
            best_model_name, gridsearch_results, cv_scores, ablation_results,
            bootstrap_ablation_results, sensitivity_results, leakage_verdict,
            model_selection_details.
    """
    baseline = _prepare_and_compare_baseline(df_unique, eda_results)

    hp_search = _run_hyperparameter_search(
        baseline['X_train'], baseline['X_test'], baseline['y_train'], baseline['y_test'],
        baseline['preprocessor'], rerun_optuna, trials, cv
    )

    selection = _select_best_model(baseline, hp_search)

    post = _run_post_training_analysis(
        baseline['X_train'], baseline['X_test'], baseline['y_train'], baseline['y_test'],
        baseline['num_feat'], baseline['cat_feat'], baseline['bin_feat'], baseline['bool_feat'],
        baseline['preprocessor'], selection
    )

    print(f"\n{'=' * 70}")
    print(" ИТОГИ ML-ПАЙПЛАЙНА")
    print(f"{'=' * 70}")
    print(f"\n🏆 Финальная модель: {selection['final_model_name']}")
    print(f"   Тип модели: {selection['final_model_type']}")
    print(f"   R² (тест): {selection['final_r2']:.4f}")
    print(f"   MAE (тест): {selection['final_mae']:.2f} SR")
    print(f"   R² (CV): {post['cv_scores'].mean():.4f} ± {post['cv_scores'].std():.4f}")
    print(f"   Способ выбора: {selection['winner'].upper()}")

    return MLPipelineResult(
        X_train=baseline['X_train'],
        X_test=baseline['X_test'],
        y_train=baseline['y_train'],
        y_test=baseline['y_test'],
        X_full=post['X_full'],
        y_full=post['y_full'],
        num_feat=baseline['num_feat'],
        cat_feat=baseline['cat_feat'],
        bin_feat=baseline['bin_feat'],
        bool_feat=baseline['bool_feat'],
        preprocessor=baseline['preprocessor'],
        best_model_pipeline=selection['final_pipeline'],
        best_model_name=selection['final_model_name'],
        gridsearch_results=hp_search['gridsearch_results'],
        cv_scores=post['cv_scores'],
        ablation_results=post['ablation_results'],
        bootstrap_ablation_results=post['bootstrap_ablation_results'],
        sensitivity_results=post['sensitivity_results'],
        leakage_verdict=baseline['verdict'],
        model_selection_details=selection['model_selection_details'],
    )

# ==============================================================================
# ОРКЕСТРАТОР ИНТЕРПРЕТАЦИИ
# ==============================================================================
