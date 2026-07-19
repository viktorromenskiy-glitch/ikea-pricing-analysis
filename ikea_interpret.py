"""
ikea_interpret.py — часть модульного разбиения Step_project_Q14.
"""
from ikea_core import *


def analyze_feature_importances(
        best_model: Pipeline,
        X_data: pd.DataFrame,
        y_data: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
   Оценивает, группирует и визуализирует важность признаков (Feature Importance)
    обученной модели для технической и бизнес-интерпретации.

    Функция реализует гибкий алгоритм анализа:
    1. Пытается извлечь встроенные важности (`feature_importances_`) из финальной модели.
       Если они недоступны, рассчитывает `permutation_importance` на логарифмированном
       таргете (оценка падения метрики R² при перемешивании признака).
    2. Парсит технические имена колонок после Sklearn-препроцессора, очищая их
       от сервисных префиксов ('num__', 'cat__', 'bin__').
    3. Агрегирует (суммирует) важность дамми-переменных One-Hot кодирования
       обратно в единый родительский признак 'category' для формирования понятного
       бизнес-отчета.
    4. Классифицирует признаки по доменным типам и строит два диагностических
       горизонтальных графика (детальный топ-20 и сгруппированный бизнес-профиль).

    Args:
        best_model (Pipeline): Обученный итоговый Sklearn Pipeline, обязательно
            содержащий шаги с именами 'preprocessor' и 'model'.
        X_data (pd.DataFrame): Матрица признаков, на которой будет производиться
            оценка (например, X_test или вся выборка X).
        y_data (pd.Series): Фактические значения целевой переменной (в оригинальном
            масштабе цен), используемые для валидации перестановок.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            Результаты анализа важности в виде трех структур данных:
            1.  importance_df (pd.DataFrame): Детальная таблица важности каждого отдельного
                признака на выходе препроцессора (включая единичные категории OHE).
                Колонки: ['Feature', 'Importance', 'Importance_%', 'Base_Feature', 'Type'].
            2.  type_importance (pd.Series): Сгруппированная важность по макро-типам
                признаков (индекс — тип, значение — суммарный %).
            3.  grouped_importance_df (pd.DataFrame): Очищенная бизнес-таблица важности,
                где все подкатегории схлопнуты в базовые фичи.
                Колонки: ['Feature', 'Importance_%'].

    Note:
        - Расчет `permutation_importance` запускается в многопоточном режиме (`n_jobs=-1`)
          и производит по 10 повторений перестановок на каждый признак для стабильности оценки.
        - Сгенерированные графики сохраняются локально в форматах PNG и SVG с помощью
          функции `save_plot()`.
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 70)

    # Извлекаем модель из pipeline
    model = best_model.named_steps['model']

    # ========================================================================
    # Попытка 1: feature_importances_ (для XGBoost, RandomForest, etc.)
    # ========================================================================
    if hasattr(model, 'feature_importances_'):
        print("\n📊 Используем feature_importances_ (встроенный метод модели)")
        importances = model.feature_importances_

        # Получаем имена признаков после препроцессинга
        prep = best_model.named_steps['preprocessor']
        try:
            feature_names = prep.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

    # ========================================================================
    # Попытка 2: permutation importance (универсальный метод)
    # ========================================================================
    else:
        print("\n📊 Используем permutation importance (универсальный метод)")

        print("   Вычисление (может занять 1-2 минуты)...")

        # Модель в pipeline предсказывает log(price), поэтому y тоже должен быть в log
        y_data_log = np.log1p(y_data)

        result = permutation_importance(
            best_model, X_data, y_data_log,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring='r2'
        )
        importances = result.importances_mean

        # используем имена колонок X (без префиксов)
        feature_names = X_data.columns.tolist()

    # ========================================================================
    # Создаём DataFrame с важностями
    # ========================================================================
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # Нормализуем в проценты (только для положительных значений)
    positive_importances = importance_df[importance_df['Importance'] > 0]['Importance']
    total = positive_importances.sum() if len(positive_importances) > 0 else 1
    importance_df['Importance_%'] = importance_df['Importance'].apply(
        lambda x: max(0, x) / total * 100 if total > 0 else 0
    )

    # ========================================================================
    # 🆕 ГРУППИРОВКА ПРИЗНАКОВ (для бизнес-отчёта)
    # ========================================================================
    print("\n🔍 Группировка признаков...")

    def get_base_name(fname: str) -> str:
        """Определяет базовое имя признака для группировки."""
        fname = str(fname)

        # One-Hot Encoding: cat__category_Wardrobes -> category
        if fname.startswith('cat__category_'):
            return 'category'

        # Числовые признаки: num__depth -> depth
        elif fname.startswith('num__'):
            return fname[5:]  # убираем 'num__'

        # Бинарные признаки: bin__is_team -> is_team
        elif fname.startswith('bin__'):
            return fname[5:]  # убираем 'bin__'

        # Другие категориальные (если есть)
        elif fname.startswith('cat__'):
            parts = fname[5:].split('_')
            return parts[0] if len(parts) > 1 else fname[5:]

        # Без префикса (permutation importance)
        return fname

    importance_df['Base_Feature'] = importance_df['Feature'].apply(get_base_name)

    # Группируем по базовому имени и суммируем важность
    grouped_importance_df = importance_df.groupby('Base_Feature')['Importance_%'].sum().reset_index()
    grouped_importance_df.columns = ['Feature', 'Importance_%']

    # Перенормализуем (сумма должна быть 100%)
    total_grouped = grouped_importance_df['Importance_%'].sum()
    if total_grouped > 0:
        grouped_importance_df['Importance_%'] = (
                grouped_importance_df['Importance_%'] / total_grouped * 100
        )

    grouped_importance_df = grouped_importance_df.sort_values('Importance_%', ascending=False).reset_index(drop=True)

    # ========================================================================
    # Классификация по типам (ОБНОВЛЁННЫЕ СПИСКИ)
    # ========================================================================
    # 📌 Числовые признаки (после удаления мультиколлинеарных)
    # 🔧 Синхронизировано с prepare_ml_data() после исключения
    # desc_quality_score, complexity_x_premium, discount_pct, has_discount
    # (см. комментарии в prepare_ml_data — Bootstrap Ablation + структурная утечка)
    numeric_names = [
        'depth', 'height', 'width', 'volume',
        'desc_length', 'desc_word_count',
        'premium_materials_count', 'designer_freq',
        'category_price_level',
        'assembly_complexity',
    ]

    # 📌 Бинарные признаки (после удаления мультиколлинеарных)
    binary_names = [
        'is_team', 'has_other_colors', 'is_composite',
        'has_old_price',
        'has_depth', 'has_height', 'has_width',
        'is_premium_category', 'is_large_item',
    ]

    categorical_names = ['category']
    bool_names = ['sellable_online']

    def get_type(fname: str) -> str:
        fname = str(fname)
        # Проверяем префиксы (для feature_importances_)
        if fname.startswith('num__'):
            return 'Числовые (габариты + NLP)'
        elif fname.startswith('cat__'):
            return 'Категория товара'
        elif fname.startswith('bin__'):
            return 'Бинарные признаки'
        # Проверяем точное соответствие (для permutation importance)
        elif fname in numeric_names:
            return 'Числовые (габариты + NLP)'
        elif fname in categorical_names:
            return 'Категория товара'
        elif fname in binary_names or fname in bool_names:
            return 'Бинарные признаки'
        return 'Другие'

    importance_df['Type'] = importance_df['Feature'].apply(get_type)
    type_importance = importance_df.groupby('Type')['Importance_%'].sum().sort_values(ascending=False)

    # ========================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 ДЕТАЛЬНАЯ ВАЖНОСТЬ (для Data Science)")
    print("=" * 70)
    print(f"\nТоп-20 признаков:")
    print(importance_df.head(20).to_markdown(index=False, floatfmt=".4f"))

    print(f"\nВажность по типам:")
    for t, imp in type_importance.items():
        print(f"  • {t}: {imp:.1f}%")

    print("\n" + "=" * 70)
    print("📊 СГРУППИРОВАННАЯ ВАЖНОСТЬ (для бизнеса/менеджмента)")
    print("=" * 70)
    print(f"\nВажность по базовым признакам:")
    print(grouped_importance_df.to_markdown(index=False, floatfmt=".2f"))

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ 1: Топ-20 детальных признаков
    # ========================================================================
    print("\n📈 Создание графиков...")

    fig1, ax1 = plt.subplots(figsize=(12, 10))
    top_20 = importance_df.head(20).iloc[::-1]
    ax1.barh(top_20['Feature'], top_20['Importance_%'], color='steelblue')
    ax1.set_xlabel('Важность (%)')
    ax1.set_title('Топ-20 признаков по важности (детально)', fontweight='bold')
    plt.tight_layout()
    save_plot(fig1, 'ml_feature_importances_detailed')
    plt.close()
    print("  ✓ Сохранено: ml_feature_importances_detailed.png и .svg")

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ 2: Сгруппированные признаки (для бизнеса)
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    grouped_sorted = grouped_importance_df.iloc[::-1]  # сортируем для горизонтального barh

    # Цвета по типам
    colors = []
    for feature in grouped_sorted['Feature']:
        if feature in categorical_names:
            colors.append('gold')
        elif feature in numeric_names:
            colors.append('steelblue')
        elif feature in binary_names or feature in bool_names:
            colors.append('lightgreen')
        else:
            colors.append('gray')

    bars = ax2.barh(grouped_sorted['Feature'], grouped_sorted['Importance_%'], color=colors, edgecolor='black')
    ax2.set_xlabel('Важность (%)')
    ax2.set_title('Важность признаков (сгруппировано для бизнеса)', fontweight='bold')

    # Добавляем значения на столбцах
    for bar, val in zip(bars, grouped_sorted['Importance_%']):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_plot(fig2, 'ml_feature_importances_grouped')
    plt.close()
    print("  ✓ Сохранено: ml_feature_importances_grouped.png и .svg")

    # ========================================================================
    # ИТОГОВЫЕ ВЫВОДЫ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ИТОГОВЫЕ ВЫВОДЫ:")
    print(f"{'=' * 70}")

    top_feature = grouped_importance_df.iloc[0]['Feature']
    top_importance = grouped_importance_df.iloc[0]['Importance_%']
    print(f"\n🏆 Самый важный признак: {top_feature} ({top_importance:.1f}%)")

    print(f"\nТоп-5 признаков:")
    for i, row in grouped_importance_df.head(5).iterrows():
        print(f"  {i + 1}. {row['Feature']:30s} {row['Importance_%']:5.1f}%")

    # 🆕 ВАЖНО: возвращаем 3 значения вместо 2!
    return importance_df, type_importance, grouped_importance_df

# =============================================================================
# ЧАСТЬ 19: OPTUNA — БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ
# ============================================================================

def analyze_model_errors(
        best_model_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет детальный анализ и визуализацию ошибок финальной модели в разрезе
    ценовых категорий и диапазонов.

    Функция автоматизирует следующие аналитические шаги:
    1.  Получает предсказания модели и переводит их из логарифмического масштаба
        `log(price + 1)` обратно в оригинальный масштаб цен в саудовских риалах (SR)
        с помощью экспоненты `np.expm1`.
    2.  Вычисляет абсолютные ошибки (Absolute Errors) и относительные процентные ошибки
        (Percentage Errors) с использованием маски для защиты от деления на ноль.
    3.  Выводит сводные статистические метрики: MAE, Median AE, среднюю/медианную
        относительную ошибку и MAPE.
    4.  Агрегирует метрики по категориям мебели и по 5 фиксированным ценовым диапазонам
        (0–300, 300–600, 600–1000, 1000–2000, 2000+ SR).
    5.  Находит топ-5 наблюдений с наибольшей абсолютной ошибкой для точечного анализа аномалий.
    6.  Строит два диагностических графика: диаграмму рассеяния "Факт vs Предсказание"
        и гистограмму распределения абсолютных ошибок.

    Args:
        best_model_pipeline (Pipeline): Обученный финальный пайплайн, на последнем шаге
            которого находится регрессионная модель, предсказывающая логарифм цены.
        X_test (pd.DataFrame): Тестовая матрица признаков. Должна содержать исходную
            колонку 'category' для группировки ошибок.
        y_test (pd.Series): Истинные значения целевой переменной (цены в SR) на тесте.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Векторы ошибок для дальнейшего стат-анализа:
            1.  abs_errors (np.ndarray): Вектор абсолютных ошибок $|y_{true} - y_{pred}|$
                для каждого наблюдения.
            2.  errors_pct (np.ndarray): Вектор относительных ошибок в процентах
                $|y_{true} - y_{pred}| / y_{true} \times 100$ (для нулевых фактических цен
                записано значение 0.0).

    Raises:
        ValueError: Если X_test не содержит колонку 'category'.
        Exception: При ошибке предсказания модели или построения графиков.

    Note:
        - Использование MAPE (Mean Absolute Percentage Error) крайне важно для мебельного
          ритейла, так как ошибка в 100 SR на дешевом стуле критична, а на дорогом
          шкафу — практически незаметна.
        - Изображения графиков сохраняются локально в директорию проекта под единым
          именем 'ml_model_errors' в векторном (.svg) и растровом (.png) форматах.
    """

    try:
        # Предсказания (модель обучена на log, поэтому обратно преобразуем)
        y_test_log = np.log1p(y_test)
        y_pred_log = best_model_pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        # Абсолютные ошибки
        abs_errors = np.abs(y_test.values - y_pred)

        # Относительные ошибки (%) — с защитой от деления на 0
        mask_nonzero = y_test.values != 0
        errors_pct = np.zeros_like(abs_errors)
        errors_pct[mask_nonzero] = np.abs(
            (y_test.values[mask_nonzero] - y_pred[mask_nonzero]) / y_test.values[mask_nonzero]) * 100

        # ========================================================================
        # СТАТИСТИКА ОШИБОК
        # ========================================================================
        print(f"\n Статистика ошибок:")
        print(f"  • MAE: {abs_errors.mean():.2f} SR")
        print(f"  • Median AE: {np.median(abs_errors):.2f} SR")
        print(f"  • Mean % error: {errors_pct.mean():.2f}%")
        print(f"  • Median % error: {np.median(errors_pct):.2f}%")
        print(f"  • MAPE: {errors_pct[mask_nonzero].mean():.2f}% (только для ненулевых цен)")

        # ========================================================================
        # ТОП-5 ХУДШИХ ПРЕДСКАЗАНИЙ
        # ========================================================================
        print(f"\n🔴 Топ-5 худших предсказаний:")
        worst_indices = np.argsort(abs_errors)[-5:][::-1]
        for idx in worst_indices:
            true_val = y_test.values[idx]
            pred_val = y_pred[idx]
            err = abs_errors[idx]
            pct = errors_pct[idx]
            print(f"  • Факт: {true_val:.0f} SR → Предсказано: {pred_val:.0f} SR "
                  f"(ошибка: {err:.0f} SR, {pct:.1f}%)")

        # ========================================================================
        # ОШИБКИ ПО КАТЕГОРИЯМ (с MAPE)
        # ========================================================================
        print(f"\n Ошибки по категориям:")

        # Проверяем наличие колонки 'category'
        if 'category' not in X_test.columns:
            raise ValueError("X_test не содержит колонку 'category' для группировки ошибок")

        # Создаём DataFrame для группировки
        errors_df = pd.DataFrame({
            'category': X_test['category'].values,
            'y_true': y_test.values,
            'y_pred': y_pred,
            'abs_error': abs_errors,
            'pct_error': errors_pct
        })

        # Группировка по категориям
        errors_by_category = errors_df.groupby('category').agg(
            mae=('abs_error', 'mean'),
            pct=('pct_error', 'mean'),  # это MAPE в процентах
            count=('abs_error', 'size')
        ).sort_values('mae', ascending=False)

        for cat in errors_by_category.index:
            mae = errors_by_category.loc[cat, 'mae']
            mape = errors_by_category.loc[cat, 'pct']
            count = errors_by_category.loc[cat, 'count']
            print(f"  • {cat:40s} | MAE: {mae:6.0f} | MAPE: {mape:5.1f}% | Кол-во: {count}")

        # ========================================================================
        # ОШИБКИ ПО ЦЕНОВЫМ ДИАПАЗОНАМ (с MAPE)
        # ========================================================================
        print(f"\n Ошибки по ценовым диапазонам:")

        # Создаём ценовые диапазоны
        bins = [0, 300, 600, 1000, 2000, float('inf')]
        labels = ['0-300', '300-600', '600-1000', '1000-2000', '2000+']
        errors_df['price_range'] = pd.cut(errors_df['y_true'], bins=bins, labels=labels, right=True)

        errors_by_price_range = errors_df.groupby('price_range', observed=False).agg(
            mae=('abs_error', 'mean'),
            pct=('pct_error', 'mean'),
            count=('abs_error', 'size')
        )

        for range_name in errors_by_price_range.index:
            mae = errors_by_price_range.loc[range_name, 'mae']
            mape = errors_by_price_range.loc[range_name, 'pct']
            count = errors_by_price_range.loc[range_name, 'count']
            if count > 0:
                print(f"  • {range_name:15s} | MAE: {mae:6.0f} | MAPE: {mape:5.1f}% | Кол-во: {count}")

        # ========================================================================
        # ВИЗУАЛИЗАЦИЯ
        # ========================================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Факт vs Предсказание
        ax1 = axes[0]
        ax1.scatter(y_test, y_pred, alpha=0.5, s=20, c='steelblue')
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеальное предсказание')
        ax1.set_xlabel('Фактическая цена (SR)')
        ax1.set_ylabel('Предсказанная цена (SR)')
        ax1.set_title('Факт vs Предсказание', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Распределение абсолютных ошибок
        ax2 = axes[1]
        ax2.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Средняя: {abs_errors.mean():.0f} SR')
        ax2.axvline(np.median(abs_errors), color='green', linestyle='--', linewidth=2,
                    label=f'Медиана: {np.median(abs_errors):.0f} SR')
        ax2.set_xlabel('Абсолютная ошибка (SR)')
        ax2.set_ylabel('Количество товаров')
        ax2.set_title('Распределение ошибок', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(fig, 'ml_model_errors')
        plt.close()

        print(f"\n  ✓ Сохранено: ml_model_errors.png и ml_model_errors.svg")

        return abs_errors, errors_pct

    except Exception as e:
        print(f"⚠️ Критическая ошибка в analyze_model_errors: {e}")
        raise

# =============================================================================
#  22. Проверка утечки
# =============================================================================

def bootstrap_mae_confidence_interval(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
) -> Tuple[float, float, float, np.ndarray]:
    """
  Рассчитывает доверительный интервал для средней абсолютной ошибки (MAE)
    методом непараметрического бутстрапа (Bootstrap Percentile Method).

    Метод генерирует `n_bootstrap` случайных выборок с возвращением из тестового
    набора данных, вычисляет MAE для каждой выборки и определяет границы доверительного
    интервала на основе заданных перцентилей распределения. Это позволяет оценить
    стабильность предсказаний модели без допущений о нормальном распределении ошибок.

    Args:
        model (Pipeline): Обученный пайплайн Sklearn (или совместимая модель),
            предсказывающая целевую переменную в логарифмическом масштабе `log(price + 1)`.
        X_test (pd.DataFrame): Тестовая матрица признаков для генерации предсказаний.
        y_test (pd.Series): Истинные цены в оригинальном (не логарифмированном) масштабе.
        n_bootstrap (int, optional): Количество генерируемых бутстрап-выборок.
            Чем выше значение, тем точнее оценка границ, но тем дольше расчеты. Defaults to 1000.
        confidence (float, optional): Уровень доверия для интервальной оценки (значение
            в интервале от 0 до 1). Defaults to 0.95.

    Returns:
        Tuple[float, float, float, np.ndarray]:
            Результаты бутстрап-анализа:
            1.  mae_original (float): Исходное значение MAE модели на всей тестовой выборке.
            2.  mae_ci_lower (float): Нижняя граница доверительного интервала (соответствует
                перцентилю $\alpha/2$, где $\alpha = 1 - \text{confidence}$).
            3.  mae_ci_upper (float): Верхняя граница доверительного интервала (соответствует
                перцентилю $1 - \alpha/2$).
            4.  mae_bootstraps (np.ndarray): Одномерный массив размером `(n_bootstrap,)`,
                содержащий значения MAE для каждой сгенерированной бутстрап-выборки.

    Note:
        - Перед расчетом метрик функция преобразует предсказания модели из логарифмического
          масштаба в исходный с помощью `np.expm1(y_pred_log)`.
        - Для воспроизводимости результатов бутстрапа жестко зафиксирован генератор
          случайных чисел (`np.random.seed(42)`).
        - Функция сохраняет графики распределения MAE (гистограмму плотности распределения
          и диаграмму "ящик с усами") под именем 'ml_mae_confidence_interval' в растровом (.png)
          и векторном (.svg) форматах.
    """

    # Предсказания модели
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # обратное преобразование log → price

    # Исходный MAE
    mae_original = np.mean(np.abs(y_test.values - y_pred))
    print(f"\n Исходный MAE: {mae_original:.2f} SR")

    # Bootstrap
    print(f"\n Bootstrap ({n_bootstrap} итераций)...")
    np.random.seed(42)
    mae_bootstraps = []

    n_samples = len(y_test)

    for i in range(n_bootstrap):
        # Сэмплируем с заменой
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Считаем MAE на бутстрап-выборке
        y_test_boot = y_test.values[indices]
        y_pred_boot = y_pred[indices]
        mae_boot = np.mean(np.abs(y_test_boot - y_pred_boot))
        mae_bootstraps.append(mae_boot)

    mae_bootstraps = np.array(mae_bootstraps)

    # Доверительный интервал
    alpha = (1 - confidence) / 2
    mae_ci_lower = np.percentile(mae_bootstraps, alpha * 100)
    mae_ci_upper = np.percentile(mae_bootstraps, (1 - alpha) * 100)

    print(f"\n РЕЗУЛЬТАТЫ:")
    print(f"  • MAE: {mae_original:.2f} SR")
    print(f"  • 95% CI: [{mae_ci_lower:.2f}, {mae_ci_upper:.2f}]")
    print(f"  • Ширина CI: {mae_ci_upper - mae_ci_lower:.2f} SR")
    print(f"  • Относительная ширина: {(mae_ci_upper - mae_ci_lower) / mae_original * 100:.1f}%")

    # Статистика бутстрапа
    print(f"\n Статистика бутстрап-распределения:")
    print(f"  • Среднее MAE: {mae_bootstraps.mean():.2f} SR")
    print(f"  • Std MAE: {mae_bootstraps.std():.2f} SR")
    print(f"  • Минимум: {mae_bootstraps.min():.2f} SR")
    print(f"  • Максимум: {mae_bootstraps.max():.2f} SR")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Гистограмма бутстрап-распределения
    axes[0].hist(mae_bootstraps, bins=50, color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=0.5)
    axes[0].axvline(mae_original, color='red', linestyle='--', linewidth=2,
                    label=f'Исходный MAE: {mae_original:.1f}')
    axes[0].axvline(mae_ci_lower, color='green', linestyle=':', linewidth=2,
                    label=f'95% CI: [{mae_ci_lower:.1f}, {mae_ci_upper:.1f}]')
    axes[0].axvline(mae_ci_upper, color='green', linestyle=':', linewidth=2)
    axes[0].set_xlabel('MAE (SR)')
    axes[0].set_ylabel('Частота')
    axes[0].set_title('Бутстрап-распределение MAE', fontweight='bold')
    axes[0].legend()

    # График 2: Box plot

    axes[1].boxplot(mae_bootstraps, orientation='vertical')
    axes[1].axhline(mae_original, color='red', linestyle='--', linewidth=2,
                    label=f'Исходный MAE: {mae_original:.1f}')
    axes[1].set_ylabel('MAE (SR)')
    axes[1].set_title('Box plot бутстрап-распределения', fontweight='bold')
    axes[1].set_xticklabels(['MAE'])
    axes[1].legend()

    plt.tight_layout()
    save_plot(fig, 'ml_mae_confidence_interval')
    plt.close()
    print(f"\n  ✓ Сохранено: ml_mae_confidence_interval.png и ml_mae_confidence_interval.svg")

    return mae_original, mae_ci_lower, mae_ci_upper, mae_bootstraps

# =============================================================================
# 25. Сравнение с  baseline'ами
# =============================================================================

def compare_with_baselines(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        best_model_pipeline: Pipeline,
        our_model_name: str = 'HistGB + Optuna'
) -> pd.DataFrame:
    """
   Сравнивает качество работы финальной ML-модели с серией классических бейслайнов.

    Функция последовательно обучает и оценивает 6 подходов (от наивных до сложных):
    1.  **Zero Rule**: Константное предсказание медианы цен обучающей выборки.
    2.  **Category Median**: Назначение товару медианной цены его категории из `y_train`.
    3.  **Linear Regression**: Линейная регрессия на базовых габаритных признаках
        (`width`, `height`, `depth`, `volume`) без использования текстовых или категориальных фичей.
    4.  **Random Forest**: Случайный лес со стандартными параметрами, обученный на признаках,
        обработанных через `ColumnTransformer` из финального пайплайна.
    5.  **HistGradientBoosting (базовый)**: Градиентный бустинг со стандартными настройками
        библиотеки Scikit-Learn (без тюнинга).
    6.  **Финальная модель**: Твоя лучшая оптимизированная модель (например, `HistGB + Optuna`).

    Args:
        X_train (pd.DataFrame): Обучающий набор признаков. Используется для извлечения
            медиан категорий и обучения бейслайнов.
        X_test (pd.DataFrame): Тестовый набор признаков для валидации всех моделей.
        y_train (pd.Series): Истинные цены (в оригинальном масштабе) для обучения.
        y_test (pd.Series): Истинные цены (в оригинальном масштабе) для оценки качества.
        best_model_pipeline (Pipeline): Обученный финальный пайплайн, содержащий
            `preprocessor` на первом шаге и оптимизированную модель на последнем.
        our_model_name (str, optional): Кастомное имя твоей лучшей модели для отображения
            на графиках и в финальной таблице. Defaults to 'HistGB + Optuna'.

    Returns:
        pd.DataFrame: Сводная таблица результатов сравнения, отсортированная по возрастанию
            метрики MAE. Содержит колонки:
            - `Model` (str): Название модели/бейслайна.
            - `MAE` (float): Средняя абсолютная ошибка в саудовских риалах (SR).
            - `R²` (float): Коэффициент детерминации.
            - `Type` (str): Категория подхода (Наивный, Простой, Наша модель и т.д.).
            - `Улучшение vs Zero Rule` (str): Процент снижения MAE относительно константного
              прогноза.

    Note:
        - Для борьбы с мультиколлинеарностью из списка базовых признаков линейной регрессии
          исключена колонка площади (`area`), так как она является прямой производной от
          линейных размеров.
        - Логика работы с габаритами включает автоматическую импутацию пропусков в `volume`
          и других размерах медианными значениями, рассчитанными строго на `X_train`.
        - Модели 3, 4, 5 обучаются на логарифмированном таргете `np.log1p(y)`, после чего
          предсказания переводятся обратно через `np.expm1` для расчета метрик.
        - Функция сохраняет сопоставительные графики MAE и R² в форматах .png и .svg
          с именем 'ml_baseline_comparison'.
    """

    results = []

    # Логарифмируем y_train для обучения моделей
    y_train_log = np.log1p(y_train)

    # ========================================================================
    # 1. Zero Rule (наивный baseline) — медиана цены
    # ========================================================================
    print("\n Baseline 1: Zero Rule (медиана цены)")
    median_price = y_train.median()
    y_pred_zero = np.full(len(y_test), median_price)
    mae_zero = mean_absolute_error(y_test, y_pred_zero)
    r2_zero = r2_score(y_test, y_pred_zero)
    print(f"  • Предсказание: {median_price:.0f} SR для всех товаров")
    print(f"  • MAE: {mae_zero:.2f} SR")
    print(f"  • R²: {r2_zero:.4f}")
    results.append({
        'Model': 'Zero Rule (медиана)',
        'MAE': mae_zero,
        'R²': r2_zero,
        'Type': 'Наивный'
    })

    # ========================================================================
    # 2. Category Median — медиана по категории
    # ========================================================================
    print("\n Baseline 2: Медиана по категории")
    category_medians = y_train.groupby(X_train['category']).median()
    y_pred_cat = X_test['category'].map(category_medians)
    # Если категория не нашлась — используем общую медиану
    y_pred_cat = y_pred_cat.fillna(median_price).values
    mae_cat = mean_absolute_error(y_test, y_pred_cat)
    r2_cat = r2_score(y_test, y_pred_cat)
    print(f"  • MAE: {mae_cat:.2f} SR")
    print(f"  • R²: {r2_cat:.4f}")
    results.append({
        'Model': 'Медиана по категории',
        'MAE': mae_cat,
        'R²': r2_cat,
        'Type': 'Простой'
    })

    # ========================================================================
    # 3. Linear Regression — только габариты (БЕЗ area!)
    # ========================================================================
    print("\n Baseline 3: Linear Regression (только габариты)")

    # 🆕 ИСПРАВЛЕНИЕ: удалили 'area' (мультиколлинеарность)
    basic_features = ['width', 'height', 'depth', 'volume']

    X_train_basic = X_train[basic_features].copy()
    X_test_basic = X_test[basic_features].copy()

    # Заполняем пропуски медианой (на train)
    for col in basic_features:
        median_val = X_train_basic[col].median()
        X_train_basic[col] = X_train_basic[col].fillna(median_val)
        X_test_basic[col] = X_test_basic[col].fillna(median_val)

    lr = LinearRegression()
    lr.fit(X_train_basic, y_train_log)
    y_pred_lr_log = lr.predict(X_test_basic)
    y_pred_lr = np.expm1(y_pred_lr_log)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"  • MAE: {mae_lr:.2f} SR")
    print(f"  • R²: {r2_lr:.4f}")
    results.append({
        'Model': 'Linear Regression (габариты)',
        'MAE': mae_lr,
        'R²': r2_lr,
        'Type': 'Простой ML'
    })

    # ========================================================================
    # 4. Random Forest — используем preprocessor из pipeline
    # ========================================================================
    print("\n Baseline 4: Random Forest (без настройки)")
    # ИСПРАВЛЕНИЕ: используем preprocessor для преобразования данных
    preprocessor = best_model_pipeline.named_steps['preprocessor']
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_processed, y_train_log)
    y_pred_rf_log = rf.predict(X_test_processed)
    y_pred_rf = np.expm1(y_pred_rf_log)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"  • MAE: {mae_rf:.2f} SR")
    print(f"  • R²: {r2_rf:.4f}")
    results.append({
        'Model': 'Random Forest (без настройки)',
        'MAE': mae_rf,
        'R²': r2_rf,
        'Type': 'Стандартный ML'
    })

    # ========================================================================
    # 5. HistGradientBoosting (базовая версия, без Optuna)
    # ========================================================================
    print("\n Baseline 5: HistGradientBoosting (без Optuna)")

    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb.fit(X_train_processed, y_train_log)
    y_pred_hgb_log = hgb.predict(X_test_processed)
    y_pred_hgb = np.expm1(y_pred_hgb_log)
    mae_hgb = mean_absolute_error(y_test, y_pred_hgb)
    r2_hgb = r2_score(y_test, y_pred_hgb)
    print(f"  • MAE: {mae_hgb:.2f} SR")
    print(f"  • R²: {r2_hgb:.4f}")
    results.append({
        'Model': 'HistGB (без Optuna)',
        'MAE': mae_hgb,
        'R²': r2_hgb,
        'Type': 'Наша модель (базовая)'
    })

    # ========================================================================
    # 6. Наша финальная модель
    # ========================================================================
    print(f"\n Наша финальная модель: {our_model_name}")
    y_pred_final_log = best_model_pipeline.predict(X_test)
    y_pred_final = np.expm1(y_pred_final_log)
    mae_final = mean_absolute_error(y_test, y_pred_final)
    r2_final = r2_score(y_test, y_pred_final)

    # ИСПРАВЛЕНО: используем mae_final и r2_final вместо несуществующих our_mae/our_r2
    print(f"  • MAE: {mae_final:.2f} SR")
    print(f"  • R²: {r2_final:.4f}")

    # ИСПРАВЛЕНО: динамическое название вместо хардкода
    results.append({
        'Model': f'{our_model_name} (наша)',
        'MAE': mae_final,
        'R²': r2_final,
        'Type': 'Наша модель (финальная)'
    })

    # ========================================================================
    # Сводная таблица
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" СВОДНАЯ ТАБЛИЦА:")
    print(f"{'=' * 70}")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('MAE', ascending=True).reset_index(drop=True)

    # Добавляем колонку "Улучшение vs Zero Rule"
    mae_zero = df_results[df_results['Model'] == 'Zero Rule (медиана)']['MAE'].values[0]
    df_results['Улучшение vs Zero Rule'] = df_results['MAE'].apply(
        lambda x: f"+{(mae_zero - x) / mae_zero * 100:.1f}%"
    )
    print(df_results.to_markdown(index=False, floatfmt=".2f"))

    # ========================================================================
    # Визуализация
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['red', 'orange', 'gold', 'lightgreen', 'lightblue', 'darkgreen']

    # График 1: MAE по моделям
    bars = axes[0].barh(df_results['Model'], df_results['MAE'], color=colors)
    axes[0].set_xlabel('MAE (SR)')
    axes[0].set_title('Сравнение MAE: наша модель vs baseline', fontweight='bold')
    axes[0].invert_yaxis()
    for bar, mae in zip(bars, df_results['MAE']):
        axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                     f'{mae:.0f}', va='center', fontsize=9)

    # График 2: R² по моделям
    bars2 = axes[1].barh(df_results['Model'], df_results['R²'], color=colors)
    axes[1].set_xlabel('R²')
    axes[1].set_title('Сравнение R²: наша модель vs baseline', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='black', linewidth=0.5)
    for bar, r2 in zip(bars2, df_results['R²']):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{r2:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, 'ml_baseline_comparison')
    plt.close()
    print(f"\n  ✓ Сохранено: ml_baseline_comparison.png и ml_baseline_comparison.svg")

    # ========================================================================
    # Выводы
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВЫВОДЫ:")
    print(f"{'=' * 70}")
    improvement_vs_zero = (mae_zero - mae_final) / mae_zero * 100
    improvement_vs_cat = (mae_cat - mae_final) / mae_cat * 100
    improvement_vs_lr = (mae_lr - mae_final) / mae_lr * 100
    improvement_vs_rf = (mae_rf - mae_final) / mae_rf * 100

    print(f"\n Наша модель (MAE = {mae_final:.0f} SR) улучшает:")
    print(f"  • vs Zero Rule:              на {improvement_vs_zero:.1f}%")
    print(f"  • vs Медиана категории:      на {improvement_vs_cat:.1f}%")
    print(f"  • vs Linear Regression:      на {improvement_vs_lr:.1f}%")
    print(f"  • vs Random Forest:          на {improvement_vs_rf:.1f}%")

    return df_results

# =============================================================================
# 26. SHAP-анализ для интерпретации модел
# =============================================================================

def analyze_with_shap(
        best_model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> Optional[pd.DataFrame]:
    """
    Проводит комплексный SHAP-анализ (SHapley Additive exPlanations) для интерпретации
    предсказаний финальной модели.

    Функция позволяет заглянуть внутрь "черного ящика" модели машинного обучения и решает
    следующие аналитические задачи:
    1.  Изолирует шаг предобработки (`preprocessor`) и финальный алгоритм (`model`) из пайплайна,
        трансформируя тестовые данные и извлекая итоговые имена признаков (включая One-Hot кодирование).
    2.  Использует оптимизированный `shap.TreeExplainer` для быстрого расчета вкладов Шепли на тесте.
    3.  Генерирует и сохраняет глобальные графики важности признаков:
        - `shap_summary`: Summary Plot (показывает не только силу, но и направление влияния признаков).
        - `shap_importance`: Bar Plot (среднее абсолютное влияние признаков).
    4.  Идентифицирует наилучшее и наихудшее предсказания модели на тестовой выборке по метрике
        абсолютной ошибки и строит для них локальные графики объяснения (`Waterfall Plot`),
        визуализирующие, какие именно фичи приблизили или отдалили предсказание от базового значения (`expected_value`).
    5.  Формирует детальный отчет по глобальной важности признаков в процентах, агрегируя их
        по типам (числовые, категориальные, бинарные).

    Args:
        best_model (Pipeline): Обученный финальный пайплайн Scikit-Learn, на последнем шаге
            которого находится древесный регрессор (например, `HistGradientBoostingRegressor`),
            а на первом — препроцессор `ColumnTransformer`.
        X_test (pd.DataFrame): Тестовая матрица признаков в исходном (сыром) виде.
        y_test (pd.Series): Истинные значения цен на тестовой выборке в оригинальном масштабе (SR).

    Returns:
        Optional[pd.DataFrame]:
            - `pd.DataFrame`: Таблица глобальной важности признаков (если библиотека `shap`
              успешно импортирована), отсортированная по убыванию абсолютного влияния. Содержит
              колонки `Feature`, `SHAP_Importance`, `Importance_%` и `Type`.
            - `None`: Если библиотека `shap` отсутствует в окружении.

    Note:
        - Функция устойчива к ошибкам отсутствия библиотеки `shap` и не ломает выполнение
          основного скрипта, выводя вместо этого понятную инструкцию по установке.
        - Локальный анализ Waterfall строится на оригинальных (трансформированных препроцессором)
          признаках конкретных объектов. Для этого создаются специализированные объекты `shap.Explanation`.
        - Изображения всех четырех графиков сохраняются локально в форматах .png и .svg.
    """

    try:
        import shap
        print(" shap установлен")
    except ImportError:
        print("❌ Установите: pip install shap")
        return None

    model = best_model.named_steps['model']
    preprocessor = best_model.named_steps['preprocessor']

    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

    print(f"\n Количество признаков после препроцессинга: {len(feature_names)}")

    print("\n Создание SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print(" Вычисление SHAP values...")
    shap_values = explainer.shap_values(X_test_processed)

    # Получаем base_value как скаляр
    base_value = explainer.expected_value
    if hasattr(base_value, '__len__'):
        base_value = float(np.asarray(base_value).flatten()[0])
    else:
        base_value = float(base_value)

    # ========================================================================
    # 1. SUMMARY PLOT
    # ========================================================================
    print("\n Создание summary plot...")

    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title('SHAP Summary Plot: важность признаков и направление влияния',
              fontweight='bold', fontsize=12)
    plt.tight_layout()
    save_plot(fig, 'shap_summary')
    plt.close()
    print("  ✓ Сохранено: shap_summary.png и shap_summary.svg")

    # ========================================================================
    # 2. BAR PLOT
    # ========================================================================
    print("\n Создание bar plot...")

    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=20)
    plt.title('SHAP: средняя важность признаков (|SHAP value|)',
              fontweight='bold', fontsize=12)
    plt.tight_layout()
    save_plot(fig, 'shap_importance')
    plt.close()
    print("  ✓ Сохранено: shap_importance.png и shap_importance.svg")

    # ========================================================================
    # 3. WATERFALL PLOT — худшее предсказание
    # ========================================================================
    print("\n Создание waterfall plot для худшего предсказания...")

    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    errors = np.abs(y_test.values - y_pred)
    worst_idx = errors.argmax()

    print(f"\n🔴 Худшее предсказание:")
    print(f"   Фактическая цена: {y_test.iloc[worst_idx]:.0f} SR")
    print(f"   Предсказанная цена: {y_pred[worst_idx]:.0f} SR")
    print(f"   Ошибка: {errors[worst_idx]:.0f} SR ({errors[worst_idx] / y_test.iloc[worst_idx] * 100:.1f}%)")

    fig = plt.figure(figsize=(12, 8))

    explanation = shap.Explanation(
        values=shap_values[worst_idx],
        base_values=base_value,
        feature_names=feature_names,
        data=X_test_processed[worst_idx]
    )

    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(f'Объяснение худшего предсказания\n'
              f'Факт: {y_test.iloc[worst_idx]:.0f} SR → Предсказано: {y_pred[worst_idx]:.0f} SR',
              fontweight='bold', fontsize=11)
    plt.tight_layout()
    save_plot(fig, 'shap_waterfall_worst')
    plt.close()
    print("  ✓ Сохранено: shap_waterfall_worst.png и shap_waterfall_worst.svg")

    # ========================================================================
    # 4. WATERFALL PLOT — лучшее предсказание
    # ========================================================================
    print("\n Создание waterfall plot для лучшего предсказания...")

    best_idx = errors.argmin()

    print(f"\n🟢 Лучшее предсказание:")
    print(f"   Фактическая цена: {y_test.iloc[best_idx]:.0f} SR")
    print(f"   Предсказанная цена: {y_pred[best_idx]:.0f} SR")
    print(f"   Ошибка: {errors[best_idx]:.0f} SR ({errors[best_idx] / y_test.iloc[best_idx] * 100:.1f}%)")

    fig = plt.figure(figsize=(12, 8))

    explanation_best = shap.Explanation(
        values=shap_values[best_idx],
        base_values=base_value,
        feature_names=feature_names,
        data=X_test_processed[best_idx]
    )

    shap.plots.waterfall(explanation_best, show=False, max_display=15)
    plt.title(f'Объяснение лучшего предсказания\n'
              f'Факт: {y_test.iloc[best_idx]:.0f} SR → Предсказано: {y_pred[best_idx]:.0f} SR',
              fontweight='bold', fontsize=11)
    plt.tight_layout()
    save_plot(fig, 'shap_waterfall_best')
    plt.close()
    print("  ✓ Сохранено: shap_waterfall_best.png и shap_waterfall_best.svg")

    # ========================================================================
    # 5. ТОП-10 SHAP
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ТОП-10 ПРИЗНАКОВ ПО SHAP ВАЖНОСТИ:")
    print(f"{'=' * 70}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

    shap_importance_df['Importance_%'] = (
            shap_importance_df['SHAP_Importance'] /
            shap_importance_df['SHAP_Importance'].sum() * 100
    )

    print(shap_importance_df.head(10).to_markdown(index=False, floatfmt=".4f"))

    # ========================================================================
    # 6. ВЫВОДЫ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВЫВОДЫ SHAP-АНАЛИЗА:")
    print(f"{'=' * 70}")

    top_feature = shap_importance_df.iloc[0]['Feature']
    top_importance = shap_importance_df.iloc[0]['Importance_%']
    print(f"\n1. Самый важный признак: {top_feature} ({top_importance:.1f}%)")


    def get_shap_type(fname: str) -> str:
        fname = str(fname)
        if fname.startswith('num__'):
            return 'Числовые'
        elif fname.startswith('cat__'):
            return 'Категория'
        elif fname.startswith('bin__'):
            return 'Бинарные'
        return 'Другие'

    shap_importance_df['Type'] = shap_importance_df['Feature'].apply(get_shap_type)
    type_shap = shap_importance_df.groupby('Type')['Importance_%'].sum().sort_values(ascending=False)

    print(f"\n2. Важность по типам (SHAP):")
    for t, imp in type_shap.items():
        print(f"   • {t}: {imp:.1f}%")

    return shap_importance_df


# =============================================================================
# ЧАСТЬ XX: ПРОВЕРКА РАСПРЕДЕЛЕНИЯ sellable_online
# =============================================================================

def plot_residual_analysis(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_prefix: str = 'ml_residual_analysis'
) -> None:
    """
    Создаёт и визуализирует графики остатков для глубокого анализа качества регрессионной модели.

    Функция рассчитывает остатки (разность между фактическими и предсказанными значениями),
    выводит в консоль их базовые статистические метрики и строит двухпанельный график:
    1. **Residuals vs Predicted (График остатков)**: Показывает зависимость остатков от
       предсказанных значений. Точки группируются и окрашиваются по ценовым сегментам
       (Бюджетный, Средний, Премиум), а также накладывается линия скользящего среднего
       для поиска нелинейных трендов (смещения).
    2. **Distribution of Residuals (Распределение остатков)**: Строит гистограмму плотности
       остатков и накладывает на неё кривую идеального нормального распределения для
       проверки симметричности и выявления тяжелых хвостов.

    В конце работы функция проводит дополнительный анализ в консоли:
    - Оценивает систематическое смещение (Bias) в разрезе ценовых сегментов.
    - Выполняет эвристический тест на гетероскедастичность (различие дисперсии остатков
      в разных квинтилях предсказанных цен).

    Args:
        model (Pipeline): Обученный пайплайн машинного обучения, у которого есть метод
            `predict`. Предполагается, что модель предсказывает логарифмированные значения.
        X_test (pd.DataFrame): Матрица признаков тестовой выборки.
        y_test (pd.Series): Фактические ответы тестовой выборки (в исходном масштабе, SR).
        save_prefix (str, optional): Префикс имени файлов для сохранения графиков
            в форматах .png и .svg. По умолчанию 'ml_residual_analysis'.

    Returns:
        None: Функция выводит отчет в консоль и сохраняет графики на диск.

    Raises:
        Exception: Если построение скользящего среднего или визуализация графиков
            вызывает ошибку (обрабатывается внутри функции через try-except block).
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ОСТАТКОВ (RESIDUAL ANALYSIS)")
    print("=" * 70)

    # Получаем предсказания
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # Обратное логарифмирование

    # Вычисляем остатки
    residuals = y_test - y_pred

    # Статистика остатков
    print(f"\n📊 Статистика остатков:")
    print(f"  • Mean residual: {residuals.mean():.2f} SR")
    print(f"  • Median residual: {np.median(residuals):.2f} SR")
    print(f"  • Std residual: {residuals.std():.2f} SR")
    print(f"  • Min residual: {residuals.min():.2f} SR")
    print(f"  • Max residual: {residuals.max():.2f} SR")

    # Создаём фигуру с 2 подграфика
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ========================================================================
    # График 1: Residuals vs Predicted
    # ========================================================================
    ax1 = axes[0]

    # Определяем ценовые сегменты для цветовой маркировки
    price_segments = []
    for price in y_test:
        if price < 400:
            price_segments.append('Бюджетный')
        elif price < 1000:
            price_segments.append('Средний')
        else:
            price_segments.append('Премиум')

    # Создаём DataFrame для удобной фильтрации
    df_residuals = pd.DataFrame({
        'y_pred': y_pred,
        'residuals': residuals,
        'segment': price_segments
    })

    # Цвета для сегментов
    colors = {'Бюджетный': '#3498db', 'Средний': '#2ecc71', 'Премиум': '#e74c3c'}

    # Рисуем точки для каждого сегмента
    for segment, color in colors.items():
        mask = df_residuals['segment'] == segment
        ax1.scatter(
            df_residuals.loc[mask, 'y_pred'],
            df_residuals.loc[mask, 'residuals'],
            alpha=0.6, s=30, color=color, label=segment, edgecolors='black', linewidth=0.3
        )

    # Горизонтальная линия на y=0
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Идеальное предсказание')

    # Добавляем скользящее среднее для выявления тренда
    try:
        # Сортируем по x для сглаживания
        sorted_indices = np.argsort(y_pred)
        x_sorted = y_pred[sorted_indices]
        # 🔧 ИСПРАВЛЕНИЕ: используем .values для позиционной индексации
        # (residuals — Series с не-sequential индексами после train_test_split)
        y_sorted = residuals.values[sorted_indices]

        # Простое скользящее среднее для сглаживания
        window = 50
        y_smooth = pd.Series(y_sorted).rolling(window=window, center=True).mean()

        ax1.plot(x_sorted, y_smooth.values, color='purple', linewidth=2,
                 linestyle='-', label='Тренд (скользящее среднее)', alpha=0.7)

    except Exception as e:
        print(f"  ⚠️ Не удалось построить тренд: {e}")

    ax1.set_xlabel('Предсказанная цена (SR)', fontsize=11)
    ax1.set_ylabel('Остаток (Факт - Предсказание) (SR)', fontsize=11)
    ax1.set_title('Residuals vs Predicted\nАнализ гетероскедастичности и смещения',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # График 2: Distribution of Residuals
    # ========================================================================
    ax2 = axes[1]

    # Гистограмма остатков
    ax2.hist(residuals, bins=50, color='steelblue', edgecolor='black',
             linewidth=0.5, alpha=0.7, density=True)

    # Добавляем нормальное распределение для сравнения

    mu, std = norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    p = norm.pdf(x_range, mu, std)
    ax2.plot(x_range, p, 'r-', linewidth=2, label=f'Нормальное распределение\n(μ={mu:.1f}, σ={std:.1f})')

    # Вертикальная линия на x=0
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевое смещение')

    ax2.set_xlabel('Остаток (SR)', fontsize=11)
    ax2.set_ylabel('Плотность', fontsize=11)
    ax2.set_title('Распределение остатков\nПроверка нормальности',
                  fontweight='bold', fontsize=13, pad=15)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, save_prefix)
    plt.close()

    print(f"\n✓ Сохранено: {save_prefix}.png и .svg")

    # ========================================================================
    # Анализ по ценовым сегментам
    # ========================================================================
    print(f"\n📊 Анализ остатков по ценовым сегментам:")
    for segment in ['Бюджетный', 'Средний', 'Премиум']:
        mask = df_residuals['segment'] == segment
        segment_residuals = df_residuals.loc[mask, 'residuals']

        print(f"\n  {segment}:")
        print(f"    • Количество товаров: {mask.sum()}")
        print(f"    • Mean residual: {segment_residuals.mean():.2f} SR")
        print(f"    • Median residual: {segment_residuals.median():.2f} SR")
        print(f"    • Std residual: {segment_residuals.std():.2f} SR")

        # Интерпретация
        mean_res = segment_residuals.mean()
        if mean_res > 50:
            print(f"    ⚠️ СИСТЕМАТИЧЕСКАЯ НЕДООЦЕНКА (модель занижает цены)")
        elif mean_res < -50:
            print(f"    ⚠️ СИСТЕМАТИЧЕСКАЯ ПЕРЕОЦЕНКА (модель завышает цены)")
        else:
            print(f"    ✅ Смещение в пределах нормы")

    # ========================================================================
    # Тест на гетероскедастичность
    # ========================================================================
    print(f"\n🔍 Тест на гетероскедастичность:")

    # Разделяем предсказания на квинтили
    quintiles = pd.qcut(y_pred, q=5, labels=False, duplicates='drop')

    # Вычисляем дисперсию остатков для каждого квинтиля
    var_by_quintile = []
    for q in range(5):
        mask = quintiles == q
        if mask.sum() > 0:
            var_by_quintile.append(residuals[mask].var())

    if len(var_by_quintile) > 1:
        variance_ratio = max(var_by_quintile) / min(var_by_quintile)
        print(f"  • Дисперсия остатков по квинтилям: {[f'{v:.0f}' for v in var_by_quintile]}")
        print(f"  • Отношение max/min дисперсии: {variance_ratio:.2f}")

        if variance_ratio > 2:
            print(f"  ⚠️ ОБНАРУЖЕНА ГЕТЕРОСКЕДАСТИЧНОСТЬ (дисперсия растёт с ростом цены)")
            print(f"     Рекомендация: использовать взвешенную регрессию или трансформацию")
        else:
            print(f"  ✅ Гетероскедастичность не обнаружена")

    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ОСТАТКОВ ЗАВЕРШЁН")
    print(f"{'=' * 70}")

#=====================================================================================
# Проверка нелинейной зависмост признаков от целевой переменной
#====================================================================================

def check_nonlinearity(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n_features: int = 5
) -> Dict[str, Any]:
    """
    Проверяет нелинейность зависимости признаков от целевой переменной
    с использованием Partial Dependence Plots (PDP) и SHAP dependence plots.

    Функция анализирует топ-N признаков модели (по feature_importances_) и
    определяет, является ли их влияние на целевую переменную линейным или
    нелинейным. Для каждого признака строятся два графика:
    1. PDP + ICE curves — показывают среднее и индивидуальное влияние признака.
    2. SHAP dependence plot — показывает зависимость SHAP values от значений признака.

    Нелинейность определяется по двум критериям:
    - Немонотонность PDP кривой (признак имеет локальные экстремумы).
    - Превышение R² полиномиальной регрессии (степень 2) над линейной более чем на 5%.

    🔧 ДИНАМИЧЕСКАЯ ОБРАБОТКА ТИПОВ ПРИЗНАКОВ:
    Функция автоматически определяет тип каждого признака (числовой, бинарный,
    категориальный) и применяет соответствующую стратегию анализа:
    - Числовые/бинарные: полный PDP + SHAP анализ
    - Категориальные: пропуск PDP (не поддерживается), агрегированный SHAP анализ

    Parameters
    ----------
    model : Pipeline
        Обученный sklearn Pipeline с шагами 'preprocessor' и 'model'.
        Модель должна поддерживать feature_importances_ (деревья, ансамбли)
        или быть совместима с SHAP TreeExplainer.
    X_test : pd.DataFrame
        Тестовая выборка признаков (до препроцессинга).
    y_test : pd.Series
        Тестовые целевые значения (в оригинальной шкале, не log).
    top_n_features : int, default=5
        Количество наиболее важных признаков для анализа.
        Рекомендуемые значения: 5-10 для интерпретируемости.

    Returns
    -------
    Dict[str, Any]
        Словарь с результатами анализа для каждого признака:
        {
            'feature_name': {
                'feature_type': str,           # 'numeric', 'binary', 'categorical'
                'pdp_range': float,            # Размах PDP кривой (или None для категориальных)
                'is_monotonic': bool,          # Монотонна ли PDP кривая
                'linear_r2': float,            # R² линейной регрессии
                'poly_r2': float,              # R² полиномиальной регрессии
                'r2_improvement': float,       # ΔR² = poly_r2 - linear_r2
                'nonlinearity_detected': bool  # Итоговый вердикт
            },
            ...
            'summary': {
                'total_features': int,
                'nonlinear_features': int,
                'nonlinear_feature_names': List[str],
                'recommendation': str
            }
        }

    Notes
    -----
    Методологическое обоснование:
    - PDP (Partial Dependence Plot) показывает маргинальный эффект признака
      на предсказание модели, усреднённый по всем остальным признакам.
    - ICE (Individual Conditional Expectation) показывает индивидуальные
      кривые для каждого объекта — позволяет выявить скрытую нелинейность.
    - SHAP dependence plot показывает, как SHAP value (вклад признака в
      предсказание) зависит от значения признака. Нелинейная форма кривой
      указывает на сложные взаимодействия с другими признаками.

    🔧 Обработка категориальных признаков:
    - Для PDP: категориальные признаки пропускаются (sklearn не поддерживает
      PDP для некодированных категориальных признаков напрямую).
    - Для SHAP: используется агрегация SHAP значений по всем one-hot колонкам,
      соответствующим данному категориальному признаку. Это позволяет оценить
      общее влияние категории на предсказание.

    Examples
    --------
    >>> results = check_nonlinearity(
    ...     model=best_model_pipeline,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     top_n_features=5
    ... )
    >>> for feat, data in results.items():
    ...     if feat == 'summary':
    ...         continue
    ...     print(f"{feat}: нелинейность = {data['nonlinearity_detected']}")

    See Also
    --------
    plot_residual_analysis : Анализ остатков модели.
    analyze_with_shap : Глобальный SHAP-анализ.
    """
    import shap
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    print("\n" + "=" * 70)
    print("ПРОВЕРКА НЕЛИНЕЙНОСТИ ЗАВИСИМОСТИ ПРИЗНАКОВ ОТ ЦЕНЫ")
    print("=" * 70)

    # Извлекаем модель и препроцессор из Pipeline
    pipeline_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']

    # Получаем имена признаков после препроцессинга
    try:
        feature_names_processed = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names_processed = [f"feature_{i}" for i in range(X_test.shape[1])]

    # ========================================================================
    # 🔧 Вспомогательная функция: определение типа признака
    # ========================================================================
    def _get_feature_type(feature_name: str, X: pd.DataFrame) -> str:
        """Определяет тип признака по имени и данным.

        Returns: 'numeric', 'binary', 'categorical'
        """
        # 🔧 ИСПРАВЛЕНИЕ: расширенная проверка строковых типов
        # pandas может использовать dtype='object', 'string', 'StringDtype' или 'category'
        if feature_name in X.columns:
            col = X[feature_name]
            # Проверяем dtype через строковое представление (надёжнее, чем прямое сравнение)
            dtype_str = str(col.dtype).lower()
            is_string_dtype = (
                col.dtype == object or
                dtype_str in ('object', 'string', 'category', 'stringdtype') or
                pd.api.types.is_string_dtype(col)
            )

            # Дополнительная проверка: если колонка не пустая, проверяем тип первого элемента
            if not is_string_dtype and len(col) > 0:
                first_val = col.dropna().iloc[0] if col.notna().any() else None
                if first_val is not None and isinstance(first_val, str):
                    is_string_dtype = True

            if is_string_dtype:
                return 'categorical'

            # Проверяем бинарность (только 2 уникальных значения)
            unique_vals = col.dropna().unique()
            if len(unique_vals) == 2:
                return 'binary'
            return 'numeric'
        return 'unknown'

    # ========================================================================
    # 🔧 Вспомогательная функция: извлечение базового имени признака
    # ========================================================================
    def get_base_name(fname):
        fname = str(fname)
        if fname.startswith('cat__category_'):
            return 'category'
        elif fname.startswith('num__'):
            return fname[5:]
        elif fname.startswith('bin__'):
            return fname[5:]
        elif fname.startswith('cat__'):
            parts = fname[5:].split('_')
            return parts[0] if len(parts) > 1 else fname[5:]
        return fname

    # ========================================================================
    # Определяем топ-N признаков по важности
    # ========================================================================
    if hasattr(pipeline_model, 'feature_importances_'):
        importances = pipeline_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names_processed,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        importance_df['Base_Feature'] = importance_df['Feature'].apply(get_base_name)
        grouped_importance = importance_df.groupby('Base_Feature')['Importance'].sum()
        grouped_importance = grouped_importance.sort_values(ascending=False)
        top_features = grouped_importance.head(top_n_features).index.tolist()
    else:
        # Fallback: используем первые N признаков
        top_features = list(X_test.columns[:top_n_features])

    print(f"\n📊 Топ-{top_n_features} признаков для анализа:")
    for i, feat in enumerate(top_features, 1):
        ftype = _get_feature_type(feat, X_test)
        print(f"  {i}. {feat} (тип: {ftype})")

    results = {}

    # ========================================================================
    # 1. PARTIAL DEPENDENCE PLOTS (PDP)
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. PARTIAL DEPENDENCE PLOTS (PDP)")
    print("=" * 70)

    # 🔧 Конвертируем integer-колонки в float для PDP
    X_test_float = X_test.copy()
    n_converted = 0
    for col in X_test_float.columns:
        if X_test_float[col].dtype in ['int64', 'int32', 'int16', 'int8']:
            X_test_float[col] = X_test_float[col].astype(float)
            n_converted += 1
    print(f"  🔧 Конвертировано integer-колонок в float: {n_converted}")

    # Создаём фигуру для PDP графиков
    fig_pdp, axes_pdp = plt.subplots(2, 3, figsize=(18, 10))
    axes_pdp = axes_pdp.flatten()

    for i, feature in enumerate(top_features):
        if i >= len(axes_pdp):
            break

        ax = axes_pdp[i]

        # 🔧 ДИНАМИЧЕСКАЯ ОБРАБОТКА: определяем тип признака
        feature_type = _get_feature_type(feature, X_test)

        try:
            # 🔧 Для категориальных признаков — пропускаем PDP
            if feature_type == 'categorical':
                ax.text(0.5, 0.5,
                       f'Категориальный признак\n{feature}\n(PDP не поддерживается)',
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                results[feature] = {
                    'feature_type': 'categorical',
                    'pdp_range': None,
                    'is_monotonic': None,
                    'linear_r2': None,
                    'poly_r2': None,
                    'r2_improvement': None,
                    'nonlinearity_detected': None,
                    'pdp_skipped': True
                }
                print(f"\n  ⚠️ {feature}: категориальный признак — PDP пропущен")
                continue

            # 🔧 Для числовых и бинарных признаков — строим PDP
            if feature in X_test_float.columns:
                PartialDependenceDisplay.from_estimator(
                    model, X_test_float, [feature],
                    ax=ax,
                    feature_names=X_test_float.columns.tolist()
                )
                feature_idx = list(X_test_float.columns).index(feature)
                feature_values = X_test_float.iloc[:, feature_idx].values

                ax.set_title(f'{feature}\n(PDP, тип: {feature_type})', fontweight='bold', fontsize=10)

                # Анализ нелинейности PDP
                lines = ax.get_lines()
                if len(lines) > 0:
                    pdp_line = lines[0]
                    pdp_values = pdp_line.get_ydata()
                    pdp_range = float(np.max(pdp_values) - np.min(pdp_values))

                    # Проверяем монотонность
                    diffs = np.diff(pdp_values)
                    is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

                    # Полиномиальная регрессия для проверки нелинейности
                    feature_values_clean = feature_values[~np.isnan(feature_values)]
                    pdp_values_clean = pdp_values[:len(feature_values_clean)]

                    if len(feature_values_clean) > 10:
                        # Линейная модель
                        lin_model = LinearRegression()
                        lin_model.fit(feature_values_clean.reshape(-1, 1), pdp_values_clean)
                        lin_r2 = lin_model.score(feature_values_clean.reshape(-1, 1), pdp_values_clean)

                        # Полиномиальная модель (степень 2)
                        poly = PolynomialFeatures(degree=2)
                        feature_poly = poly.fit_transform(feature_values_clean.reshape(-1, 1))
                        poly_model = LinearRegression()
                        poly_model.fit(feature_poly, pdp_values_clean)
                        poly_r2 = poly_model.score(feature_poly, pdp_values_clean)

                        r2_improvement = poly_r2 - lin_r2
                        nonlinearity_detected = not is_monotonic or r2_improvement > 0.05

                        results[feature] = {
                            'feature_type': feature_type,
                            'pdp_range': pdp_range,
                            'is_monotonic': is_monotonic,
                            'linear_r2': lin_r2,
                            'poly_r2': poly_r2,
                            'r2_improvement': r2_improvement,
                            'nonlinearity_detected': nonlinearity_detected,
                            'pdp_skipped': False
                        }

                        print(f"\n  ✓ {feature} (тип: {feature_type}):")
                        print(f"    • PDP range: {pdp_range:.3f}")
                        print(f"    • Monotonic: {is_monotonic}")
                        print(f"    • Linear R²: {lin_r2:.3f}")
                        print(f"    • Polynomial R²: {poly_r2:.3f}")
                        print(f"    • ΔR²: {r2_improvement:+.3f}")
                        if nonlinearity_detected:
                            print(f"    • ⚠️ Обнаружена нелинейность")
                        else:
                            print(f"    • ✅ Линейная зависимость")
                    else:
                        results[feature] = {'feature_type': feature_type, 'error': 'insufficient_data'}
                else:
                    results[feature] = {'feature_type': feature_type, 'error': 'no_pdp_lines'}
            else:
                ax.text(0.5, 0.5, f'Признак не найден:\n{feature}',
                       ha='center', va='center', fontsize=10)
                results[feature] = {'feature_type': feature_type, 'error': 'not_found'}

        except Exception as e:
            print(f"\n  ⚠️ Ошибка при построении PDP для {feature}: {e}")
            ax.text(0.5, 0.5, f'Ошибка:\n{str(e)[:50]}',
                   ha='center', va='center', fontsize=9)
            results[feature] = {'feature_type': feature_type, 'error': str(e)}

    # Скрываем неиспользованные подграфики
    for i in range(len(top_features), len(axes_pdp)):
        axes_pdp[i].set_visible(False)

    plt.tight_layout()
    save_plot(fig_pdp, 'check_nonlinearity_pdp')
    plt.close()
    print("\n  ✓ Сохранено: check_nonlinearity_pdp.png и .svg")

    # ========================================================================
    # 2. SHAP DEPENDENCE PLOTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. SHAP DEPENDENCE PLOTS")
    print("=" * 70)

    # Преобразуем X_test через препроцессор
    X_test_processed = preprocessor.transform(X_test)

    # Создаём SHAP explainer
    explainer = shap.TreeExplainer(pipeline_model)
    shap_values = explainer.shap_values(X_test_processed)

    # Создаём фигуру для SHAP dependence plots
    fig_shap, axes_shap = plt.subplots(2, 3, figsize=(18, 10))
    axes_shap = axes_shap.flatten()

    for i, feature in enumerate(top_features):
        if i >= len(axes_shap):
            break

        ax = axes_shap[i]

        try:
            # 🔧 ДИНАМИЧЕСКАЯ ОБРАБОТКА: определяем тип признака
            feature_type = _get_feature_type(feature, X_test)

            # 🔧 Для КАТЕГОРИАЛЬНЫХ признаков — агрегируем SHAP по one-hot колонкам
            if feature_type == 'categorical':
                # Находим все one-hot колонки для этого признака
                cat_prefix = f'cat__{feature}_'
                cat_indices = [
                    idx for idx, name in enumerate(feature_names_processed)
                    if name.startswith(cat_prefix)
                ]

                if len(cat_indices) > 0:
                    # Агрегируем SHAP значения (сумма абсолютных значений по всем one-hot)
                    shap_aggregated = np.sum(np.abs(shap_values[:, cat_indices]), axis=1)

                    # Для визуализации используем первый one-hot индекс как представитель
                    representative_idx = cat_indices[0]
                    feature_values = X_test_processed[:, representative_idx]

                    # Строим scatter plot вручную (shap.dependence_plot не подходит для агрегации)
                    ax.scatter(feature_values, shap_aggregated, alpha=0.3, s=10, c='steelblue')
                    ax.set_xlabel(f'{feature} (one-hot representative)', fontsize=9)
                    ax.set_ylabel('|SHAP value| (сумма)', fontsize=9)
                    ax.set_title(f'{feature}\n(SHAP, категориальный)', fontweight='bold', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Анализ нелинейности через SHAP (агрегированные значения)
                    # Для категориальных используем дисперсию SHAP как меру влияния
                    shap_mean = np.mean(shap_aggregated)
                    shap_std = np.std(shap_aggregated)

                    # Упрощённая проверка: если std > 0, есть нелинейность
                    nonlinearity_detected = shap_std > 0.01 * shap_mean if shap_mean > 0 else False

                    # Сохраняем результаты
                    if feature not in results:
                        results[feature] = {'feature_type': 'categorical'}
                    results[feature]['shap_aggregated_mean'] = shap_mean
                    results[feature]['shap_aggregated_std'] = shap_std
                    results[feature]['shap_onehot_count'] = len(cat_indices)
                    results[feature]['nonlinearity_detected'] = nonlinearity_detected

                    print(f"\n  ✓ {feature} (SHAP, категориальный):")
                    print(f"    • One-hot колонок: {len(cat_indices)}")
                    print(f"    • Средний |SHAP|: {shap_mean:.4f}")
                    print(f"    • Std |SHAP|: {shap_std:.4f}")
                    if nonlinearity_detected:
                        print(f"    • ⚠️ Обнаружена нелинейность в SHAP")
                    else:
                        print(f"    • ✅ Слабое влияние в SHAP")
                else:
                    ax.text(0.5, 0.5, f'Категория не найдена\nв one-hot кодировке',
                           ha='center', va='center', fontsize=10)
                    if feature not in results:
                        results[feature] = {'feature_type': 'categorical'}
                    results[feature]['error'] = 'no_onehot_columns'

                continue

            # 🔧 Для ЧИСЛОВЫХ и БИНАРНЫХ признаков — стандартный SHAP анализ
            if feature in X_test.columns:
                # Находим индекс в X_test_processed
                processed_name_num = f'num__{feature}'
                processed_name_bin = f'bin__{feature}'

                if processed_name_num in feature_names_processed:
                    feature_idx = list(feature_names_processed).index(processed_name_num)
                elif processed_name_bin in feature_names_processed:
                    feature_idx = list(feature_names_processed).index(processed_name_bin)
                else:
                    # 🔧 ИСПРАВЛЕНИЕ: не используем base_idx_in_X — это неверно для X_test_processed
                    print(f"\n  ⚠️ {feature}: не найден в processed features (пропущен)")
                    ax.text(0.5, 0.5, f'Признак не найден\nв processed features',
                           ha='center', va='center', fontsize=10)
                    if feature not in results:
                        results[feature] = {'feature_type': feature_type}
                    results[feature]['error'] = 'not_in_processed'
                    continue

                feature_values = X_test_processed[:, feature_idx]
                shap_vals = shap_values[:, feature_idx]

                # SHAP dependence plot
                shap.dependence_plot(
                    feature_idx, shap_values, X_test_processed,
                    feature_names=list(feature_names_processed),
                    ax=ax,
                    show=False
                )

                ax.set_title(f'{feature}\n(SHAP, тип: {feature_type})', fontweight='bold', fontsize=10)

                # Анализ нелинейности через SHAP
                # Линейная модель
                lin_model = LinearRegression()
                lin_model.fit(feature_values.reshape(-1, 1), shap_vals)
                lin_r2 = lin_model.score(feature_values.reshape(-1, 1), shap_vals)

                # Полиномиальная модель (степень 2)
                poly = PolynomialFeatures(degree=2)
                feature_poly = poly.fit_transform(feature_values.reshape(-1, 1))
                poly_model = LinearRegression()
                poly_model.fit(feature_poly, shap_vals)
                poly_r2 = poly_model.score(feature_poly, shap_vals)

                r2_improvement = poly_r2 - lin_r2
                nonlinearity_detected = r2_improvement > 0.05

                # Сохраняем значения в results
                if feature not in results:
                    results[feature] = {'feature_type': feature_type}
                results[feature]['shap_linear_r2'] = lin_r2
                results[feature]['shap_poly_r2'] = poly_r2
                results[feature]['shap_r2_improvement'] = r2_improvement
                if nonlinearity_detected:
                    results[feature]['nonlinearity_detected'] = True
                else:
                    results[feature]['nonlinearity_detected'] = False

                print(f"\n  ✓ {feature} (SHAP, тип: {feature_type}):")
                print(f"    • Linear R²: {lin_r2:.3f}")
                print(f"    • Polynomial R²: {poly_r2:.3f}")
                print(f"    • ΔR²: {r2_improvement:+.3f}")
                if nonlinearity_detected:
                    print(f"    • ⚠️ Обнаружена нелинейность в SHAP")
                else:
                    print(f"    • ✅ Линейная зависимость в SHAP")

            else:
                ax.text(0.5, 0.5, f'Признак не найден:\n{feature}',
                       ha='center', va='center', fontsize=10)
                if feature not in results:
                    results[feature] = {'feature_type': 'unknown'}
                results[feature]['error'] = 'not_in_X_test'

        except Exception as e:
            print(f"\n  ⚠️ Ошибка при построении SHAP для {feature}: {e}")
            ax.text(0.5, 0.5, f'Ошибка:\n{str(e)[:50]}',
                   ha='center', va='center', fontsize=9)
            if feature not in results:
                results[feature] = {}
            results[feature]['error'] = str(e)

    # Скрываем неиспользованные подграфики
    for i in range(len(top_features), len(axes_shap)):
        axes_shap[i].set_visible(False)

    plt.tight_layout()
    save_plot(fig_shap, 'check_nonlinearity_shap')
    plt.close()
    print("\n  ✓ Сохранено: check_nonlinearity_shap.png и .svg")

    # ========================================================================
    # 3. ИТОГОВЫЙ АНАЛИЗ
    # ========================================================================
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ АНАЛИЗ НЕЛИНЕЙНОСТИ")
    print("=" * 70)

    nonlinear_features = [
        f for f, r in results.items()
        if isinstance(r, dict) and r.get('nonlinearity_detected', False)
    ]

    if len(nonlinear_features) > 0:
        print(f"\n✅ Обнаружена нелинейность в {len(nonlinear_features)} признаках:")
        for feat in nonlinear_features:
            r = results[feat]
            # Используем shap_r2_improvement если есть, иначе r2_improvement
            r2_imp = r.get('shap_r2_improvement', r.get('r2_improvement', 0))
            if r2_imp is not None:
                print(f"  • {feat}: ΔR² = {r2_imp:+.3f}")
            else:
                print(f"  • {feat}: нелинейность обнаружена")
        print(f"\n📌 РЕКОМЕНДАЦИЯ:")
        print(f"   • Деревья решений (RandomForest, XGBoost) автоматически")
        print(f"     моделируют нелинейные зависимости — они предпочтительнее")
        print(f"     линейных моделей для данного набора признаков.")
    else:
        print(f"\n⚠️ Нелинейность не обнаружена")
        print(f"   Возможно, линейные модели будут работать не хуже деревьев")

    # Добавляем summary в результаты
    results['summary'] = {
        'total_features': top_n_features,
        'nonlinear_features': len(nonlinear_features),
        'nonlinear_feature_names': nonlinear_features,
        'recommendation': (
            "Деревья решений автоматически моделируют нелинейные зависимости"
            if len(nonlinear_features) > 0
            else "Линейные модели могут быть достаточны"
        )
    }

    print(f"\n{'=' * 70}")
    print("ПРОВЕРКА НЕЛИНЕЙНОСТИ ЗАВЕРШЕНА")
    print(f"{'=' * 70}")

    return results

#======================================================================================
# Оценкак потенциального финансового эффекта внедрения ML
#====================================================================================

def plot_revenue_improvement(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_prefix: str = 'ml_revenue_improvement'
) -> None:
    """Визуализирует и оценивает потенциальный финансовый эффект от внедрения ML-модели.

    Функция сравнивает цены, предсказанные моделью (оптимальные цены), с базовым сценарием
    (baseline). В качестве baseline используется медианная цена товара внутри его категории
    (если в данных есть колонка 'category'), либо общая медиана по всей выборке.

    Args:
        model (Pipeline): Обученный пайплайн машинного обучения, предсказывающий
            логарифмированную цену (target).
        X_test (pd.DataFrame): Матрица признаков тестовой выборки. Может содержать колонку
            'category' (категория товара) для расчёта медианы внутри категории.
        y_test (pd.Series): Фактические цены товаров из тестовой выборки в исходном
            масштабе (SR). Используются для расчёта baseline-медианы по категориям.
        save_prefix (str, optional): Префикс имени файлов для сохранения визуализации
            в форматах .png и .svg. По умолчанию 'ml_revenue_improvement'.

    Returns:
        None: Функция выводит текстовый бизнес-отчет в консоль и сохраняет графики на диск.

    Raises:
        Exception: Если построение графиков или группировка по категориям вызывают ошибку
            (обрабатывается внутри функции через try-except block с переходом на fallback-вариант).

    Note:
        - Медиана по категориям рассчитывается из `y_test` (фактические цены), а не из
          `X_test['price']`, так как колонка `price` удаляется из `X_test` на этапе
          подготовки данных в `prepare_ml_data`.
        - Количество продаж `quantity = 100` является демонстрационным параметром.
          В реальном сценарии его следует заменить на фактические данные о продажах.
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ПОТЕНЦИАЛЬНОГО УЛУЧШЕНИЯ ВЫРУЧКИ")
    print("=" * 70)

    # Получаем предсказания модели
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # ========================================================================
    # 🔧 ИСПРАВЛЕНИЕ: Базовая цена (медиана по категории)
    # ========================================================================
    # Используем y_test (фактические цены) вместо X_test['price'],
    # так как колонка 'price' удаляется из X_test в prepare_ml_data()
    try:
        if isinstance(X_test, pd.DataFrame) and 'category' in X_test.columns:
            # 🔧 Базовая цена = медиана из y_test, сгруппированная по X_test['category']
            category_median = pd.Series(y_test).groupby(X_test['category'].values).transform('median')
            baseline_price = category_median.values
            print(f"  ✅ Baseline: медиана по категории (из y_test)")
        else:
            # Используем общую медиану
            baseline_price = np.full_like(y_test, np.median(y_test))
            print(f"  ⚠️ Baseline: глобальная медиана (категории отсутствуют)")
    except Exception as e:
        print(f"  ⚠️ Ошибка при расчёте baseline: {e}. Используем глобальную медиану.")
        baseline_price = np.full_like(y_test, np.median(y_test))

    # Вычисляем разницу в цене
    price_diff = y_pred - baseline_price

    # Предполагаем количество продаж (для демонстрации)
    # В реальности это были бы реальные данные о продажах
    quantity = 100  # 100 единиц товара (пример)

    # Вычисляем улучшение выручки
    revenue_improvement = price_diff * quantity

    # Статистика
    print(f"\n📊 Статистика улучшения выручки:")
    print(f"  • Средняя разница в цене: {price_diff.mean():.2f} SR")
    print(f"  • Медианная разница в цене: {np.median(price_diff):.2f} SR")
    print(f"  • Товары с улучшением цены: {(price_diff > 0).sum()} ({(price_diff > 0).mean() * 100:.1f}%)")
    print(f"  • Товары с ухудшением цены: {(price_diff < 0).sum()} ({(price_diff < 0).mean() * 100:.1f}%)")

    # Создаём фигуру с 2 подграфика
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ========================================================================
    # График 1: Price Comparison (Waterfall Chart)
    # ========================================================================
    ax1 = axes[0]

    # Сортируем по разнице в цене
    sorted_indices = np.argsort(price_diff)
    sorted_diff = price_diff[sorted_indices]

    # Показываем топ-20 улучшений и топ-20 ухудшений
    top_improvements = sorted_diff[-20:]
    top_worsenings = sorted_diff[:20]

    ax1.bar(range(20), top_improvements, color='green', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Улучшение цены')
    ax1.bar(range(20, 40), top_worsenings, color='red', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Ухудшение цены')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax1.set_xlabel('Ранг товара', fontsize=11)
    ax1.set_ylabel('Разница в цене (SR)', fontsize=11)
    ax1.set_title('Топ-20 улучшений и ухудшений цены\n(ML модель vs Baseline)',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # График 2: Revenue Impact by Category
    # ========================================================================
    ax2 = axes[1]

    # Группируем по категориям (если есть)
    try:
        if isinstance(X_test, pd.DataFrame) and 'category' in X_test.columns:
            df_revenue = pd.DataFrame({
                'category': X_test['category'].values,
                'price_diff': price_diff,
                'revenue_imp': revenue_improvement
            })

            category_impact = df_revenue.groupby('category')['revenue_imp'].sum().sort_values(ascending=False)

            # Показываем топ-10 категорий
            top_categories = category_impact.head(10)
            colors_cat = ['green' if x > 0 else 'red' for x in top_categories.values]

            ax2.barh(range(len(top_categories)), top_categories.values,
                     color=colors_cat, alpha=0.7, edgecolor='black', linewidth=0.5)

            ax2.set_yticks(range(len(top_categories)))
            ax2.set_yticklabels(top_categories.index, fontsize=9)
            ax2.set_xlabel('Влияние на выручку (SR)', fontsize=11)
            ax2.set_title('Влияние на выручку по категориям\n(Топ-10 категорий)',
                          fontweight='bold', fontsize=13, pad=15)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='x')

            # Добавляем значения на столбцах
            for i, (idx, val) in enumerate(top_categories.items()):
                ax2.text(val + (50 if val > 0 else -50), i, f'{val:.0f}',
                         ha='left' if val > 0 else 'right', va='center', fontsize=8)
        else:
            # Если нет категорий, показываем общее распределение
            ax2.hist(revenue_improvement, bins=30, color='steelblue',
                     edgecolor='black', linewidth=0.5, alpha=0.7)
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевое улучшение')
            ax2.set_xlabel('Улучшение выручки (SR)', fontsize=11)
            ax2.set_ylabel('Количество товаров', fontsize=11)
            ax2.set_title('Распределение влияния на выручку',
                          fontweight='bold', fontsize=13, pad=15)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
    except Exception as e:
        print(f"  ⚠️ Не удалось построить график по категориям: {e}")
        # Fallback: гистограмма
        ax2.hist(revenue_improvement, bins=30, color='steelblue',
                 edgecolor='black', linewidth=0.5, alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевое улучшение')
        ax2.set_xlabel('Улучшение выручки (SR)', fontsize=11)
        ax2.set_ylabel('Количество товаров', fontsize=11)
        ax2.set_title('Распределение влияния на выручку',
                      fontweight='bold', fontsize=13, pad=15)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, save_prefix)
    plt.close()

    print(f"\n✓ Сохранено: {save_prefix}.png и .svg")

    # ========================================================================
    # Итоговая статистика
    # ========================================================================
    print(f"\n💰 Итоговое влияние на выручку:")
    total_improvement = revenue_improvement.sum()
    avg_improvement = revenue_improvement.mean()

    print(f"  • Общее улучшение выручки: {total_improvement:.2f} SR")
    print(f"  • Среднее улучшение на товар: {avg_improvement:.2f} SR")

    if total_improvement > 0:
        print(f"  ✅ ML-модель потенциально УВЕЛИЧИВАЕТ выручку")
    else:
        print(f"  ⚠️ ML-модель потенциально УМЕНЬШАЕТ выручку")

    # Процентное улучшение
    baseline_revenue = baseline_price.sum() * quantity
    improved_revenue = y_pred.sum() * quantity
    pct_improvement = (improved_revenue - baseline_revenue) / baseline_revenue * 100

    print(f"\n📈 Сравнение с baseline:")
    print(f"  • Baseline выручка: {baseline_revenue:.2f} SR")
    print(f"  • ML выручка: {improved_revenue:.2f} SR")
    print(f"  • Процентное улучшение: {pct_improvement:.2f}%")

    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ВЫРУЧКИ ЗАВЕРШЁН")
    print(f"{'=' * 70}")

# =============================================================================
# ABLATION STUDY – АНАЛИЗ ВКЛАДА ГРУПП ПРИЗНАКОВ
# =============================================================================

def run_interpretation_pipeline(ml_results: 'MLPipelineResult') -> dict:
    """
    Запускает сквозной пайплайн интерпретации и бизнес-валидации модели.

    Пайплайн агрегирует методы пост-модельного анализа для оценки качества,
    стабильности, прозрачности и экономической эффективности лучшей модели:

    1. Расчёт классической важности признаков (Feature Importances / MDI / Permutation).
    2. Анализ распределения абсолютных и относительных ошибок прогнозирования.
    3. Оценка доверительного интервала для метрики MAE методом Bootstrap.
    4. Сравнение качества модели со стратегическими baseline-решениями (медианы, наивные прогнозы).
    5. Локальная и глобальная интерпретация вкладов признаков через SHAP-анализ.
    6. Проверка нелинейности зависимости признаков от целевой переменной (PDP + SHAP dependence).
    7. Анализ остатков (Residual Analysis) для проверки гомоскедастичности и смещения.
    8. Экономический анализ потенциального прироста выручки (Revenue Improvement Analysis).
    9. Финальное переобучение (refit) пайплайна на полном объёме данных с логарифмированием таргета.

    Parameters
    ----------
    ml_results : dict
        Результаты выполнения ML-пайплайна (`run_ml_pipeline`).
        Обязательно должен содержать следующие ключи:

        - 'best_model_pipeline' (Pipeline): Обученный sklearn-пайплайн модели.
        - 'best_model_name' (str): Название лучшей модели.
        - 'X_train' (pd.DataFrame): Обучающая матрица признаков.
        - 'X_test' (pd.DataFrame): Тестовая матрица признаков.
        - 'y_train' (pd.Series): Обучающий вектор целевой переменной.
        - 'y_test' (pd.Series): Тестовый вектор целевой переменной.
        - 'X_full' (pd.DataFrame): Объединенная матрица признаков (train + test).
        - 'y_full' (pd.Series): Объединенный вектор целевой переменной.

    Returns
    -------
    dict
        Словарь с артефактами интерпретации и бизнес-метриками:

        - 'importance_df' (pd.DataFrame): Попризнаковая важность (детальная, с префиксами).
        - 'type_importance' (pd.Series): Сводная важность по типам признаков.
        - 'grouped_importance_df' (pd.DataFrame): Важность, сгруппированная по бизнес-блокам.
        - 'abs_errors' (np.ndarray): Вектор абсолютных ошибок на тесте.
        - 'errors_pct' (np.ndarray): Вектор процентных ошибок (MAPE-компоненты).
        - 'mae_mean' (float): Среднее значение MAE по Bootstrap.
        - 'mae_ci_lower' (float): Нижняя граница 95% доверительного интервала MAE.
        - 'mae_ci_upper' (float): Верхняя граница 95% доверительного интервала MAE.
        - 'mae_bootstraps' (np.ndarray): Все значения MAE из бутстрап-распределения.
        - 'baseline_df' (pd.DataFrame): Сравнительная таблица с бейзлайнами.
        - 'shap_importance_df' (pd.DataFrame): Глобальная важность признаков по SHAP.
        - 'nonlinearity_results' (dict): Результаты проверки нелинейности.
        - 'best_model_pipeline' (Pipeline): Переобученная на всех данных модель.

    Notes
    -----
    Финальный пайплайн `best_model_pipeline` переобучается на всех данных `X_full`
    с использованием логарифмированного таргета `np.log1p(y_full)`.
    Переобученная модель сохраняется на диск через joblib в `models/final_best_model.joblib`.
    Baseline-сравнение сохраняется в CSV для воспроизводимости.

    Examples
    --------
     ml_results = run_ml_pipeline(df_unique, eda_results)
    interpretation_results = run_interpretation_pipeline(ml_results)
    print(f"MAE: {interpretation_results['mae_mean']:.2f} SR")
    print(f"95% CI: [{interpretation_results['mae_ci_lower']:.2f}, {interpretation_results['mae_ci_upper']:.2f}]")
    # Используем переобученную модель для предсказаний
    final_model = interpretation_results['best_model_pipeline']
     predictions = final_model.predict(new_data)
    """

    best_model_pipeline = ml_results.best_model_pipeline
    # 🔧 best_model_name — обязательное поле dataclass (не может отсутствовать,
    # в отличие от старого dict, где .get(..., 'Unknown Model') был защитой
    # от KeyError). Прямой доступ через точку.
    best_model_name = ml_results.best_model_name
    X_train = ml_results.X_train
    X_test = ml_results.X_test
    y_train = ml_results.y_train
    y_test = ml_results.y_test
    X_full = ml_results.X_full
    y_full = ml_results.y_full

    # 1. Анализ важности признаков
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print(f"{'=' * 70}")

    try:
        importance_df, type_importance, grouped_importance_df = analyze_feature_importances(
            best_model_pipeline, X_train, y_train
        )
    except Exception as e:
        print(f"⚠️ Ошибка при анализе важности признаков: {e}")
        importance_df, type_importance, grouped_importance_df = None, None, None

    # 2. Анализ ошибок
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ОШИБОК МОДЕЛИ")
    print(f"{'=' * 70}")

    try:
        abs_errors, errors_pct = analyze_model_errors(
            best_model_pipeline, X_test, y_test
        )
    except Exception as e:
        print(f"⚠️ Ошибка при анализе ошибок: {e}")
        abs_errors, errors_pct = None, None

    # 3. Bootstrap CI
    print(f"\n{'=' * 70}")
    print("ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ MAE (Bootstrap)")
    print(f"{'=' * 70}")

    try:
        mae_mean, mae_ci_lower, mae_ci_upper, mae_bootstraps = bootstrap_mae_confidence_interval(
            best_model_pipeline, X_test, y_test, n_bootstrap=1000, confidence=0.95
        )
    except Exception as e:
        print(f"⚠️ Ошибка при расчёте Bootstrap CI: {e}")
        mae_mean, mae_ci_lower, mae_ci_upper, mae_bootstraps = 0.0, 0.0, 0.0, []

    # 4. Сравнение с baseline'ами
    print(f"\n{'=' * 70}")
    print("СРАВНЕНИЕ С BASELINE'АМИ")
    print(f"{'=' * 70}")

    try:
        baseline_df = compare_with_baselines(
            X_train, X_test, y_train, y_test, best_model_pipeline,
            our_model_name=best_model_name
        )
        if baseline_df is not None:
            os.makedirs('reports_step', exist_ok=True)
            baseline_df.to_csv('reports_step/baseline_comparison.csv', index=False)
            print(f"\n  ✓ Baseline сравнение сохранено в reports_step/baseline_comparison.csv")
    except Exception as e:
        print(f"⚠️ Ошибка при сравнении с baseline: {e}")
        baseline_df = None

    # 5. SHAP-анализ
    print(f"\n{'=' * 70}")
    print("SHAP-АНАЛИЗ ДЛЯ ИНТЕРПРЕТАЦИИ МОДЕЛИ")
    print(f"{'=' * 70}")

    try:
        shap_importance_df = analyze_with_shap(
            best_model_pipeline, X_test, y_test
        )
    except Exception as e:
        print(f"⚠️ Ошибка при SHAP-анализе: {e}")
        shap_importance_df = None

    # 6. Проверка нелинейности признаков (PDP + SHAP dependence)
    print(f"\n{'=' * 70}")
    print("ПРОВЕРКА НЕЛИНЕЙНОСТИ (PDP + SHAP DEPENDENCE)")
    print(f"{'=' * 70}")

    try:
        nonlinearity_results = check_nonlinearity(
            best_model_pipeline, X_test, y_test, top_n_features=5
        )
    except Exception as e:
        print(f"⚠️ Ошибка при проверке нелинейности: {e}")
        nonlinearity_results = {}

    # 7. Residual Analysis
    print(f"\n{'=' * 70}")
    print("RESIDUAL ANALYSIS (АНАЛИЗ ОСТАТКОВ)")
    print(f"{'=' * 70}")

    try:
        plot_residual_analysis(
            best_model_pipeline, X_test, y_test,
            save_prefix='ml_residual_analysis'
        )
    except Exception as e:
        print(f"⚠️ Ошибка при анализе остатков: {e}")

    # 8. Revenue Improvement Analysis
    print(f"\n{'=' * 70}")
    print("REVENUE IMPROVEMENT ANALYSIS (АНАЛИЗ ВЛИЯНИЯ НА ВЫРУЧКУ)")
    print(f"{'=' * 70}")

    try:
        plot_revenue_improvement(
            best_model_pipeline, X_test, y_test,
            save_prefix='ml_revenue_improvement'
        )
    except Exception as e:
        print(f"⚠️ Ошибка при анализе выручки: {e}")

    # 9. Переобучение на всех данных
    print(f"\n{'=' * 70}")
    print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ НА ВСЕХ ДАННЫХ (для продакшена)")
    print(f"{'=' * 70}")
    print(f"\n⚠️ ВНИМАНИЕ: Все метрики выше получены на честной оценке (train→test).")
    print(f"   Теперь переобучаем модель на всех данных для финального использования.\n")

    try:
        best_model_pipeline.fit(X_full, np.log1p(y_full))
        print(f"✅ Модель переобучена на {len(X_full)} примерах")

        import joblib
        os.makedirs('models', exist_ok=True)
        model_path = 'models/final_best_model.joblib'
        joblib.dump(best_model_pipeline, model_path)
        print(f"💾 Модель сохранена в {model_path}")
    except Exception as e:
        print(f"⚠️ Ошибка при переобучении модели: {e}")

    return {
        'importance_df': importance_df,
        'type_importance': type_importance,
        'grouped_importance_df': grouped_importance_df,
        'abs_errors': abs_errors,
        'errors_pct': errors_pct,
        'mae_mean': mae_mean,
        'mae_ci_lower': mae_ci_lower,
        'mae_ci_upper': mae_ci_upper,
        'mae_bootstraps': mae_bootstraps,
        'baseline_df': baseline_df,
        'shap_importance_df': shap_importance_df,
        'nonlinearity_results': nonlinearity_results,
        'best_model_pipeline': best_model_pipeline
    }

# ==============================================================================
# ИТОГОВЫЕ ВЫВОДЫ
# ==============================================================================
