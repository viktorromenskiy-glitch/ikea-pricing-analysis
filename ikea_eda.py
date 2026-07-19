"""
ikea_eda.py — часть модульного разбиения Step_project_Q14.
"""
from ikea_core import *


def check_sellable_online(df: pd.DataFrame) -> Tuple[str, float, float]:
    """
    Анализирует распределение бинарного признака `sellable_online` и оценивает его
    статистическую связь с целевой переменной (`price`).

    Функция решает классическую задачу отбора признаков (Feature Selection) по двум критериям:
    1.  **Информативность (дисперсия)**: Вычисляется доля мажоритарного класса (`p_true`).
        Если признак практически константен (более 95% значений относятся к одному классу),
        его дисперсия $p(1-p)$ стремится к нулю, что делает его неинформативным для большинства моделей.
    2.  **Статистическая значимость**: С помощью непараметрического критерия Манна-Уитни
        (Mann-Whitney U test) проверяется гипотеза о равенстве распределений цен для товаров,
        доступных и недоступных для онлайн-покупки. Критерий устойчив к выбросам и не требует
        нормальности распределения цен.

    На основе этих факторов функция формирует автоматический вердикт по дальнейшей судьбе признака.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий целевую колонку `price`
            и бинарный признак `sellable_online` (тип bool или приводимый к нему).

    Returns:
        Tuple[str, float, float]:
            Результаты анализа и рекомендация по фильтрации:
            1.  verdict (str): Рекомендованное действие для пайплайна:
                - "REMOVE": Признак не имеет дисперсии (>95% True) или различие цен в группах
                  статистически не значимо.
                - "KEEP": Признак сбалансирован или дисбаланс компенсируется высокой
                  статистической значимостью связи с ценой (разница цен в группах критична).
            2.  p_true (float): Доля товаров, доступных для покупки онлайн (значение от 0.0 до 1.0).
            3.  p_value (float): Достигаемый уровень значимости (p-value) критерия Манна-Уитни.

    Note:
        - Использование медианных цен вместо средних арифметических при сравнении групп
          позволяет избежать искажения результатов из-за единичных ультра-дорогих товаров.
        - Статистический тест автоматически пропускается с присвоением `p_value = 1.0`, если
          одна из сравниваемых групп (онлайн или офлайн товаров) оказывается пустой в исходном `df`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА РАСПРЕДЕЛЕНИЯ sellable_online")
    print("=" * 70)



    # ========================================================================
    # 1. Подсчёт распределения
    # ========================================================================
    print("\n 1. Распределение признака:")
    online_counts = df['sellable_online'].value_counts()
    print(online_counts)

    online_pct = df['sellable_online'].value_counts(normalize=True) * 100
    print("\n 2. Проценты:")
    print(online_pct)

    # ========================================================================
    # 2. Статистика
    # ========================================================================
    n_true = df['sellable_online'].sum()
    n_false = len(df) - n_true
    p_true = n_true / len(df)

    print(f"\n 3. Статистика:")
    print(f"  • Всего товаров: {len(df)}")
    print(f"  • Продаются онлайн (True): {n_true} ({p_true * 100:.1f}%)")
    print(f"  • НЕ продаются онлайн (False): {n_false} ({(1 - p_true) * 100:.1f}%)")
    print(f"  • Дисперсия признака: {p_true * (1 - p_true):.4f}")

    # ========================================================================
    # 3. Сравнение цен
    # ========================================================================
    print(f"\n 4. Сравнение цен:")
    median_online = df[df['sellable_online']]['price'].median()
    median_offline = df[~df['sellable_online']]['price'].median()

    print(f"  • Медианная цена онлайн-товаров: {median_online:.0f} SR")
    print(f"  • Медианная цена офлайн-товаров: {median_offline:.0f} SR")

    if median_offline > 0:
        ratio = median_offline / median_online
        print(f"  • Соотношение цен (офлайн/онлайн): {ratio:.2f}x")

    # ========================================================================
    # 4. Статистический тест
    # ========================================================================
    online_prices = df[df['sellable_online']]['price']
    offline_prices = df[~df['sellable_online']]['price']

    print(f"\n 5. Статистический тест (Mann-Whitney U):")

    if len(offline_prices) > 0 and len(online_prices) > 0:
        stat, p_value = mannwhitneyu(online_prices, offline_prices)
        print(f"  • U-статистика: {stat:.0f}")
        print(f"  • p-value: {p_value:.6f}")

        if p_value < 0.001:
            print(f"  •  Различие ВЫСОКО значимо (p < 0.001)")
            significance = "HIGH"
        elif p_value < 0.05:
            print(f"  •  Различие статистически значимо (p < 0.05)")
            significance = "YES"
        else:
            print(f"  • ❌ Различие НЕ статистически значимо (p >= 0.05)")
            significance = "NO"
    else:
        print(f"  • ⚠️ Недостаточно данных для теста")
        p_value = 1.0
        significance = "NO_DATA"

    # ========================================================================
    # 5. Рекомендация
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" РЕКОМЕНДАЦИЯ:")
    print(f"{'=' * 70}")

    if p_true > 0.95:
        print(f"\n⚠️ ПРИЗНАК ПОЧТИ КОНСТАНТА ({p_true * 100:.1f}% True)")
        print(f"   • Дисперсия слишком мала: {p_true * (1 - p_true):.4f}")
        print(f"   • Преподаватель ПРАВ — признак бесполезен")
        print(f"   • РЕКОМЕНДАЦИЯ: УДАЛИТЬ из модели")
        verdict = "REMOVE"
    elif p_true > 0.90 and significance == "NO":
        print(f"\n⚠️ ПРИЗНАК С НИЗКОЙ ДИСПЕРСИЕЙ ({p_true * 100:.1f}% True)")
        print(f"   • Различие в ценах НЕ значимо (p = {p_value:.4f})")
        print(f"   • Преподаватель ПРАВ — признак бесполезен")
        print(f"   • РЕКОМЕНДАЦИЯ: УДАЛИТЬ из модели")
        verdict = "REMOVE"
    elif p_true > 0.90 and significance in ["YES", "HIGH"]:
        print(f"\n ПРИЗНАК ПОЛЕЗЕН, НЕСМОТРЯ НА ДИСПЕРСИЮ")
        print(f"   • Распределение: {p_true * 100:.1f}% True, {(1 - p_true) * 100:.1f}% False")
        print(f"   • Различие в ценах СТАТИСТИЧЕСКИ ЗНАЧИМО (p = {p_value:.6f})")
        print(f"   • Офлайн-товары стоят в {ratio:.2f}x дороже")
        print(f"   • РЕКОМЕНДАЦИЯ: ОСТАВИТЬ в модели (с обоснованием)")
        verdict = "KEEP"
    else:
        print(f"\n ПРИЗНАК ОПРЕДЕЛЁННО ПОЛЕЗЕН")
        print(f"   • Распределение сбалансировано: {p_true * 100:.1f}% True")
        print(f"   • РЕКОМЕНДАЦИЯ: ОСТАВИТЬ в модели")
        verdict = "KEEP"

    return verdict, p_true, p_value

#=================================================================================
# Создание визуализаций для разведочного анализа
#================================================================================

def check_old_price_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Исследует природу целевого признака `old_price` на предмет утечки данных (data leakage).

    В задачах машинного обучения предсказания текущей цены (`price`) использование
    исторической цены (`old_price`) или производных от нее величин (размер скидки)
    может приводить к искусственному завышению метрик качества модели ($R^2$, $MAE$)
    на кросс-валидации, если эти признаки функционально зависимы от таргета. Данная
    функция проводит комплексный аудит этой связи.

    Методология и этапы проверки:
    1. **Предобработка и парсинг**: Строковое поле `old_price` очищается от валютных
       аббревиатур ('SR'), пробелов и конвертируется во float. В случае интервальных цен
       (например, '100-200') берется левая граница диапазона.
    2. **Проверка 1 (Линейная корреляция)**: Вычисляется коэффициент корреляции Пирсона
       между `price` и `old_price`. Значение $r > 0.98$ интерпретируется как явный
       сигнал высокой линейной когерентности и потенциальной утечки (`LEAKAGE`).
    3. **Проверка 2 (Отношение цен)**: Рассчитывается вектор отношений:
       $$Ratio = \\frac{OldPrice}{Price}$$
       Анализируется среднее, медиана, стандартное отклонение (`std`) и квантили распределения.
    4. **Проверка 3 (Вариативность скидки)**: Рассчитывается относительный размер скидки:
       $$Discount\\% = \\frac{OldPrice - Price}{OldPrice} \\times 100$$
       Если стандартное отклонение величины скидки критически мало ($std < 5\\%$), это означает,
       что скидка является константой, и признак несет нулевую энтропию.
    5. **Проверка 4 (Детекция функциональной зависимости)**: Если среднее отношение цен
       находится в интервале $(1.15; 1.35)$, а стандартное отклонение $std < 0.05$, это
       свидетельствует о том, что `old_price` восстанавливается через линейное преобразование
       `price` с фиксированным коэффициентом (прямая утечка).

    Формирование рекомендаций:
        *   **LEAKAGE (Утечка)**: Разрешено использовать только бинарный флаг наличия скидки
            `has_old_price`. Исходные значения цен и скидки исключаются из обучения.
        *   **RISKY (Опасность)**: Рекомендуется использовать `has_old_price` и относительную
            скидку `discount_pct`, но избегать прямой подачи признака `old_price`.
        *   **SAFE (Безопасно)**: Допускается использование всех признаков: флага, относительной
            и абсолютной величины скидки, а также исходной старой цены.

        ⚠️ ПРИМЕЧАНИЕ (важно при чтении вывода этой функции): пороги проверок 1–4 выше
        детектируют только ЛИНЕЙНУЮ утечку (old_price ≈ price × const). Они не ловят
        более простую структурную утечку в самой формуле discount_pct = (old_price -
        price) / old_price, которая по построению является функцией от target
        независимо от результата этих проверок. Поэтому даже при вердикте "RISKY"
        (не "LEAKAGE") итоговый пайплайн (см. prepare_ml_data()) сознательно
        исключает 'discount_pct' из обучения, переопределяя рекомендацию
        'features_to_use' этой функции. Сама функция оставлена без изменений —
        она полезна как диагностика/аудит, решение о фактическом использовании
        признаков принимается на уровне prepare_ml_data().

    Визуализация (экспортируется как `eda_old_price_leakage_check` в форматах .png и .svg):
        Сложная матрица графиков размером 2х2:
        - **[0, 0] Scatter Plot**: Диаграмма рассеяния текущей цены относительно старой с наложением
          линии идеального совпадения ($y = x$) и расчетной линии линейной регрессии.
        - **[0, 1] Ratio Histogram**: Распределение отношения цен с вертикальными маркерами
          среднего и медианы (помогает визуально обнаружить "константную" наценку).
        - **[1, 0] Discount Histogram**: Плотность распределения скидки в процентах.
        - **[1, 1] Boxplot**: Сравнение распределения процента скидки по топ-5 наиболее
          представленным категориям товаров (определяет категориальную специфику дисконтирования).

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий непрерывную числовую целевую
            переменную `price` и потенциально строковое поле `old_price`.

    Returns:
        Dict[str, Any]: Словарь с результатами аудита, содержащий ключи:
            - 'verdict' (str): Вердикт проверки ('LEAKAGE', 'RISKY', 'SAFE').
            - 'corr' (float): Рассчитанный коэффициент корреляции Пирсона.
            - 'ratio_mean' (float): Среднее отношение старой цены к новой.
            - 'ratio_std' (float): Стандартное отклонение отношения цен.
            - 'discount_mean' (float): Средний размер скидки в процентах.
            - 'discount_std' (float): Стандартное отклонение размера скидки.
            - 'n_with_old_price' (int): Объем подвыборки со скидкой.
            - 'n_without_old_price' (int): Объем подвыборки без скидки.
            - 'pct_with_old_price' (float): Доля товаров со скидкой в процентах.
            - 'features_to_use' (List[str]): Список безопасных для обучения признаков.
            - 'features_to_avoid' (List[str]): Список признаков, подлежащих исключению.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют колонки `old_price` или `price`.
        NameError: Если в контексте не импортированы модули `numpy` (как `np`),
            `pandas` (как `pd`), `matplotlib.pyplot` (как `plt`) или утилита `save_plot`.

    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ПРИРОДЫ old_price (ДЕТЕКЦИЯ УТЕЧКИ)")
    print("=" * 70)

    # Работаем с df_unique (без дубликатов)
    df_check = df.dropna(subset=['old_price', 'price']).copy()

    # Парсим old_price (в датасете это может быть строка)
    def parse_old_price(x: Any) -> float:
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

    df_check['old_price_parsed'] = df_check['old_price'].apply(parse_old_price)
    df_check = df_check.dropna(subset=['old_price_parsed']).copy()

    print(f"\n📊 Статистика old_price:")
    print(f"  • Всего товаров: {len(df)}")
    print(f"  • Товаров со старой ценой: {len(df_check)} ({len(df_check) / len(df) * 100:.1f}%)")
    print(f"  • Товаров без старой цены: {len(df) - len(df_check)} ({(len(df) - len(df_check)) / len(df) * 100:.1f}%)")

    # ========================================================================
    # ПРОВЕРКА 1: Корреляция price ↔ old_price (детекция утечки)
    # ========================================================================
    corr = df_check['price'].corr(df_check['old_price_parsed'])
    print(f"\n[1] КОРРЕЛЯЦИЯ price ↔ old_price:")
    print(f"  • Pearson r = {corr:.4f}")

    if corr > 0.98:
        print(f"  ⚠️  КРИТИЧНО ВЫСОКАЯ (>0.98) — возможна утечка данных!")
        corr_verdict = "LEAKAGE"
    elif corr > 0.95:
        print(f"  ⚠️  Высокая (0.95-0.98) — требуется осторожность")
        corr_verdict = "RISKY"
    else:
        print(f"  ✅ В пределах нормы (<0.95)")
        corr_verdict = "SAFE"

    # ========================================================================
    # ПРОВЕРКА 2: Отношение old_price / price (структура скидки)
    # ========================================================================
    df_check['ratio'] = df_check['old_price_parsed'] / df_check['price']
    ratio_mean = df_check['ratio'].mean()
    ratio_std = df_check['ratio'].std()
    ratio_median = df_check['ratio'].median()

    print(f"\n[2] СТРУКТУРА ОТНОШЕНИЯ old_price/price:")
    print(f"  • Среднее: {ratio_mean:.4f}")
    print(f"  • Медиана: {ratio_median:.4f}")
    print(f"  • Std:     {ratio_std:.4f}")
    print(f"  • Min/Max: {df_check['ratio'].min():.4f} / {df_check['ratio'].max():.4f}")
    print(f"  • Квантили:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"      {int(q * 100):3d}%: {df_check['ratio'].quantile(q):.4f}")

    # ========================================================================
    # ПРОВЕРКА 3: Вариативность скидки
    # ========================================================================
    df_check['discount_pct'] = (df_check['old_price_parsed'] - df_check['price']) / df_check['old_price_parsed']
    discount_mean = df_check['discount_pct'].mean() * 100
    discount_std = df_check['discount_pct'].std() * 100

    print(f"\n[3] ВАРИАТИВНОСТЬ СКИДКИ (%):")
    print(df_check['discount_pct'].describe() * 100)
    print(f"\n  • Средняя скидка: {discount_mean:.2f}%")
    print(f"  • Std скидки:     {discount_std:.2f}%")

    if discount_std < 5:
        print(f"  ⚠️  Скидка почти константа (std < 5%) — признак бесполезен")
        variability_verdict = "CONSTANT"
    else:
        print(f"  ✅ Скидка вариативна (std >= 5%) — признак информативен")
        variability_verdict = "VARIABLE"

    # ========================================================================
    # ПРОВЕРКА 4: Проверка old_price = price * const
    # ========================================================================
    print(f"\n[4] ПРОВЕРКА: old_price = price × const?")
    if ratio_mean > 1.15 and ratio_mean < 1.35 and ratio_std < 0.05:
        print(f"  ❌ ДА! old_price ≈ price × {ratio_mean:.3f} (std = {ratio_std:.4f})")
        print(f"     Это ПРЯМАЯ УТЕЧКА — old_price вычисляется из price!")
        structure_verdict = "FORMULA"
    else:
        print(f"  ✅ НЕТ. old_price — независимый якорь")
        structure_verdict = "INDEPENDENT"

    # ========================================================================
    # ИТОГОВЫЙ ВЕРДИКТ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("📋 ИТОГОВЫЙ ВЕРДИКТ:")
    print(f"{'=' * 70}")

    # Определяем общий вердикт
    if corr > 0.98 and ratio_std < 0.05:
        final_verdict = "LEAKAGE"
        print(f"\n❌ ПРЯМАЯ УТЕЧКА ДАННЫХ!")
        print(f"   old_price = price × {ratio_mean:.3f} (почти константа)")
        print(f"\n📌 РЕКОМЕНДАЦИЯ:")
        print(f"   • Использовать ТОЛЬКО has_old_price (бинарный флаг)")
        print(f"   • НЕ использовать old_price, discount_pct, discount_abs")
        print(f"   • Ожидаемый прирост R²: +0.002–0.005 (минимальный)")
        features_to_use = ['has_old_price']
        features_to_avoid = ['old_price', 'discount_pct', 'discount_abs']
    elif corr > 0.95:
        final_verdict = "RISKY"
        print(f"\n⚠️  ЧАСТИЧНАЯ УТЕЧКА (высокая корреляция)")
        print(f"\n📌 РЕКОМЕНДАЦИЯ:")
        print(f"   • Использовать has_old_price + discount_pct")
        print(f"   • НЕ использовать old_price напрямую")
        print(f"   • Ожидаемый прирост R²: +0.005–0.015")
        features_to_use = ['has_old_price', 'discount_pct']
        features_to_avoid = ['old_price']
    else:
        final_verdict = "SAFE"
        print(f"\n✅ НЕЗАВИСИМЫЙ ЯКОРЬ (old_price несёт новую информацию)")
        print(f"\n📌 РЕКОМЕНДАЦИЯ:")
        print(f"   • Использовать has_old_price + discount_pct + discount_abs")
        print(f"   • old_price можно использовать с осторожностью")
        print(f"   • Ожидаемый прирост R²: +0.010–0.025")
        features_to_use = ['has_old_price', 'discount_pct', 'discount_abs']
        features_to_avoid = []

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print(f"\n📊 Создание визуализации old_price vs price...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Scatter old_price vs price
    ax1 = axes[0, 0]
    ax1.scatter(df_check['old_price_parsed'], df_check['price'],
                alpha=0.5, s=20, c='steelblue', edgecolors='black', linewidth=0.3)
    max_val = max(df_check['old_price_parsed'].max(), df_check['price'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
             label=f'Идеальное совпадение (y = x)')
    # Линия регрессии
    slope, intercept = np.polyfit(df_check['old_price_parsed'], df_check['price'], 1)
    x_line = np.linspace(0, max_val, 100)
    ax1.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2,
             label=f'Регрессия: y = {slope:.2f}x + {intercept:.0f}')
    ax1.set_xlabel('Старая цена (SR)', fontsize=11)
    ax1.set_ylabel('Текущая цена (SR)', fontsize=11)
    ax1.set_title(f'old_price vs price\nКорреляция r = {corr:.3f}',
                  fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Распределение отношения old_price/price
    ax2 = axes[0, 1]
    ax2.hist(df_check['ratio'], bins=50, color='coral',
             edgecolor='black', alpha=0.7)
    ax2.axvline(ratio_mean, color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {ratio_mean:.3f}')
    ax2.axvline(ratio_median, color='green', linestyle='--', linewidth=2,
                label=f'Медиана: {ratio_median:.3f}')
    ax2.set_xlabel('Отношение old_price / price', fontsize=11)
    ax2.set_ylabel('Частота', fontsize=11)
    ax2.set_title(f'Распределение отношения old_price/price\nStd = {ratio_std:.4f}',
                  fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Распределение скидки (%)
    ax3 = axes[1, 0]
    ax3.hist(df_check['discount_pct'] * 100, bins=50, color='gold',
             edgecolor='black', alpha=0.7)
    ax3.axvline(discount_mean, color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {discount_mean:.1f}%')
    ax3.set_xlabel('Скидка (%)', fontsize=11)
    ax3.set_ylabel('Частота', fontsize=11)
    ax3.set_title(f'Распределение скидки\nStd = {discount_std:.2f}%',
                  fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Boxplot по категориям (если есть)
    ax4 = axes[1, 1]
    if 'category' in df_check.columns:
        top_cats = df_check['category'].value_counts().head(5).index
        df_top = df_check[df_check['category'].isin(top_cats)]
        data_to_plot = [df_top[df_top['category'] == cat]['discount_pct'].values * 100
                        for cat in top_cats]
        bp = ax4.boxplot(data_to_plot, tick_labels=[c[:15] for c in top_cats],
                         patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax4.set_ylabel('Скидка (%)', fontsize=11)
        ax4.set_title('Скидки по топ-5 категориям', fontweight='bold', fontsize=12)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Нет данных о категориях',
                 ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    save_plot(fig, 'eda_old_price_leakage_check')
    plt.close()
    print(f"  ✓ Сохранено: eda_old_price_leakage_check.png и .svg")

    # ========================================================================
    # ВОЗВРАЩАЕМ РЕЗУЛЬТАТЫ
    # ========================================================================
    results = {
        'verdict': final_verdict,
        'corr': corr,
        'ratio_mean': ratio_mean,
        'ratio_std': ratio_std,
        'discount_mean': discount_mean,
        'discount_std': discount_std,
        'n_with_old_price': len(df_check),
        'n_without_old_price': len(df) - len(df_check),
        'pct_with_old_price': len(df_check) / len(df) * 100,
        'features_to_use': features_to_use,
        'features_to_avoid': features_to_avoid,
    }

    print(f"\n{'=' * 70}")
    print(f"✅ Проверка old_price завершена. Вердикт: {final_verdict}")
    print(f"{'=' * 70}")

    return results

# ==============================================================================
# ОРКЕСТРАТОР EDA-ПАЙПЛАЙНА
# ==============================================================================

def create_missing_visualizations(df_unique: pd.DataFrame) -> None:
    """
    Создаёт 12  визуализаций для разведочного анализа данных (EDA).

    Функция генерирует, настраивает и сохраняет графики в форматах PNG и SVG
    с использованием библиотек Matplotlib и Seaborn. Внутри реализована устойчивая
    логика предобработки данных (парсинг цен, расчёт площадей/объёмов, обработка
    логарифмов с защитой от нулевых значений) и обработка исключений для пустых
    или несбалансированных срезов данных.

    Список генерируемых графиков:
        1.  Количество товаров по категориям (Horizontal Bar).
        2.  Медианная цена по категориям (Horizontal Bar).
        3.  Корреляционная матрица "цена-размеры" (Seaborn Heatmap).
        4.  Топ-15 дизайнеров по количеству товаров (Horizontal Bar).
        5.  Топ-15 дизайнеров по медианной цене (Horizontal Bar).
        6.  Количество товаров с дополнительными цветами и без них (Bar Chart).
        7.  Медианная цена товаров с дополнительными цветами и без них (Bar Chart).
        8.  Количество товаров, продаваемых онлайн и офлайн (Bar Chart).
        9.  Медианная цена товаров, продаваемых онлайн и офлайн (Bar Chart).
        10. Распределение цен с линиями среднего и медианы (Histogram).
        11. Распределение логарифма цен с медианой (Histogram).
        12. Зависимость новой цены от старой с линией регрессии и R² (Scatter Plot).

    Args:
        df_unique (pd.DataFrame): Очищенный датафрейм уникальных товаров IKEA.
            Обязательные колонки: 'category', 'price', 'width', 'height', 'depth',
            'sellable_online'.
            Опциональные колонки (обрабатываются автоматически): 'designer',
            'designer_clean', 'other_colors', 'old_price'.

    Returns:
        None. Все графики сохраняются локально через внешнюю функцию `save_plot`.

    Raises:
        NameError: Если функция `save_plot` не определена в глобальной области видимости.
        KeyError: Если в `df_unique` отсутствуют критически важные колонки для построения
            базовых метрик (например, 'price' или 'category').

    Note:
        - Для корреляционной матрицы автоматически рассчитываются признаки `area` (площадь)
          и `volume` (объём), если они отсутствуют в исходном датафрейме.
        - Парсинг `old_price` корректно обрабатывает текстовые значения валюты "SR",
          двойные диапазоны цен (берет нижнюю границу) и нечисловые артефакты.
        - При построении регрессии (График 12) требуется минимум 10 валидных пар
          "старая цена — новая цена". При их отсутствии график выведет текстовое предупреждение
    """

    # ========================================================================
    # 1. Количество товаров по категориям
    # ========================================================================
    print("\n 1. Количество товаров по категориям")

    fig, ax = plt.subplots(figsize=(12, 8))
    category_counts = df_unique['category'].value_counts().sort_values(ascending=True)

    bars = ax.barh(category_counts.index, category_counts.values,
                   color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Количество товаров', fontsize=11)
    ax.set_title('Количество товаров по категориям',
                 fontweight='bold', fontsize=13, pad=15)

    # Подписи под углом 45° (правым концом к оси)
    # Для горизонтальной диаграммы это не нужно, но добавим значения
    for bar, count in zip(bars, category_counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{count}', va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, 'eda_category_count')
    plt.close()
    print("  ✓ Сохранено: eda_category_count.png и .svg")

    # ========================================================================
    # 2. Медианная цена по категориям
    # ========================================================================
    print("\n 2. Медианная цена по категориям")

    fig, ax = plt.subplots(figsize=(12, 8))
    median_prices = df_unique.groupby('category')['price'].median().sort_values(ascending=True)

    bars = ax.barh(median_prices.index, median_prices.values,
                   color='darkorange', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Медианная цена (SR)', fontsize=11)
    ax.set_title('Медианная цена товаров по категориям',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, price in zip(bars, median_prices.values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f'{price:.0f} SR', va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, 'eda_category_median_price')
    plt.close()
    print("  ✓ Сохранено: eda_category_median_price.png и .svg")

    # ========================================================================
    # 3. Корреляционная матрица цена-размеры
    # ========================================================================
    print("\n 3. Корреляционная матрица цена-размеры")

    # Вычисляем volume и area если их нет
    df_corr = df_unique.copy()
    if 'volume' not in df_corr.columns:
        df_corr['volume'] = df_corr['width'] * df_corr['height'] * df_corr['depth']
    if 'area' not in df_corr.columns:
        df_corr['area'] = df_corr['width'] * df_corr['depth']

    cols_for_corr = ['price', 'width', 'height', 'depth', 'volume', 'area']
    df_corr_clean = df_corr[cols_for_corr].dropna()

    corr_matrix = df_corr_clean.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Корреляция'})

    ax.set_title('Корреляционная матрица: цена и размеры',
                 fontweight='bold', fontsize=13, pad=15)

    plt.tight_layout()
    save_plot(fig, 'eda_correlation_matrix')
    plt.close()
    print("  ✓ Сохранено: eda_correlation_matrix.png и .svg")

    # ========================================================================
    # 4. Количество товаров по дизайнерам (топ-15)
    # ========================================================================
    print("\n 4. Количество товаров по дизайнерам (топ-15)")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Используем designer_clean если есть, иначе designer
    designer_col = 'designer_clean' if 'designer_clean' in df_unique.columns else 'designer'
    designer_counts = df_unique[designer_col].value_counts().head(15).sort_values(ascending=True)

    bars = ax.barh(designer_counts.index, designer_counts.values,
                   color='forestgreen', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Количество товаров', fontsize=11)
    ax.set_title('Топ-15 дизайнеров по количеству товаров',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, count in zip(bars, designer_counts.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f'{count}', va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, 'eda_designer_count_top15')
    plt.close()
    print("  ✓ Сохранено: eda_designer_count_top15.png и .svg")

    # ========================================================================
    # 5. Медианная цена по дизайнерам (топ-15)
    # ========================================================================
    print("\n 5. Медианная цена по дизайнерам (топ-15)")

    fig, ax = plt.subplots(figsize=(12, 8))

    designer_median = df_unique.groupby(designer_col)['price'].median()
    designer_median = designer_median.sort_values(ascending=False).head(15).sort_values(ascending=True)

    bars = ax.barh(designer_median.index, designer_median.values,
                   color='darkred', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Медианная цена (SR)', fontsize=11)
    ax.set_title('Топ-15 дизайнеров по медианной цене',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, price in zip(bars, designer_median.values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f'{price:.0f} SR', va='center', fontsize=9)

    plt.tight_layout()
    save_plot(fig, 'eda_designer_median_price_top15')
    plt.close()
    print("  ✓ Сохранено: eda_designer_median_price_top15.png и .svg")

    # ========================================================================
    # 6. Количество товаров с/без разных цветов
    # ========================================================================
    print("\n📊 6. Количество товаров с/без разных цветов")

    fig, ax = plt.subplots(figsize=(8, 6))

    # 🔧 ИСПРАВЛЕНИЕ: колонка other_colors содержит 'Yes'/'No', а не NaN
    df_colors = df_unique.copy()

    # Проверяем, что в колонке (может быть 'Yes', 'No', NaN, или другие значения)
    df_colors['has_other_colors'] = (
            df_colors['other_colors'].astype(str).str.strip().str.lower() == 'yes'
    )

    # Считаем количество
    n_with = int(df_colors['has_other_colors'].sum())
    n_without = int((~df_colors['has_other_colors']).sum())

    print(f"\n  🔍 Анализ колонки 'other_colors':")
    print(f"    Уникальных значений: {df_colors['other_colors'].nunique()}")
    print(f"    Примеры значений: {df_colors['other_colors'].unique()[:5]}")
    print(f"    С другими цветами (Yes): {n_with}")
    print(f"    Без других цветов (No/другое): {n_without}")

    colors_count = pd.Series({
        'Без других цветов': n_without,
        'С другими цветами': n_with
    })

    bars = ax.bar(colors_count.index, colors_count.values,
                  color=['steelblue', 'darkorange'],
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Количество товаров', fontsize=11)
    ax.set_title('Количество товаров с/без разных цветов',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, count in zip(bars, colors_count.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{count}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'eda_colors_count')
    plt.close()
    print("  ✓ Сохранено: eda_colors_count.png и .svg")

    # ========================================================================
    # 7. Медианная цена с/без разных цветов
    # ========================================================================
    print("\n📊 7. Медианная цена с/без разных цветов")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Используем уже созданный has_other_colors
    median_with = float(df_colors.loc[df_colors['has_other_colors'], 'price'].median()) if n_with > 0 else 0.0
    median_without = float(df_colors.loc[~df_colors['has_other_colors'], 'price'].median()) if n_without > 0 else 0.0

    print(f"\n  💰 Медианные цены:")
    print(f"    С другими цветами: {median_with:.0f} SR")
    print(f"    Без других цветов: {median_without:.0f} SR")

    median_by_colors = pd.Series({
        'Без других цветов': median_without,
        'С другими цветами': median_with
    })

    bars = ax.bar(median_by_colors.index, median_by_colors.values,
                  color=['steelblue', 'darkorange'],
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Медианная цена (SR)', fontsize=11)
    ax.set_title('Медианная цена товаров с/без разных цветов',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, price in zip(bars, median_by_colors.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{price:.0f} SR', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'eda_colors_median_price')
    plt.close()
    print("  ✓ Сохранено: eda_colors_median_price.png и .svg")

    # ========================================================================
    # 8. Количество товаров онлайн/офлайн
    # ========================================================================
    print("\n 8. Количество товаров онлайн/офлайн")

    fig, ax = plt.subplots(figsize=(8, 6))

    online_count = df_unique['sellable_online'].value_counts()

    #  ИСПРАВЛЕНИЕ: защита от случая, когда одна из групп пустая
    if len(online_count) == 1:
        existing_label = online_count.index[0]
        if existing_label:
            online_count = pd.Series([0, online_count.iloc[0]],
                                     index=[False, True])
        else:
            online_count = pd.Series([online_count.iloc[0], 0],
                                     index=[False, True])

    online_count = online_count.rename(index={
        True: 'Продаётся онлайн',
        False: 'НЕ продаётся онлайн'
    })

    bars = ax.bar(online_count.index, online_count.values,
                  color=['forestgreen', 'darkred'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Количество товаров', fontsize=11)
    ax.set_title('Количество товаров онлайн/офлайн',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, count in zip(bars, online_count.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{count}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'eda_online_count')
    plt.close()
    print("  ✓ Сохранено: eda_online_count.png и .svg")

    # ========================================================================
    # 9. Медианная цена онлайн/офлайн
    # ========================================================================
    print("\n 9. Медианная цена онлайн/офлайн")

    fig, ax = plt.subplots(figsize=(8, 6))

    median_by_online = df_unique.groupby('sellable_online')['price'].median()

    # ИСПРАВЛЕНИЕ: защита от случая, когда одна из групп пустая
    if len(median_by_online) == 1:
        existing_label = median_by_online.index[0]
        if existing_label:
            median_by_online = pd.Series([0.0, median_by_online.iloc[0]],
                                         index=[False, True])
        else:
            median_by_online = pd.Series([median_by_online.iloc[0], 0.0],
                                         index=[False, True])

    median_by_online = median_by_online.rename(index={
        True: 'Продаётся онлайн',
        False: 'НЕ продаётся онлайн'
    })

    bars = ax.bar(median_by_online.index, median_by_online.values,
                  color=['forestgreen', 'darkred'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Медианная цена (SR)', fontsize=11)
    ax.set_title('Медианная цена товаров онлайн/офлайн',
                 fontweight='bold', fontsize=13, pad=15)

    for bar, price in zip(bars, median_by_online.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{price:.0f} SR', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'eda_online_median_price')
    plt.close()
    print("  ✓ Сохранено: eda_online_median_price.png и .svg")

    # ========================================================================
    # 10. Распределение цены (гистограмма)
    # ========================================================================
    print("\n 10. Распределение цены")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df_unique['price'], bins=50, color='steelblue',
            edgecolor='black', linewidth=0.5, alpha=0.7)

    ax.axvline(df_unique['price'].median(), color='red', linestyle='--',
               linewidth=2, label=f'Медиана: {df_unique["price"].median():.0f} SR')
    ax.axvline(df_unique['price'].mean(), color='orange', linestyle='--',
               linewidth=2, label=f'Среднее: {df_unique["price"].mean():.0f} SR')

    ax.set_xlabel('Цена (SR)', fontsize=11)
    ax.set_ylabel('Количество товаров', fontsize=11)
    ax.set_title('Распределение цены товаров IKEA',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend()

    plt.tight_layout()
    save_plot(fig, 'eda_price_distribution')
    plt.close()
    print("  ✓ Сохранено: eda_price_distribution.png и .svg")

    # ========================================================================
    # 11. Распределение логарифма цены
    # ========================================================================
    print("\n 11. Распределение логарифма цены")

    fig, ax = plt.subplots(figsize=(10, 6))

    #  ИСПРАВЛЕНИЕ: используем log1p для защиты от log(0)
    price_log = np.log1p(df_unique['price'])

    ax.hist(price_log, bins=50, color='darkgreen',
            edgecolor='black', linewidth=0.5, alpha=0.7)

    ax.axvline(price_log.median(), color='red', linestyle='--',
               linewidth=2, label=f'Медиана log: {price_log.median():.2f}')

    ax.set_xlabel('Логарифм цены (log1p)', fontsize=11)
    ax.set_ylabel('Количество товаров', fontsize=11)
    ax.set_title('Распределение логарифма цены (нормализованное)',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend()

    plt.tight_layout()
    save_plot(fig, 'eda_price_log_distribution')
    plt.close()
    print("  ✓ Сохранено: eda_price_log_distribution.png и .svg")

    # ========================================================================
    # 12. Старая/новая цена с линией регрессии
    # ========================================================================
    print("\n 12. Старая/новая цена с линией регрессии")

    fig, ax = plt.subplots(figsize=(10, 8))

    #  ИСПРАВЛЕНИЕ: более надёжный парсинг old_price
    def parse_old_price(x: Any) -> float:
        if pd.isna(x) or str(x).strip() == '' or str(x).lower() == 'nan':
            return np.nan
        try:
            # Убираем пробелы, "SR", запятые
            clean = str(x).replace(' ', '').replace('SR', '').replace(',', '')
            # Может быть "1000-2000" или "1000"
            if '-' in clean:
                parts = clean.split('-')
                return float(parts[0])
            return float(clean)
        except (ValueError, TypeError):
            return np.nan

    df_old_price = df_unique.copy()
    df_old_price['old_price_parsed'] = df_old_price['old_price'].apply(parse_old_price)

    # Фильтруем только те, у которых есть old_price
    df_valid = df_old_price.dropna(subset=['old_price_parsed', 'price'])

    if len(df_valid) > 10:  # Нужно минимум 10 точек для регрессии
        ax.scatter(df_valid['old_price_parsed'], df_valid['price'],
                   alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.3)

        # Линия регрессии

        slope, intercept, r_value, p_value, std_err = linregress(
            df_valid['old_price_parsed'], df_valid['price']
        )

        x_line = np.linspace(df_valid['old_price_parsed'].min(),
                             df_valid['old_price_parsed'].max(), 100)
        y_line = slope * x_line + intercept

        ax.plot(x_line, y_line, color='red', linewidth=2,
                label=f'Регрессия: y = {slope:.2f}x + {intercept:.0f}\nR² = {r_value ** 2:.3f}')

        # Линия идеального совпадения
        max_val = max(df_valid['old_price_parsed'].max(), df_valid['price'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1,
                label='Идеальное совпадение (y = x)')

        ax.set_xlabel('Старая цена (SR)', fontsize=11)
        ax.set_ylabel('Новая цена (SR)', fontsize=11)
        ax.set_title('Соотношение старой и новой цены',
                     fontweight='bold', fontsize=13, pad=15)
        ax.legend()
    else:
        ax.text(0.5, 0.5,
                f'Недостаточно данных для построения графика\n(найдено {len(df_valid)} товаров со старой ценой)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Соотношение старой и новой цены (недостаточно данных)',
                     fontweight='bold', fontsize=13, pad=15)

    plt.tight_layout()
    save_plot(fig, 'eda_old_vs_new_price')
    plt.close()
    print("  ✓ Сохранено: eda_old_vs_new_price.png и .svg")

    print("\n" + "=" * 70)
    print(" ВСЕ 12 ОБЯЗАТЕЛЬНЫХ ВИЗУАЛИЗАЦИЙ СОЗДАНЫ!")
    print("=" * 70)

#======================================================================================
# писк гиперпараметров (GridSearchCV) для модели RandomForestRegressor
#=====================================================================================

def run_eda_pipeline(df_full: pd.DataFrame, df_unique: pd.DataFrame) -> Dict[str, Any]:
    """
  Запускает сквозной пайплайн исследовательского анализа данных (EDA).

    Пайплайн выполняет три ключевые задачи:
    1. Генерирует обязательный набор визуализаций для анализа распределений и выбросов.
    2. Проверяет дисбаланс целевого признака `sellable_online` и выносит вердикт
       о целесообразности его сохранения или удаления из обучающей выборки.
    3. Выполняет статистический тест на выявление утечки данных (Data Leakage)
       через признак `old_price` и определяет стратегию его использования.

    Args:
        df_full: Полный исходный датасет IKEA (содержит дубликаты), используемый
            исключительно для корректного построения графиков и визуализаций.
        df_unique: Очищенный датасет без дубликатов, используемый для расчёта
            статистик, проверки гипотез и валидации признаков.

    Returns:
        Словарь с результатами анализа следующей структуры:
            {
                'sellable_verdict': str,  # Вердикт по признаку ("KEEP" или "REMOVE")
                'p_true': float,          # Доля значений True в признаке sellable_online
                'p_value': float,         # p-value статистического теста Манна-Уитни
                'old_price_check': dict   # Словарь с вердиктом утечки по old_price:
                                          # {
                                          #   'verdict': str ("LEAK" или "OK"),
                                          #   'features_to_use': list[str]
                                          # }
            }
    """
    print("\n" + "=" * 70)
    print("СОЗДАНИЕ ОБЯЗАТЕЛЬНЫХ EDA-ВИЗУАЛИЗАЦИЙ")
    print("=" * 70)

    # 1. Обязательные визуализации
    create_missing_visualizations(df_full)

    # 2. Проверка sellable_online
    print("\n" + "=" * 70)
    print("ПРОВЕРКА РАСПРЕДЕЛЕНИЯ sellable_online")
    print("=" * 70)
    sellable_verdict, p_true, p_value = check_sellable_online(df_unique)

    # 3. Проверка old_price (детекция утечки)
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ПРИРОДЫ old_price")
    print("=" * 70)
    old_price_check = check_old_price_leakage(df_unique)

    print(f"\n📌 Вердикт по sellable_online: {sellable_verdict}")
    print(f"📌 Вердикт по old_price: {old_price_check['verdict']}")
    print(f"📌 Признаки к использованию: {old_price_check['features_to_use']}")

    return {
        'sellable_verdict': sellable_verdict,
        'p_true': p_true,
        'p_value': p_value,
        'old_price_check': old_price_check
    }

# ==============================================================================
# ОРКЕСТРАТОР СТАТИСТИЧЕСКИХ ГИПОТЕЗ
# ==============================================================================
