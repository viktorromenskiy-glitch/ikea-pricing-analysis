"""
ikea_hypotheses.py — часть модульного разбиения Step_project_Q14.
"""
from ikea_core import *


def check_hypothesis_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет статистическую гипотезу о наличии влияния физического объёма товара на его цену.

    Гипотеза базируется на предположении, что крупногабаритные товары (с большим объёмом)
    стоят дороже мелкогабаритных из-за более высокой материалоемкости и логистических затрат.

    Математическая и статистическая методология исследования:
    1. **Расчет объёма**: Объём товара ($V$) рассчитывается как произведение линейных
       размеров: $V = \\text{width} \\times \\text{height} \\times \\text{depth}$.
    2. **Логарифмирование целевого признака**: Для стабилизации дисперсии и приближения
       распределения цен к нормальному виду рассчитывается натуральный логарифм цены:
       $\\ln(\\text{price})$.
    3. **Бинаризация по медиане**: Набор данных делится на две равные группы («Большой объём»
       и «Малый объём») относительно медианного значения рассчитанного объёма.
    4. **Тест 1 (Непараметрическая корреляция)**: Рассчитывается коэффициент ранговой
       корреляции Спирмена ($\\rho$) между объёмом и исходной ценой для оценки монотонной
       взаимосвязи без жестких требований к нормальности распределения.
    5. **Тест 2 (Параметрическое сравнение средних)**: Проводится независимый двухвыборочный
       t-критерий Стьюдента (`stats.ttest_ind`) над логарифмированной ценой $\\ln(\\text{price})$
       для сравнения средних показателей двух групп.
    6. **Тест 3 (Непараметрический бутстрап)**: Методом Монте-Карло генерируется 10 000
       бутстрап-выборок для каждой группы. Строится эмпирическое распределение разности
       медиан, вычисляется 95%-й доверительный интервал (перцентильным методом: 2.5-й
       и 97.5-й перцентили) и оценивается эмпирическое значение $p$-value.

    Критерий принятия гипотезы:
        Гипотеза признается подтвержденной, если как минимум в 2 из 3 вышеописанных тестов
        достигнута статистическая значимость на уровне $p < 0.05$ (и доверительный интервал
        бутстрапа не включает в себя 0).

    Визуализации (экспортируются через вспомогательную функцию `save_plot`):
        1. `hypothesis_1_bootstrap_distribution` (.png, .svg): Гистограмма распределения
           бутстрап-разностей медиан с отображением доверительного интервала и наблюдаемой разности.
        2. `hypothesis_1_volume_vs_price` (.png, .svg): Матрица графиков 2х2:
           - [0,0]: Диаграмма рассеяния цены от объёма с линией линейной регрессии.
           - [0,1]: Диаграмма размаха («ящик с усами») распределения цен в двух группах объёма.
           - [1,0]: Гистограмма распределения объёмов товаров с границей медианного разделения.
           - [1,1]: Диаграмма рассеяния логарифма цены от объёма с регрессионной прямой.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий числовые столбцы `width`,
            `height`, `depth` и `price`.

    Returns:
        pd.DataFrame: Очищенная от пропусков копия датафрейма (`df_test`), дополненная
            рассчитанными признаками `volume`, `log_price` и `is_large_volume` для
            исследуемых товаров.

    Raises:
        KeyError: Если в переданном датафрейме отсутствует любая из колонок:
            `width`, `height`, `depth` или `price`.
        NameError: Если в контексте не импортированы модули `numpy` (как `np`),
            `scipy.stats` (как `stats`), `matplotlib.pyplot` (как `plt`) или утилита `save_plot`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ 1: ВЛИЯНИЕ ОБЪЁМА ТОВАРА НА ЦЕНУ")
    print("=" * 70)

    df_test = df.copy()
    df_test['volume'] = df_test['width'] * df_test['height'] * df_test['depth']
    df_test['log_price'] = np.log(df_test['price'])

    df_test = df_test.dropna(subset=['volume', 'price']).copy()

    print(f"\nКоличество товаров для анализа: {len(df_test)}")

    median_volume = df_test['volume'].median()
    df_test['is_large_volume'] = df_test['volume'] > median_volume

    print(f"Медианный объём: {median_volume:.0f} см³")
    print(f"Товаров с большим объёмом: {df_test['is_large_volume'].sum()}")
    print(f"Товаров с малым объёмом: {(~df_test['is_large_volume']).sum()}")

    print(f"\nОписательная статистика:")
    print(f"  • Товары с большим объёмом:")
    print(f"    - Медианная цена: {df_test[df_test['is_large_volume']]['price'].median():.2f}")
    print(f"  • Товары с малым объёмом:")
    print(f"    - Медианная цена: {df_test[~df_test['is_large_volume']]['price'].median():.2f}")

    # ТЕСТ 1: Корреляция Спирмена
    spearman_corr, spearman_pvalue = stats.spearmanr(df_test['volume'], df_test['price'])
    print(f"\nТЕСТ 1: КОРРЕЛЯЦИЯ СПИРМЕНА")
    print(f"  • ρ: {spearman_corr:.3f}, p-value: {spearman_pvalue:.6f}")

    # ТЕСТ 2: T-test
    large_log = df_test[df_test['is_large_volume']]['log_price']
    small_log = df_test[~df_test['is_large_volume']]['log_price']
    t_stat, t_pvalue = stats.ttest_ind(large_log, small_log)
    print(f"\nТЕСТ 2: T-TEST")
    print(f"  • t: {t_stat:.3f}, p-value: {t_pvalue:.6f}")

    # ТЕСТ 3: Bootstrap
    np.random.seed(42)
    n_iterations = 10000
    bootstrap_diffs = []

    large_arr = df_test[df_test['is_large_volume']]['price'].values
    small_arr = df_test[~df_test['is_large_volume']]['price'].values

    for _ in range(n_iterations):
        large_sample = np.random.choice(large_arr, size=len(large_arr), replace=True)
        small_sample = np.random.choice(small_arr, size=len(small_arr), replace=True)
        bootstrap_diffs.append(np.median(large_sample) - np.median(small_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = np.mean(bootstrap_diffs <= 0)

    print(f"\nТЕСТ 3: BOOTSTRAP")
    print(f"  • Разность медиан: {np.median(large_arr) - np.median(small_arr):.2f}")
    print(f"  • 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение разности медиан
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(large_arr) - np.median(small_arr)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} SR')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (SR)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title('Бутстрап-распределение разности медиан\n'
                    'Большой объём vs Малый объём',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_1_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_1_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.scatter(df_test['volume'], df_test['price'], alpha=0.3, s=10)
    z = np.polyfit(df_test['volume'], df_test['price'], 1)
    p = np.poly1d(z)
    ax1.plot(df_test['volume'], p(df_test['volume']), "r--", linewidth=2, label='Регрессия')
    ax1.set_xlabel('Объём (см³)')
    ax1.set_ylabel('Цена')
    ax1.set_title(f'Зависимость цены от объёма (ρ={spearman_corr:.3f})', fontweight='bold')
    ax1.legend()

    ax2 = axes[0, 1]
    data_to_plot = [
        df_test[~df_test['is_large_volume']]['price'],
        df_test[df_test['is_large_volume']]['price']
    ]
    bp = ax2.boxplot(data_to_plot, tick_labels=['Малый объём', 'Большой объём'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Цена')
    ax2.set_title('Распределение цен по объёму', fontweight='bold')

    ax3 = axes[1, 0]
    ax3.hist(df_test['volume'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(median_volume, color='red', linestyle='--', linewidth=2, label=f'Медиана: {median_volume:.0f}')
    ax3.set_xlabel('Объём (см³)')
    ax3.set_ylabel('Частота')
    ax3.set_title('Распределение объёма товаров', fontweight='bold')
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.scatter(df_test['volume'], df_test['log_price'], alpha=0.3, s=10)
    z_log = np.polyfit(df_test['volume'], df_test['log_price'], 1)
    p_log = np.poly1d(z_log)
    ax4.plot(df_test['volume'], p_log(df_test['volume']), "r--", linewidth=2)
    ax4.set_xlabel('Объём (см³)')
    ax4.set_ylabel('log(Цена)')
    ax4.set_title('Зависимость log(цены) от объёма', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'hypothesis_1_volume_vs_price')

    # ИТОГ
    tests_passed = sum([
        spearman_pvalue < 0.05,
        t_pvalue < 0.05,
        ci_lower > 0 and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ 1: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

    return df_test

# =============================================================================
# ЧАСТЬ 9: ГИПОТЕЗА 2 - КОМАНДЫ ДИЗАЙНЕРОВ
# =============================================================================

def check_hypothesis_2(df: pd.DataFrame) -> None:
    r"""
    Проверяет статистическую гипотезу о влиянии командной работы дизайнеров на цену товара.

    Гипотеза базируется на предположении, что товары, разработанные командами дизайнеров
    (коллаборациями), имеют более высокую стоимость по сравнению с товарами от одиночных
    авторов за счет синергетического эффекта, сложности согласования или маркетинговой
    ценности соавторства. Из анализа исключаются записи с неизвестным автором (`Unknown`).

    Математическая и статистическая методология исследования:
    1. **Фильтрация и трансформация**: Из выборки удаляются строки, где `designer_clean`
       равен 'Unknown'. Для стабилизации дисперсии цен рассчитывается их натуральный
       логарифм: $\ln(\text{price})$.
    2. **Тест 1 (Критерий Манна-Уитни)**: Проводится односторонний непараметрический
       тест Манна-Уитни (`stats.mannwhitneyu` с альтернативой `greater`) для проверки
       гипотезы о том, что распределение цен товаров от команд сдвинуто вправо (в сторону
       больших значений) относительно одиночных авторов.
    3. **Тест 2 (Параметрическое сравнение средних)**: Проводится независимый двухвыборочный
       t-критерий Стьюдента (`stats.ttest_ind`) над логарифмированными ценами для сравнения
       средних значений двух независимых групп.
    4. **Тест 3 (Непараметрический бутстрап)**: Методом Монте-Карло генерируется 10 000
       бутстрап-выборок для обеих групп. Строится эмпирическое распределение разности
       медиан, вычисляется 95%-й доверительный интервал (перцентильным методом: 2.5-й
       и 97.5-й перцентили) и оценивается эмпирическое значение $p$-value.

    Критерий принятия гипотезы:
        Гипотеза признается подтвержденной, если как минимум в 2 из 3 вышеописанных тестов
        достигнута статистическая значимость на уровне $p < 0.05$ (и доверительный интервал
        бутстрапа не включает в себя 0).

    Визуализации (экспортируются через вспомогательную функцию `save_plot`):
        1. `hypothesis_2_bootstrap_distribution` (.png, .svg): Гистограмма распределения
           бутстрап-разностей медиан с отображением доверительного интервала и наблюдаемой разности.
        2. `hypothesis_2_team_vs_single` (.png, .svg): Матрица графиков 2х2:
           - [0,0]: Диаграмма размаха («ящик с усами») распределения исходных цен.
           - [0,1]: Диаграмма размаха распределения логарифмированных цен $\ln(\text{price})$.
           - [1,0]: Столбчатая диаграмма, наглядно сравнивающая медианные цены двух групп.
           - [1,1]: Совмещенная гистограмма распределения частот исходных цен для обеих групп.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий:
            - Текстовый признак автора: `designer_clean`.
            - Бинарный флаг командной работы: `is_team` (1 или 0).
            - Числовой целевой признак: `price`.

    Returns:
        None: Функция выводит результаты статистических тестов в консоль и сохраняет
            графики на диск.

    Raises:
        KeyError: Если в переданном датафрейме отсутствует любая из колонок:
            `designer_clean`, `is_team` или `price`.
        NameError: Если в контексте не импортированы модули `numpy` (как `np`),
            `scipy.stats` (как `stats`), `matplotlib.pyplot` (как `plt`) или утилита `save_plot`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ 2: ВЛИЯНИЕ КОМАНДНОЙ РАБОТЫ ДИЗАЙНЕРОВ")
    print("=" * 70)

    df_test = df.copy()
    df_test['log_price'] = np.log(df_test['price'])
    df_test = df_test[df_test['designer_clean'] != 'Unknown'].copy()

    team_prices = df_test[df_test['is_team'] == 1]['price']
    single_prices = df_test[df_test['is_team'] == 0]['price']

    print(f"\nТоваров от команд: {len(team_prices)}")
    print(f"Товаров от одиночных: {len(single_prices)}")
    print(f"\nМедианная цена команд: {team_prices.median():.2f}")
    print(f"Медианная цена одиночных: {single_prices.median():.2f}")

    # ТЕСТ 1: Mann-Whitney
    u_stat, u_pvalue = stats.mannwhitneyu(team_prices, single_prices, alternative='greater')
    print(f"\nТЕСТ 1: MANN-WHITNEY U")
    print(f"  • U: {u_stat:.0f}, p-value: {u_pvalue:.6f}")

    # ТЕСТ 2: T-test
    team_log = df_test[df_test['is_team'] == 1]['log_price']
    single_log = df_test[df_test['is_team'] == 0]['log_price']
    t_stat, t_pvalue = stats.ttest_ind(team_log, single_log)
    print(f"\nТЕСТ 2: T-TEST")
    print(f"  • t: {t_stat:.3f}, p-value: {t_pvalue:.6f}")

    # ТЕСТ 3: Bootstrap
    np.random.seed(42)
    bootstrap_diffs = []
    team_arr = team_prices.values
    single_arr = single_prices.values

    for _ in range(10000):
        team_sample = np.random.choice(team_arr, size=len(team_arr), replace=True)
        single_sample = np.random.choice(single_arr, size=len(single_arr), replace=True)
        bootstrap_diffs.append(np.median(team_sample) - np.median(single_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = np.mean(bootstrap_diffs <= 0)

    print(f"\nТЕСТ 3: BOOTSTRAP")
    print(f"  • Разность медиан: {np.median(team_arr) - np.median(single_arr):.2f}")
    print(f"  • 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение разности медиан
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(team_arr) - np.median(single_arr)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} SR')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (SR)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title('Бутстрап-распределение разности медиан\n'
                    'Команды дизайнеров vs Одиночные',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_2_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_2_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    bp = ax1.boxplot([single_prices, team_prices],
                     tick_labels=['Одиночные', 'Команды'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Цена')
    ax1.set_title('Распределение цен: команды vs одиночные', fontweight='bold')

    ax2 = axes[0, 1]
    bp2 = ax2.boxplot([np.log(single_prices), np.log(team_prices)],
                      tick_labels=['Одиночные', 'Команды'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('log(Цена)')
    ax2.set_title('Распределение log(цен)', fontweight='bold')

    ax3 = axes[1, 0]
    medians = [single_prices.median(), team_prices.median()]
    bars = ax3.bar(['Одиночные', 'Команды'], medians,
                   color=['lightgreen', 'lightcoral'], edgecolor='black')
    ax3.set_ylabel('Медианная цена')
    ax3.set_title('Сравнение медианных цен', fontweight='bold')
    for bar, median in zip(bars, medians):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{median:.0f}', ha='center', va='bottom', fontweight='bold')

    ax4 = axes[1, 1]
    ax4.hist(single_prices, bins=50, alpha=0.6, label='Одиночные', color='lightgreen')
    ax4.hist(team_prices, bins=50, alpha=0.6, label='Команды', color='lightcoral')
    ax4.set_xlabel('Цена')
    ax4.set_ylabel('Частота')
    ax4.set_title('Распределение цен', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    save_plot(fig, 'hypothesis_2_team_vs_single')

    tests_passed = sum([
        u_pvalue < 0.05,
        t_pvalue < 0.05,
        ci_lower > 0 and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ 2: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

# =============================================================================
# ЧАСТЬ 10: ГИПОТЕЗА 3 - КАТЕГОРИИ
# =============================================================================

def check_hypothesis_3(df: pd.DataFrame) -> None:
    r"""
   Проверяет статистическую гипотезу о наличии ценовой дифференциации между категориями товаров.

    Гипотеза базируется на предположении, что различные товарные категории в IKEA (например,
    'Beds' по сравнению с 'Mirrors') имеют принципиально разные распределения цен из-за
    различий в габаритах, сложности производства и используемых материалах.

    Математическая и статистическая методология исследования:
    1. **Первичный групповой анализ**: Рассчитываются описательные статистики (медиана,
       среднее значение, объем выборки) для каждой уникальной категории товаров.
    2. **Тест 1 (Критерий Краскела-Уоллиса)**: Проводится непараметрический дисперсионный
       анализ (`stats.kruskal`) для проверки общей нулевой гипотезы о равенстве медиан
       цен во всех категориях одновременно.
    3. **Тест 2 (Попарный Post-hoc анализ)**: Для всех возможных пар категорий рассчитывается
       двусторонний критерий Манна-Уитни (`stats.mannwhitneyu`). Для минимизации вероятности
       ошибки первого рода ($I$ типа) из-за множественных сравнений (эффект вылавливания значимости)
       применяется строгая **поправка Бонферрони**: уровень значимости корректируется как
       $\\alpha_{\\text{corrected}} = 0.05 / N$, где $N$ — общее число парных сравнений.
    4. **Тест 3 (Непараметрический бутстрап экстремальной пары)**: Сравниваются самая дорогая
       и самая дешевая категории по медианной цене. Методом Монте-Карло генерируется 10 000
       бутстрап-выборок для обеих групп, строится доверительный интервал разности медиан
       (перцентильным методом на уровне 95%) и рассчитывается эмпирическое значение $p$-value.

    Критерий принятия гипотезы:
        Гипотеза признается подтвержденной, если как минимум в 2 из 3 вышеописанных тестов
        достигнута статистическая значимость на уровне $p < 0.05$ (для post-hoc — с учетом
        поправки Бонферрони, а для бутстрапа — если доверительный интервал не включает в себя 0).

    Визуализации (экспортируются через вспомогательную функцию `save_plot`):
        1. `hypothesis_3_bootstrap_distribution` (.png, .svg): Гистограмма распределения
           бутстрап-разностей медиан между самой дорогой и самой дешевой категориями с
           отображением доверительного интервала и наблюдаемой разности.
        2. `hypothesis_3_categories` (.png, .svg): Сложная матрица графиков 2х2:
           - [0,0]: Горизонтальный столбчатый график отсортированных медианных цен по категориям.
           - [0,1]: Диаграмма размаха («ящик с усами») цен для топ-5 самых дорогих и топ-5 дешёвых категорий.
           - [1,0]: Диаграмма рассеяния взаимосвязи между объемом предложения категории (count) и её медианной ценой.
           - [1,1]: Тепловая карта $-\log_{10}(p\text{-value})$ попарных тестов Манна-Уитни для топ-10 категорий.
             *Выполнено требование ментора*: на ячейки тепловой карты нанесены контрастные числовые метки
             значений логарифма p-value для упрощения интерпретации результатов.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий категориальный признак `category`
            и числовой целевой признак `price`.

    Returns:
        None: Функция выводит результаты статистических тестов в консоль и сохраняет
            графики на диск.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные
            колонки `category` или `price`.
        NameError: Если в контексте не импортированы модули `numpy` (как `np`),
            `pandas` (как `pd`), `scipy.stats` (как `stats`), `matplotlib.pyplot`
            (как `plt`) или утилита `save_plot`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ 3: ЦЕНОВАЯ ДИФФЕРЕНЦИАЦИЯ ПО КАТЕГОРИЯМ")
    print("=" * 70)

    df_test = df.copy()
    category_stats = df_test.groupby('category')['price'].agg(['median', 'mean', 'count'])
    category_stats = category_stats.sort_values('median', ascending=False)

    print(f"\nСтатистика по категориям:")
    print(category_stats.to_markdown())

    # ========================================================================
    # ТЕСТ 1: Kruskal-Wallis
    # ========================================================================
    groups = [group['price'].values for name, group in df_test.groupby('category')]
    kw_stat, kw_pvalue = stats.kruskal(*groups)

    print(f"\nТЕСТ 1: KRUSKAL-WALLIS")
    print(f"  • H: {kw_stat:.3f}, p-value: {kw_pvalue:.6f}")

    # ========================================================================
    # ТЕСТ 2: Post-hoc Mann-Whitney
    # ========================================================================
    categories = df_test['category'].unique()
    pairwise_results = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            prices1 = df_test[df_test['category'] == categories[i]]['price']
            prices2 = df_test[df_test['category'] == categories[j]]['price']
            if len(prices1) > 0 and len(prices2) > 0:
                _, p_value = stats.mannwhitneyu(prices1, prices2, alternative='two-sided')
                pairwise_results.append({
                    'cat1': categories[i], 'cat2': categories[j],
                    'median1': prices1.median(), 'median2': prices2.median(),
                    'p_value': p_value
                })

    pairwise_df = pd.DataFrame(pairwise_results)
    n_comparisons = len(pairwise_df)
    alpha_corrected = 0.05 / n_comparisons
    pairwise_df['significant'] = pairwise_df['p_value'] < alpha_corrected
    significant_pairs = pairwise_df[pairwise_df['significant']]

    print(f"\nТЕСТ 2: POST-HOC")
    print(f"  • Сравнений: {n_comparisons}")
    print(f"  • Порог Бонферрони: {alpha_corrected:.6f}")
    print(f"  • Значимых различий: {len(significant_pairs)}")

    # ========================================================================
    # ТЕСТ 3: Bootstrap для топ-пары
    # ========================================================================
    most_expensive = category_stats.index[0]
    cheapest = category_stats.index[-1]
    prices_exp = df_test[df_test['category'] == most_expensive]['price'].values
    prices_che = df_test[df_test['category'] == cheapest]['price'].values

    np.random.seed(42)
    bootstrap_diffs = []
    for _ in range(10000):
        sample_exp = np.random.choice(prices_exp, size=len(prices_exp), replace=True)
        sample_che = np.random.choice(prices_che, size=len(prices_che), replace=True)
        bootstrap_diffs.append(np.median(sample_exp) - np.median(sample_che))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = np.mean(bootstrap_diffs <= 0)

    print(f"\nТЕСТ 3: BOOTSTRAP ({most_expensive} vs {cheapest})")
    print(f"  • Разность медиан: {np.median(prices_exp) - np.median(prices_che):.2f}")
    print(f"  • 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # 🆕 ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(prices_exp) - np.median(prices_che)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} SR')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (SR)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title(f'Бутстрап-распределение разности медиан\n'
                    f'{most_expensive} vs {cheapest}',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_3_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_3_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Медианные цены по категориям
    ax1 = axes[0, 0]
    cats_sorted = category_stats.sort_values('median', ascending=True)
    ax1.barh(range(len(cats_sorted)), cats_sorted['median'], color='skyblue', edgecolor='black')
    ax1.set_yticks(range(len(cats_sorted)))
    ax1.set_yticklabels(cats_sorted.index, fontsize=9)
    ax1.set_xlabel('Медианная цена')
    ax1.set_title('Медианные цены по категориям', fontweight='bold')

    # График 2: Boxplot топ-5 дорогих vs топ-5 дешёвых
    ax2 = axes[0, 1]
    top_cats = list(category_stats.head(5).index) + list(category_stats.tail(5).index)
    data_to_plot = [df_test[df_test['category'] == cat]['price'].values for cat in top_cats]
    bp = ax2.boxplot(data_to_plot, tick_labels=top_cats, patch_artist=True)
    colors = ['lightcoral'] * 5 + ['lightblue'] * 5
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Цена')
    ax2.set_title('Топ-5 дорогих vs топ-5 дешёвых', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # График 3: Scatter количество vs медианная цена
    ax3 = axes[1, 0]
    ax3.scatter(category_stats['count'], category_stats['median'], s=100, alpha=0.7, c='steelblue')
    for cat in category_stats.index:
        ax3.annotate(cat[:20], (category_stats.loc[cat, 'count'], category_stats.loc[cat, 'median']),
                     fontsize=7, alpha=0.7, xytext=(5, 5), textcoords="offset points")
    ax3.set_xlabel('Количество товаров')
    ax3.set_ylabel('Медианная цена')
    ax3.set_title('Количество vs медианная цена', fontweight='bold')

    # График 4: Матрица -log10(p-value) с числовыми метками
    ax4 = axes[1, 1]
    top_10 = category_stats.head(10).index
    p_matrix = np.zeros((len(top_10), len(top_10)))
    for i, cat1 in enumerate(top_10):
        for j, cat2 in enumerate(top_10):
            if i != j:
                p1 = df_test[df_test['category'] == cat1]['price']
                p2 = df_test[df_test['category'] == cat2]['price']
                _, p_val = stats.mannwhitneyu(p1, p2, alternative='two-sided')
                p_matrix[i, j] = -np.log10(p_val + 1e-10)

    im = ax4.imshow(p_matrix, cmap='YlOrRd', aspect='auto')

    # 🆕 ИСПРАВЛЕНИЕ: Добавляем числовые метки на ячейки матрицы
    for i in range(len(top_10)):
        for j in range(len(top_10)):
            if i != j:
                # Определяем цвет текста в зависимости от фона
                value = p_matrix[i, j]
                text_color = 'white' if value > 5 else 'black'
                ax4.text(j, i, f'{value:.1f}',
                         ha='center', va='center', fontsize=7,
                         color=text_color, fontweight='bold')

    ax4.set_xticks(range(len(top_10)))
    ax4.set_yticks(range(len(top_10)))
    ax4.set_xticklabels([c[:15] for c in top_10], rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels([c[:15] for c in top_10], fontsize=8)
    ax4.set_title('-log10(p-value) попарных сравнений', fontweight='bold')
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    save_plot(fig, 'hypothesis_3_categories')

    # ========================================================================
    # ИТОГ
    # ========================================================================
    tests_passed = sum([
        kw_pvalue < 0.05,
        len(significant_pairs) > 0,
        ci_lower > 0 and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ 3: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

# =============================================================================
# ЧАСТЬ 10-Б : Создание Boxplot  рапределения по категориям
# =============================================================================

def plot_price_distribution_by_category(df: pd.DataFrame) -> None:
    """
    Строит и сохраняет диаграмму размаха (boxplot) распределения цен по ценовым сегментам.

    Функция группирует категории товаров IKEA в три упорядоченных ценовых сегмента
    (Бюджетный, Средний, Премиум) на основе их внутренних медианных цен. На основе
    этой классификации строится график распределения цен, исправляющий типичные
    ошибки автоматического сопоставления цветов и порядка категорий.

    Алгоритм работы и ключевые исправления:
    1. **Сегментация по медианам**: Каждой категории присваивается сегмент на основе порога:
       - 'Бюджетный': медиана категории < 400 SR
       - 'Средний': 400 SR <= медиана категории < 1000 SR
       - 'Премиум': медиана категории >= 1000 SR
    2. **Строгое упорядочивание (`segment_order`)**: Переменная сегмента принудительно
       приводится к типу `pd.Categorical` с флагом `ordered=True`. Это гарантирует,
       что на оси абсцисс сегменты всегда будут следовать слева направо: Бюджетный → Средний → Премиум.
    3. **Контролируемая палитра (`segment_colors`)**: Цвета (зеленый, оранжевый, красный)
       жестко привязаны к конкретным именам сегментов через словарь, исключая случайную
       перетасовку цветов библиотекой seaborn при изменении состава данных.
    4. **Оптимизация отображения**:
       - Для улучшения читаемости скрываются экстремальные выбросы (`showfliers=False`).
       - Значения медиан каждого сегмента наносятся текстом прямо над соответствующими «ящиками».
       - Верхний лимит оси Y динамически ограничивается 95-м перцентилем цены плюс отступ,
         что защищает текстовые метки от обрезания рамкой графика.

    График экспортируется в растровом (.png) и векторном (.svg) форматах в локальную
    директорию `plots_step`.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий текстовое поле `category`
            и числовое поле `price`.

    Returns:
        None: Функция выводит аналитические логи проверки в консоль и сохраняет
            файлы визуализации на диск.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные
            колонки `category` или `price`.
        NameError: Если в модуле не импортированы библиотеки `pandas` (как `pd`),
            `matplotlib.pyplot` (как `plt`) или `seaborn` (как `sns`).
    """

    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ: Распределение цен по ценовым сегментам")
    print("=" * 70)

    # ========================================================================
    # ШАГ 1: Определяем ценовые сегменты для каждого товара
    # ========================================================================
    # Используем медианные цены по категориям из предыдущего анализа
    # (эти значения должны быть вычислены ранее в скрипте)

    # Создаём словарь медианных цен по категориям
    category_medians = df.groupby('category')['price'].median().to_dict()

    # Определяем границы сегментов на основе медиан
    # Бюджетный: медиана категории < 400 SR
    # Средний: 400 <= медиана категории < 1000 SR
    # Премиум: медиана категории >= 1000 SR

    def assign_price_segment(category: str) -> str:
        median_price = category_medians.get(category, 0)
        if median_price < 400:
            return 'Бюджетный'
        elif median_price < 1000:
            return 'Средний'
        else:
            return 'Премиум'

    df_copy = df.copy()
    df_copy['price_segment_name'] = df_copy['category'].apply(assign_price_segment)

    # ========================================================================
    # ШАГ 2: ЯВНО задаём порядок сегментов
    # ========================================================================
    # 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: используем Categorical с явным порядком
    segment_order = ['Бюджетный', 'Средний', 'Премиум']
    df_copy['price_segment_name'] = pd.Categorical(
        df_copy['price_segment_name'],
        categories=segment_order,
        ordered=True
    )

    # ========================================================================
    # ШАГ 3: Вычисляем медианы для проверки
    # ========================================================================
    segment_medians = df_copy.groupby('price_segment_name', observed=False)['price'].median()

    print("\n Медианные цены по сегментам:")
    for segment in segment_order:
        median_val = segment_medians[segment]
        print(f"  • {segment}: {median_val:.0f} SR")

    # ========================================================================
    # ШАГ 4: ЯВНО задаём цвета для каждого сегмента
    # ========================================================================
    #  КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: словарь цветов по имени сегмента
    segment_colors = {
        'Бюджетный': '#2ecc71',  # Зелёный
        'Средний': '#f39c12',  # Оранжевый/жёлтый
        'Премиум': '#e74c3c'  # Красный/розовый
    }

    # Создаём список цветов в правильном порядке
    colors_ordered = [segment_colors[seg] for seg in segment_order]

    # ========================================================================
    # ШАГ 5: Строим график
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Boxplot с явным порядком категорий
    sns.boxplot(
        data=df_copy,
        x='price_segment_name',
        y='price',
        order=segment_order,  # 🔑 ЯВНЫЙ ПОРЯДОК
        palette=colors_ordered,  # 🔑 ПРАВИЛЬНЫЕ ЦВЕТА
        ax=ax,
        showfliers=False  # Скрываем выбросы для читаемости
    )

    # ========================================================================
    # ШАГ 6: Добавляем подписи медиан на график
    # ========================================================================
    for i, segment in enumerate(segment_order):
        median_val = segment_medians[segment]
        ax.text(
            i, median_val + 100,  # Позиция текста
            f'{median_val:.0f} SR',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=11,
            color=segment_colors[segment]
        )

    # ========================================================================
    # ШАГ 7: Настройка оформления
    # ========================================================================
    ax.set_xlabel('Ценовой сегмент', fontsize=12, fontweight='bold')
    ax.set_ylabel('Цена (SR)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Распределение цен по ценовым сегментам IKEA\n'
        '(Бюджетный → Средний → Премиум)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Добавляем сетку для удобства чтения
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Увеличиваем лимиты по Y, чтобы подписи медиан были видны
    ax.set_ylim(0, df_copy['price'].quantile(0.95) + 300)

    plt.tight_layout()

    # ========================================================================
    # ШАГ 8: Сохранение графика
    # ========================================================================
    # Создаём папку plots_step, если её нет
    os.makedirs('plots_step', exist_ok=True)

    # Сохраняем в PNG
    png_path = 'plots_step/price_distribution_by_category_boxplot.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Сохранено: {png_path}")

    # Сохраняем в SVG
    svg_path = 'plots_step/price_distribution_by_category_boxplot.svg'
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Сохранено: {svg_path}")

    plt.close()

    # ========================================================================
    # ШАГ 9: Проверка корректности
    # ========================================================================
    print("\n✅ ПРОВЕРКА КОРРЕКТНОСТИ ГРАФИКА:")
    print(f"  • Порядок сегментов: {' → '.join(segment_order)}")
    print(f"  • Бюджетный (зелёный): {segment_medians['Бюджетный']:.0f} SR")
    print(f"  • Средний (оранжевый): {segment_medians['Средний']:.0f} SR")
    print(f"  • Премиум (красный): {segment_medians['Премиум']:.0f} SR")

    # Проверяем, что порядок правильный
    if (segment_medians['Бюджетный'] < segment_medians['Средний'] <
            segment_medians['Премиум']):
        print("  ✅ Порядок медиан корректен: Бюджетный < Средний < Премиум")
    else:
        print("  ⚠️ ВНИМАНИЕ: Порядок медиан нарушен!")

    return fig

# =============================================================================
# ЧАСТЬ 11: ГИПОТЕЗА 4 - ЦВЕТОВАЯ ВАРИАТИВНОСТЬ
# =============================================================================

def check_hypothesis_4(df: pd.DataFrame) -> None:
    """
    Проверяет гипотезу 4: Влияние цветовой вариативности на цену.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ 4: ВЛИЯНИЕ ЦВЕТОВОЙ ВАРИАТИВНОСТИ")
    print("=" * 70)

    df_test = df.copy()
    df_test['log_price'] = np.log(df_test['price'])
    df_test['has_other_colors'] = df_test['other_colors'] == 'Yes'

    with_colors = df_test[df_test['has_other_colors']]['price']
    without_colors = df_test[~df_test['has_other_colors']]['price']

    print(f"\nТоваров с другими цветами: {len(with_colors)}")
    print(f"Товаров без других цветов: {len(without_colors)}")
    print(f"\nМедианная цена с другими цветами: {with_colors.median():.2f}")
    print(f"Медианная цена без других цветов: {without_colors.median():.2f}")

    # ТЕСТ 1: Mann-Whitney
    u_stat, u_pvalue = stats.mannwhitneyu(with_colors, without_colors, alternative='two-sided')
    print(f"\nТЕСТ 1: MANN-WHITNEY U")
    print(f"  • U: {u_stat:.0f}, p-value: {u_pvalue:.6f}")

    # ТЕСТ 2: T-test
    with_log = df_test[df_test['has_other_colors']]['log_price']
    without_log = df_test[~df_test['has_other_colors']]['log_price']
    t_stat, t_pvalue = stats.ttest_ind(with_log, without_log)
    print(f"\nТЕСТ 2: T-TEST")
    print(f"  • t: {t_stat:.3f}, p-value: {t_pvalue:.6f}")

    # ТЕСТ 3: Bootstrap
    np.random.seed(42)
    bootstrap_diffs = []
    with_arr = with_colors.values
    without_arr = without_colors.values

    for _ in range(10000):
        with_sample = np.random.choice(with_arr, size=len(with_arr), replace=True)
        without_sample = np.random.choice(without_arr, size=len(without_arr), replace=True)
        bootstrap_diffs.append(np.median(with_sample) - np.median(without_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = 2 * min(np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0))

    print(f"\nТЕСТ 3: BOOTSTRAP")
    print(f"  • Разность медиан: {np.median(with_arr) - np.median(without_arr):.2f}")
    print(f"  • 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение разности медиан
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(with_arr) - np.median(without_arr)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} SR')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (SR)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title('Бутстрап-распределение разности медиан\n'
                    'С другими цветами vs Без других цветов',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_4_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_4_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    bp = ax1.boxplot([without_colors, with_colors],
                     tick_labels=['Без других цветов', 'С другими цветами'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Цена')
    ax1.set_title('Распределение цен', fontweight='bold')

    ax2 = axes[0, 1]
    bp2 = ax2.boxplot([np.log(without_colors), np.log(with_colors)],
                      tick_labels=['Без других', 'С другими'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('log(Цена)')
    ax2.set_title('Распределение log(цен)', fontweight='bold')

    ax3 = axes[1, 0]
    medians = [without_colors.median(), with_colors.median()]
    bars = ax3.bar(['Без других', 'С другими'], medians,
                   color=['lightblue', 'lightcoral'], edgecolor='black')
    ax3.set_ylabel('Медианная цена')
    ax3.set_title('Сравнение медианных цен', fontweight='bold')
    for bar, median in zip(bars, medians):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{median:.0f}', ha='center', va='bottom', fontweight='bold')

    ax4 = axes[1, 1]
    counts = [df_test['has_other_colors'].sum(), (~df_test['has_other_colors']).sum()]
    ax4.pie(counts, labels=['С другими', 'Без других'], autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue'], startangle=90)
    ax4.set_title('Распределение товаров', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'hypothesis_4_colors')

    tests_passed = sum([
        u_pvalue < 0.05,
        t_pvalue < 0.05,
        (ci_lower > 0 or ci_upper < 0) and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ 4: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

# =============================================================================
# ЧАСТЬ 12: ГИПОТЕЗА A - СКИДКИ И ОБЪЁМ
# =============================================================================

def check_hypothesis_A(df: pd.DataFrame) -> None:
    r"""
    Проверяет статистическую гипотезу о том, что товары со скидкой имеют бо́льшие габариты (объём).

    Гипотеза основывается на предположении ритейл-аналитики: крупногабаритные товары
    (например, шкафы, кровати) занимают много места на складах IKEA, поэтому компания
    более склонна предоставлять на них скидки для ускорения оборачиваемости запасов.

    Математическая и статистическая методология исследования:
    1. **Расчёт объёма**: Физический объём товара вычисляется как произведение сторон:
       $$Volume = Width \\times Height \\times Depth$$
       Записи с пропущенными линейными размерами исключаются из анализа.
    2. **Идентификация скидки**: Наличие скидки определяется по условию
       `old_price != 'No old price'`.
    3. **Тест 1 (Критерий Манна-Уитни)**: Проводится односторонний непараметрический
       тест (`stats.mannwhitneyu` с альтернативой `'greater'`) для проверки того,
       что распределение объёмов товаров со скидкой стохастически больше распределения
       объёмов товаров без скидки.
    4. **Тест 2 (Двухвыборочный t-критерий Стьюдента)**: Проводится параметрический
       тест (`stats.ttest_ind`). Поскольку распределение объёмов физических тел
       обладает сильной правосторонней асимметрией, к переменной объёма предварительно
       применяется логарифмическое преобразование ($\ln(\text{volume})$) для приближения
       распределения к нормальному.
    5. **Тест 3 (Непараметрический бутстрап)**: Методом Монте-Карло генерируется 10 000
       бутстрап-выборок для обеих групп. Строится 95%-й доверительный интервал для разности
       медиан. Рассчитывается эмпирическое значение $p$-value как доля случаев, когда разность
       медиан бутстрап-выборок оказалась меньше или равна нулю.

    Критерий принятия гипотезы:
        Гипотеза признается подтвержденной, если как минимум в 2 из 3 вышеописанных тестов
        достигнута статистическая значимость на уровне $p < 0.05$ (а для бутстрапа — если
        95% доверительный интервал разности медиан лежит строго правее нуля).

    Визуализации (экспортируются через вспомогательную функцию `save_plot`):
        1. `hypothesis_A_bootstrap_distribution` (.png, .svg): Гистограмма распределения
           бутстрап-разностей медиан между товарами со скидкой и без неё с отображением
           наблюдаемой разницы, доверительного интервала и линии нулевой гипотезы.
        2. `hypothesis_A_discount_volume` (.png, .svg): Сдвоенный график 1х2:
           - Левый график: Диаграмма размаха («ящик с усами») распределения объёмов в обеих группах.
           - Правый график: Столбчатая диаграмма сравнения точечных медианных значений
             с числовыми текстовыми метками на вершинах столбцов.

    Args:
        df (pd.DataFrame): Исходный датафрейм, обязательно содержащий колонки
            `old_price` (для идентификации скидки), а также `width`, `height` и `depth`
            (для вычисления объёма).

    Returns:
        None: Функция выводит результаты расчётов и тестов в консоль, а также сохраняет
            графики на диск.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные для вычислений
            колонки `old_price`, `width`, `height` или `depth`.
        NameError: Если в контексте выполнения не импортированы модули `numpy` (как `np`),
            `pandas` (как `pd`), `scipy.stats` (как `stats`), `matplotlib.pyplot`
            (как `plt`) или утилита `save_plot`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ A: СКИДКИ И ОБЪЁМ")
    print("=" * 70)

    df_test = df.copy()
    df_test['has_discount'] = df_test['old_price'] != 'No old price'
    df_test['volume'] = df_test['width'] * df_test['height'] * df_test['depth']
    df_test = df_test.dropna(subset=['volume']).copy()

    discount_vol = df_test[df_test['has_discount']]['volume']
    no_discount_vol = df_test[~df_test['has_discount']]['volume']

    print(f"\nТоваров со скидкой: {len(discount_vol)}")
    print(f"Товаров без скидки: {len(no_discount_vol)}")
    print(f"\nМедианный объём со скидкой: {discount_vol.median():.0f} см³")
    print(f"Медианный объём без скидки: {no_discount_vol.median():.0f} см³")

    # ТЕСТ 1: Mann-Whitney
    u_stat, u_pvalue = stats.mannwhitneyu(discount_vol, no_discount_vol, alternative='greater')
    print(f"\nТЕСТ 1: MANN-WHITNEY U")
    print(f"  • U: {u_stat:.0f}, p-value: {u_pvalue:.6f}")

    # ТЕСТ 2: T-test
    t_stat, t_pvalue = stats.ttest_ind(np.log(discount_vol), np.log(no_discount_vol))
    print(f"\nТЕСТ 2: T-TEST")
    print(f"  • t: {t_stat:.3f}, p-value: {t_pvalue:.6f}")

    # ТЕСТ 3: Bootstrap
    np.random.seed(42)
    bootstrap_diffs = []
    discount_arr = discount_vol.values
    no_discount_arr = no_discount_vol.values

    for _ in range(10000):
        d_sample = np.random.choice(discount_arr, size=len(discount_arr), replace=True)
        n_sample = np.random.choice(no_discount_arr, size=len(no_discount_arr), replace=True)
        bootstrap_diffs.append(np.median(d_sample) - np.median(n_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = np.mean(bootstrap_diffs <= 0)

    print(f"\nТЕСТ 3: BOOTSTRAP")
    print(f"  • Разность медиан: {np.median(discount_arr) - np.median(no_discount_arr):.0f} см³")
    print(f"  • 95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение разности медиан
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(discount_arr) - np.median(no_discount_arr)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} см³')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (см³)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title('Бутстрап-распределение разности медиан\n'
                    'Со скидкой vs Без скидки',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_A_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_A_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (1x2)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    bp = ax1.boxplot([no_discount_vol, discount_vol],
                     tick_labels=['Без скидки', 'Со скидкой'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Объём (см³)')
    ax1.set_title('Распределение объёма', fontweight='bold')

    ax2 = axes[1]
    medians = [no_discount_vol.median(), discount_vol.median()]
    bars = ax2.bar(['Без скидки', 'Со скидкой'], medians,
                   color=['lightblue', 'lightcoral'], edgecolor='black')
    ax2.set_ylabel('Медианный объём (см³)')
    ax2.set_title('Сравнение медианных объёмов', fontweight='bold')
    for bar, median in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{median:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'hypothesis_A_discount_volume')

    tests_passed = sum([
        u_pvalue < 0.05,
        t_pvalue < 0.05,
        ci_lower > 0 and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ A: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

# =============================================================================
# ЧАСТЬ 13: ГИПОТЕЗА B - ПРЕМИАЛЬНЫЕ МАТЕРИАЛЫ
# =============================================================================

def check_hypothesis_B(df: pd.DataFrame) -> None:
    """
    "Проверяет статистическую гипотезу о том, что товары из премиальных материалов стоят дороже.

    Гипотеза основывается на предположении, что использование в производстве мебели IKEA
    определенных видов сырья (например, массива дуба, ясеня, натуральной кожи или мрамора)
    значительно увеличивает себестоимость единицы товара и, как следствие, его розничную цену.

    Математическая и статистическая методология исследования:
    1. **Текстовый майнинг и разметка**: Наличие премиальных материалов определяется с помощью
       регулярного выражения по маске `short_description`. Список паттернов включает:
       'oak' (дуб), 'walnut' (орех), 'ash' (ясень), 'birch' (береза), 'leather' (кожа),
       'steel' (сталь), 'brass' (латунь), 'marble' (мрамор), 'glass' (стекло). Регистр строк
       при поиске игнорируется.
    2. **Тест 1 (Критерий Манна-Уитни)**: Односторонний непараметрический тест
       (`stats.mannwhitneyu` с альтернативой `'greater'`). Проверяет гипотезу о том, что
       распределение цен товаров с премиум-компонентами стохастически больше распределения
       цен базовых аналогов.
    3. **Тест 2 (Двухвыборочный t-критерий Стьюдента)**: Параметрический тест (`stats.ttest_ind`).
       Поскольку цены в ритейле имеют сильное смещение вправо (тяжёлые хвосты распределения),
       для стабилизации дисперсии и приведения данных к нормальному виду используется
       логарифмическое преобразование:
       $$LogPrice = \\ln(Price)$$
    4. **Тест 3 (Непараметрический бутстрап)**: Компьютерное моделирование методом Монте-Карло
       (10 000 повторений с возвращением). Строится эмпирическое распределение разности медиан,
       на основе которого рассчитывается двухсторонний 95%-й доверительный интервал (перцентильным
       методом) и точечный $p$-value.

    Критерий принятия гипотезы:
        Гипотеза признается подтвержденной, если как минимум в 2 из 3 вышеописанных тестов
        достигнута статистическая значимость на уровне $p < 0.05$ (для бутстрапа — если
        95%-й доверительный интервал разности медиан лежит строго правее нуля, не включая 0).

    Визуализации (экспортируются через вспомогательную функцию `save_plot`):
        1. `hypothesis_B_bootstrap_distribution` (.png, .svg): Гистограмма плотности бутстрап-разностей
           медиан с наложением линий истинного наблюдаемого сдвига, доверительного интервала
           и нулевой отметки (маркера отсутствия эффекта).
        2. `hypothesis_B_premium_materials` (.png, .svg): Сдвоенная панель графиков 1х2:
           - Левый график: Boxplot распределения исходных цен в группах «С премиум» и «Без премиум»
             (для наглядности масштабов).
           - Правый график: Столбчатая диаграмма точечного сравнения медиан с числовыми
             метками-подписями над каждым баром.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий текстовое поле краткого описания
            `short_description` и непрерывную числовую переменную `price`.

    Returns:
        None: Функция агрегирует логи тестов в терминале и генерирует графические
            артефакты на диске.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные
            колонки `short_description` или `price`.
        NameError: Если в рабочей среде не импортированы модули `numpy` (как `np`),
            `pandas` (как `pd`), `scipy.stats` (как `stats`), `matplotlib.pyplot`
            (как `plt`) или утилита `save_plot`.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ B: ПРЕМИАЛЬНЫЕ МАТЕРИАЛЫ")
    print("=" * 70)

    df_test = df.copy()
    df_test['log_price'] = np.log(df_test['price'])

    premium_materials = ['oak', 'walnut', 'ash', 'birch', 'leather', 'steel', 'brass', 'marble', 'glass']
    premium_pattern = '|'.join(premium_materials)

    df_test['has_premium_material'] = df_test['short_description'].str.contains(
        premium_pattern, case=False, na=False
    )

    premium_prices = df_test[df_test['has_premium_material']]['price']
    no_premium_prices = df_test[~df_test['has_premium_material']]['price']

    print(f"\nТоваров с премиум материалами: {len(premium_prices)}")
    print(f"Товаров без премиум материалов: {len(no_premium_prices)}")
    print(f"\nМедианная цена с премиум: {premium_prices.median():.2f}")
    print(f"Медианная цена без премиум: {no_premium_prices.median():.2f}")

    # ТЕСТ 1: Mann-Whitney
    u_stat, u_pvalue = stats.mannwhitneyu(premium_prices, no_premium_prices, alternative='greater')
    print(f"\nТЕСТ 1: MANN-WHITNEY U")
    print(f"  • U: {u_stat:.0f}, p-value: {u_pvalue:.6f}")

    # ТЕСТ 2: T-test
    premium_log = df_test[df_test['has_premium_material']]['log_price']
    no_premium_log = df_test[~df_test['has_premium_material']]['log_price']
    t_stat, t_pvalue = stats.ttest_ind(premium_log, no_premium_log)
    print(f"\nТЕСТ 2: T-TEST")
    print(f"  • t: {t_stat:.3f}, p-value: {t_pvalue:.6f}")

    # ТЕСТ 3: Bootstrap
    np.random.seed(42)
    bootstrap_diffs = []
    premium_arr = premium_prices.values
    no_premium_arr = no_premium_prices.values

    for _ in range(10000):
        p_sample = np.random.choice(premium_arr, size=len(premium_arr), replace=True)
        n_sample = np.random.choice(no_premium_arr, size=len(no_premium_arr), replace=True)
        bootstrap_diffs.append(np.median(p_sample) - np.median(n_sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    bootstrap_pvalue = np.mean(bootstrap_diffs <= 0)

    print(f"\nТЕСТ 3: BOOTSTRAP")
    print(f"  • Разность медиан: {np.median(premium_arr) - np.median(no_premium_arr):.2f}")
    print(f"  • 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  • p-value: {bootstrap_pvalue:.6f}")

    # ========================================================================
    # ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Бутстрап-распределение разности медиан
    # ========================================================================
    print("\n📊 Создание бутстрап-визуализации разности медиан...")
    fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
    ax_bs.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_bs.axvline(0, color='red', linestyle='--', linewidth=2,
                  label='Нулевая гипотеза (разница = 0)')
    observed_diff = np.median(premium_arr) - np.median(no_premium_arr)
    ax_bs.axvline(observed_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Наблюдаемая разница: {observed_diff:.0f} SR')
    ax_bs.axvline(ci_lower, color='purple', linestyle=':', linewidth=1.5,
                  label=f'95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]')
    ax_bs.axvline(ci_upper, color='purple', linestyle=':', linewidth=1.5)
    ax_bs.set_xlabel('Разность медиан (SR)', fontsize=11)
    ax_bs.set_ylabel('Частота', fontsize=11)
    ax_bs.set_title('Бутстрап-распределение разности медиан\n'
                    'С премиум материалами vs Без премиум',
                    fontweight='bold', fontsize=13, pad=15)
    ax_bs.legend(loc='best', fontsize=9)
    ax_bs.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig_bs, 'hypothesis_B_bootstrap_distribution')
    plt.close()
    print("  ✓ Сохранено: hypothesis_B_bootstrap_distribution.png и .svg")

    # ========================================================================
    # ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ (1x2)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    bp = ax1.boxplot([no_premium_prices, premium_prices],
                     tick_labels=['Без премиум', 'С премиум'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'gold']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Цена')
    ax1.set_title('Распределение цен', fontweight='bold')

    ax2 = axes[1]
    medians = [no_premium_prices.median(), premium_prices.median()]
    bars = ax2.bar(['Без премиум', 'С премиум'], medians,
                   color=['lightblue', 'gold'], edgecolor='black')
    ax2.set_ylabel('Медианная цена')
    ax2.set_title('Сравнение медианных цен', fontweight='bold')
    for bar, median in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{median:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'hypothesis_B_premium_materials')

    tests_passed = sum([
        u_pvalue < 0.05,
        t_pvalue < 0.05,
        ci_lower > 0 and bootstrap_pvalue < 0.05
    ])

    print(f"\n{'=' * 70}")
    print(f"ИТОГ ПО ГИПОТЕЗЕ B: {'✓ ПОДТВЕРЖДЕНА' if tests_passed >= 2 else '✗ НЕ ПОДТВЕРЖДЕНА'}")
    print(f"{'=' * 70}")

# ==============================================================================
# ЧАСТЬ 13 - Б: ПРОВЕРКА ПРИРОДЫ old_price(ДЕТЕКЦИЯ УТЕЧКИ)
#===============================================================================

def analyze_flatpack_hypotheses(df: pd.DataFrame) -> None:
    """
    Проводит статистический анализ дополнительных гипотез, связанных
    с flat-pack концепцией IKEA и ценообразованием.

    Функция рассчитывает геометрические показатели товаров (объем) и их
    удельную стоимость, после чего проверяет четыре ключевые гипотезы:
    - F1: Взаимосвязь плотности flat-pack (объема) и конечной цены товара.
    - F2: Различие в плотности (объеме) между составными и одиночными товарами.
    - F3: Различие в плотности между премиум и стандартными категориями.
    - F4: Различия в эффективности плотности по различным категориям товаров.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм с данными о товарах IKEA. Обязательно должен
        содержать колонки: 'price', 'category', 'name', а также геометрические
        размеры 'width', 'height', 'depth' (или готовую колонку 'volume').

    Returns
    -------
    pd.DataFrame
        Копия исходного датафрейма с добавленными расчетными признаками:
        - 'volume' (если отсутствовал): физический объем товара.
        - 'is_composite': флаг составного товара (набора).
        - 'flatpack_density': показатель плотности (объем / цена).
        - 'price_per_volume': удельная стоимость объема (цена / объем).

    Note
    ----
    Для проверки гипотез используются непараметрические статистические тесты:
    коэффициент ранговой корреляции Спирмена, критерий Манна-Уитни (U-тест)
    и критерий Краскела-Уоллиса, так как распределения объемов и цен
    значительно отличаются от нормального.
    """
    print("\n" + "=" * 70)
    print("ДОПОЛНИТЕЛЬНЫЕ ГИПОТЕЗЫ: FLAT-PACK КОНЦЕПЦИЯ IKEA")
    print("=" * 70)

    df = df.copy()

    # ========================================================================
    # Создаём необходимые признаки
    # ========================================================================
    print("\n Создание признаков для flat-pack анализа...")

    if 'volume' not in df.columns:
        df['volume'] = df['width'] * df['height'] * df['depth']
        print(f"  ✓ Создан volume: {df['volume'].notna().sum()} значений")

    if 'is_composite' not in df.columns:
        all_dims_missing = (
                df['depth'].isna() &
                df['height'].isna() &
                df['width'].isna()
        )
        has_slash = df['name'].str.contains('/', na=False)
        df['is_composite'] = (all_dims_missing | has_slash).astype(int)
        print(f"  ✓ Создан is_composite: {df['is_composite'].sum()} составных товаров")

    # Признаки flat-pack эффективности
    df['flatpack_density'] = df['volume'] / df['price']
    df['price_per_volume'] = df['price'] / (df['volume'] + 1)

    print(f"\n СТАТИСТИКА FLAT-PACK ЭФФЕКТИВНОСТИ:")
    print(f"  • Средняя flatpack_density: {df['flatpack_density'].mean():.6f}")
    print(f"  • Медианная flatpack_density: {df['flatpack_density'].median():.6f}")
    print(f"  • Средняя price_per_volume: {df['price_per_volume'].mean():.4f} SR/см³")

    # ========================================================================
    # ГИПОТЕЗА F1
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ГИПОТЕЗА F1: Плотность flat-pack vs Цена")
    print(f"{'=' * 70}")
    print("H₀: Плотность flat-pack не коррелирует с ценой")
    print("H₁: Существует значимая корреляция")

    valid_mask = df['flatpack_density'].notna() & df['price'].notna()
    df_valid = df[valid_mask]

    rho, p_spearman = stats.spearmanr(df_valid['flatpack_density'], df_valid['price'])
    print(f"\n  Spearman ρ (flatpack_density vs price): {rho:.4f}, p = {p_spearman:.6f}")

    rho2, p2 = stats.spearmanr(df_valid['price_per_volume'], df_valid['price'])
    print(f"  Spearman ρ (price_per_volume vs price): {rho2:.4f}, p = {p2:.6f}")

    # ========================================================================
    # ГИПОТЕЗА F2
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ГИПОТЕЗА F2: Составные vs Одиночные товары")
    print(f"{'=' * 70}")
    print("H₀: Flat-pack эффективность одинакова")
    print("H₁: Составные товары эффективнее")

    composite = df[df['is_composite'] == 1]['flatpack_density'].dropna()
    single = df[df['is_composite'] == 0]['flatpack_density'].dropna()

    print(f"\n  Составных товаров: {len(composite)}")
    print(f"  Одиночных товаров: {len(single)}")

    p_mw = 1.0
    if len(composite) > 0 and len(single) > 0:
        print(f"  Медианная плотность составных: {composite.median():.6f}")
        print(f"  Медианная плотность одиночных: {single.median():.6f}")

        #  используем stats.mannwhitneyu
        _, p_mw = stats.mannwhitneyu(composite, single, alternative='greater')
        print(f"\n  Mann-Whitney U (составные > одиночные): p = {p_mw:.6f}")
    else:
        print(f"\n   Недостаточно данных для сравнения")

    # ========================================================================
    # ГИПОТЕЗА F3
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ГИПОТЕЗА F3: Премиум vs Стандарт (flat-pack эффективность)")
    print(f"{'=' * 70}")
    print("H₀: Flat-pack эффективность одинакова")
    print("H₁: Премиум-товары менее эффективны")

    premium_categories = ['Wardrobes', 'Sofas & armchairs', 'Beds',
                          'Sideboards, buffets & console tables', 'Cabinets & cupboards']
    premium = df[df['category'].isin(premium_categories)]['flatpack_density'].dropna()
    standard = df[~df['category'].isin(premium_categories)]['flatpack_density'].dropna()

    print(f"\n  Премиум-товаров: {len(premium)}")
    print(f"  Стандартных товаров: {len(standard)}")
    print(f"  Медианная плотность премиум: {premium.median():.6f}")
    print(f"  Медианная плотность стандарт: {standard.median():.6f}")

    #  используем stats.mannwhitneyu
    _, p_mw_prem = stats.mannwhitneyu(premium, standard, alternative='less')
    print(f"\n  Mann-Whitney U (премиум < стандарт): p = {p_mw_prem:.6f}")

    # ========================================================================
    # ГИПОТЕЗА F4
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ГИПОТЕЗА F4: Flat-pack эффективность по категориям")
    print(f"{'=' * 70}")
    print("H₀: Все категории имеют одинаковую эффективность")
    print("H₁: Категории значимо отличаются")

    groups = [group['flatpack_density'].dropna().values
              for name, group in df.groupby('category')
              if len(group['flatpack_density'].dropna()) > 5]

    p_kw = 1.0
    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis H: {h_stat:.3f}, p = {p_kw:.6f}")

        cat_stats = df.groupby('category')['flatpack_density'].agg(['median', 'mean', 'count'])
        cat_stats = cat_stats.sort_values('median', ascending=False)

        print(f"\n  Топ-5 категорий по flat-pack эффективности:")
        for cat, row in cat_stats.head(5).iterrows():
            print(f"    • {cat:40s} | медиана: {row['median']:.6f} | n={int(row['count'])}")

        print(f"\n  Топ-5 категорий по НЕЭФФЕКТИВНОСТИ:")
        for cat, row in cat_stats.tail(5).iterrows():
            print(f"    • {cat:40s} | медиана: {row['median']:.6f} | n={int(row['count'])}")

    # ========================================================================
    # ВЫВОДЫ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ВЫВОДЫ ПО FLAT-PACK ГИПОТЕЗАМ:")
    print(f"{'=' * 70}")

    print(f"\n1. Корреляция flatpack_density с ценой: ρ = {rho:.4f}")
    if abs(rho) > 0.3 and p_spearman < 0.05:
        print(f"    ЗНАЧИМАЯ умеренная корреляция — flat-pack влияет на цену!")
        if rho < 0:
            print(f"   → Отрицательная: высокая плотность = низкая цена")
            print(f"   → Это ЛОГИЧНО: дешёвые товары IKEA имеют большой объём за малые деньги")
    else:
        print(f"   ⚠️ Слабая корреляция")

    print(f"\n2. Составные vs Одиночные:")
    if p_mw < 0.05:
        print(f"    ЗНАЧИМОЕ различие (p = {p_mw:.6f})")
        if composite.median() > single.median():
            print(f"   → Составные товары БОЛЕЕ эффективны")
        else:
            print(f"   → Составные товары МЕНЕЕ эффективны (контринтуитивно!)")
    else:
        print(f"   ❌ Нет значимого различия")

    print(f"\n3. Премиум vs Стандарт:")
    if p_mw_prem < 0.05:
        print(f"    ЗНАЧИМОЕ различие (p = {p_mw_prem:.6f})")
        print(f"   → Премиум-товары имеют худшую flat-pack эффективность")
        print(f"   → Это объясняет недооценку модели!")
    else:
        print(f"   ❌ Нет значимого различия")

    print(f"\n4. Различия по категориям:")
    if len(groups) >= 2 and p_kw < 0.05:
        print(f"    ЗНАЧИМЫЕ различия между категориями (p = {p_kw:.6f})")
    else:
        print(f"   ❌ Нет значимых различий")

    return df

# =============================================================================
# ЧАСТЬ 14: ПОДГОТОВКА ДАННЫХ ДЛЯ ML (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# =============================================================================

def run_hypothesis_tests(df_unique: pd.DataFrame) -> None:
    """
   Запускает комплекс статистических тестов для проверки бизнес-гипотез.

    Последовательно проверяет семейство гипотез H1-H4, HA, HB, а также специфические
    гипотезы для плоской упаковки (Flatpack) товаров IKEA. Внутри каждого теста
    выполняются проверки на нормальность распределений, рассчитываются критерии
    (t-test, Mann-Whitney, ANOVA или Chi-Square) и строятся подтверждающие графики.

    Args:
        df_unique: Очищенный от дубликатов датасет IKEA, на основе которого
            формируются выборки для статистического анализа.

    Returns:
        None. Все результаты тестов, включая статистические показатели и p-value,
        выводятся напрямую в лог/консоль, а графики сохраняются на диск.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ")
    print("=" * 70)

    check_hypothesis_1(df_unique)
    check_hypothesis_2(df_unique)
    check_hypothesis_3(df_unique)
    plot_price_distribution_by_category(df_unique)
    check_hypothesis_4(df_unique)
    check_hypothesis_A(df_unique)
    check_hypothesis_B(df_unique)
    analyze_flatpack_hypotheses(df_unique)

    print("\n✅ Все статистические тесты завершены")

# ==============================================================================
# ОРКЕСТРАТОР ML-ПАЙПЛАЙНА
# ==============================================================================
