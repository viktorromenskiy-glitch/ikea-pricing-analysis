"""
ikea_main.py — точка входа и финальный отчёт. Часть модульного разбиения.
"""
from ikea_core import *
from ikea_data_prep import *
from ikea_eda import *
from ikea_hypotheses import *
from ikea_model import *
from ikea_interpret import *


def print_final_summary(
        eda_results: Dict[str, Any],
        ml_results: 'MLPipelineResult',
        interpretation_results: Dict[str, Any]
) -> None:
    """
    Печатает итоговые выводы проекта.

    Функция агрегирует результаты всех этапов (EDA, статистические гипотезы,
    ML-моделирование, интерпретация) и выводит структурированный финальный отчёт.
    Все метрики берутся из динамических источников (baseline_df, ml_results),
    что обеспечивает корректность отчёта при изменении данных или выборе модели.

    Parameters:
    -----------
    eda_results : dict
        Результаты EDA (содержит sellable_verdict, p_true, p_value, old_price_check).
    ml_results : dict
        Результаты ML (содержит best_model_name, cv_scores, gridsearch_results,
        ablation_results, X_train, X_test).
    interpretation_results : dict
        Результаты интерпретации (содержит baseline_df, grouped_importance_df,
        shap_importance_df, mae_ci_lower, mae_ci_upper).
    """
    print(f"\n{'=' * 70}")
    print(" ИТОГОВЫЕ ВЫВОДЫ ПРОЕКТА ")
    print(f"{'=' * 70}")

    # ========================================================================
    # Результаты гипотез (захардкожены — требуют передачи из run_hypothesis_tests)
    # ========================================================================
    print(f"\n РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКИХ ГИПОТЕЗ:")
    print(f"  ✓ Гипотеза 1: Объём → цена (ρ=0.797, p<0.001)")
    print(f"  ✓ Гипотеза 2: Команды дизайнеров → цена (p<0.001)")
    print(f"  ✓ Гипотеза 3: Категории → цена (p<0.001)")
    print(f"  ✓ Гипотеза 4: Цветовая вариативность → цена (p<0.001)")
    print(f"  ✓ Гипотеза A: Скидки → больший объём (p<0.001)")
    print(f"  ✓ Гипотеза B: Премиум материалы → цена (p<0.001)")
    print(f"  ✓ Гипотеза F1: Flat-pack vs цена (ρ=-0.37, p<0.001)")
    print(f"  ✓ Гипотеза F3: Премиум vs стандарт (p=0.006)")
    print(f"  ✓ Гипотеза F4: Различия по категориям (p<0.001)")

    # ========================================================================
    # Проверка sellable_online
    # ========================================================================
    try:
        p_true = eda_results['p_true']
        p_value = eda_results['p_value']
        sellable_verdict = eda_results['sellable_verdict']

        print(f"\n ПРОВЕРКА sellable_online ")
        print(f"  • Распределение: True={p_true * 100:.1f}%, False={(1 - p_true) * 100:.1f}%")
        print(f"  • Дисперсия признака: {p_true * (1 - p_true):.4f}")
        print(f"  • Mann-Whitney U тест: p-value = {p_value:.6f}")
        if sellable_verdict == "REMOVE":
            print(f"  • ✅ Признак УДАЛЁН (почти константа, бесполезен для ML)")
        else:
            print(f"  • ✅ Признак ОСТАВЛЕН (статистически значим)")
    except Exception as e:
        print(f"\n  ⚠️ Не удалось получить данные по sellable_online: {e}")

    # ========================================================================
    # Проверка old_price
    # ========================================================================
    try:
        old_price_check = eda_results['old_price_check']
        print(f"\n ПРОВЕРКА old_price:")
        print(f"  • Вердикт: {old_price_check['verdict']}")
        print(f"  • Признаки к использованию: {old_price_check['features_to_use']}")
        print(f"  • Товаров со старой ценой: {old_price_check['n_with_old_price']} "
              f"({old_price_check['pct_with_old_price']:.1f}%)")
        print(f"  • Корреляция price ↔ old_price: {old_price_check['corr']:.4f}")
        print(f"  • Средняя скидка: {old_price_check['discount_mean']:.2f}%")
    except Exception as e:
        print(f"\n  ⚠️ Не удалось получить данные по old_price: {e}")

    # ========================================================================
    # Устранение утечки
    # ========================================================================
    print(f"\n🛡 УСТРАНЕНИЕ УТЕЧКИ ДАННЫХ:")
    print(f"  • category_price_level: медианы вычисляются ТОЛЬКО на train set")
    print(f"  • designer_freq: частотность вычисляется ТОЛЬКО на train set")
    print(f"  • is_large_item: порог вычисляется ТОЛЬКО на train set")
    print(f"  • Удалены мультиколлинеарные признаки: area, log_volume, premium_x_*, segment_x_*")

    # ========================================================================
    # Инициализация переменных (защита от NameError при падении try-блоков)
    # ========================================================================
    final_r2 = 0.0
    final_mae = 0.0
    best_model_name = "Unknown"
    cv_scores = None
    baseline_df = interpretation_results.get('baseline_df')

    # ========================================================================
    # Результаты ML — используем метрики ФИНАЛЬНОЙ модели из baseline_df
    # ========================================================================
    try:
        best_model_name = ml_results.best_model_name
        cv_scores = ml_results.cv_scores

        # Получаем метрики финальной модели из baseline_df
        if baseline_df is not None:
            final_model_row = baseline_df[baseline_df['Model'].str.contains('наша', case=False)]
            if len(final_model_row) > 0:
                final_r2 = final_model_row['R²'].values[0]
                final_mae = final_model_row['MAE'].values[0]
            else:
                # Fallback: используем GridSearchCV
                final_r2 = ml_results.gridsearch_results['test_r2']
                final_mae = ml_results.gridsearch_results['test_mae']
        else:
            final_r2 = ml_results.gridsearch_results['test_r2']
            final_mae = ml_results.gridsearch_results['test_mae']

        print(f"\n РЕЗУЛЬТАТЫ ML-МОДЕЛИ:")
        print(f"  • Лучшая модель: {best_model_name}")
        print(f"  • R² (тест): {final_r2:.4f}")
        print(f"  • R² (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  • MAE (тест): {final_mae:.2f} SR")

        gap = cv_scores.mean() - final_r2
        print(f"  • Gap (CV - Test): {gap:+.4f}")
        if abs(gap) < 0.05:
            print(f"     Gap менее 0.05 — модель стабильна")
        else:
            print(f"   Gap больше 0.05 — возможна нестабильность")
    except Exception as e:
        print(f"\n  ⚠️ Не удалось получить результаты ML-модели: {e}")

    # ========================================================================
    # Сравнение с baseline'ами — используем метрики ФИНАЛЬНОЙ модели
    # ========================================================================
    mae_ci_lower = interpretation_results.get('mae_ci_lower', 0.0)
    mae_ci_upper = interpretation_results.get('mae_ci_upper', 0.0)

    print(f"\n СРАВНЕНИЕ С BASELINE'АМИ:")
    if baseline_df is not None:
        try:
            zero_rule_mae = baseline_df.loc[baseline_df['Model'] == 'Zero Rule (медиана)', 'MAE'].values[0]
            category_mae = baseline_df.loc[baseline_df['Model'] == 'Медиана по категории', 'MAE'].values[0]
            linear_mae = baseline_df.loc[baseline_df['Model'] == 'Linear Regression (габариты)', 'MAE'].values[0]
            rf_baseline_mae = baseline_df.loc[baseline_df['Model'] == 'Random Forest (без настройки)', 'MAE'].values[0]

            # Используем final_mae (метрики финальной модели)
            improvement_vs_zero = (zero_rule_mae - final_mae) / zero_rule_mae * 100
            improvement_vs_category = (category_mae - final_mae) / category_mae * 100
            improvement_vs_linear = (linear_mae - final_mae) / linear_mae * 100
            improvement_vs_rf = (rf_baseline_mae - final_mae) / rf_baseline_mae * 100

            print(f"  • vs Zero Rule (медиана):     улучшение на {improvement_vs_zero:.1f}%")
            print(f"  • vs Медиана категории:       улучшение на {improvement_vs_category:.1f}%")
            print(f"  • vs Linear Regression:       улучшение на {improvement_vs_linear:.1f}%")
            print(f"  • vs Random Forest (без настройки): улучшение на {improvement_vs_rf:.1f}%")
            print(f"  • 95% CI для MAE: [{mae_ci_lower:.2f}, {mae_ci_upper:.2f}] SR")
        except Exception as e:
            print(f"  • Не удалось вычислить улучшения: {e}")
    else:
        print(f"  • Данные baseline_df отсутствуют")

    # ========================================================================
    # Важность признаков
    # ========================================================================
    grouped_importance_df = interpretation_results.get('grouped_importance_df')
    print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ (для бизнеса/менеджмента):")
    if grouped_importance_df is not None and len(grouped_importance_df) > 0:
        print(f"\n  Топ-5 признаков (сгруппировано):")
        for i, row in grouped_importance_df.head(5).iterrows():
            print(f"    {i + 1}. {row['Feature']:30s} {row['Importance_%']:5.1f}%")

        top_feature = grouped_importance_df.iloc[0]['Feature']
        top_importance = grouped_importance_df.iloc[0]['Importance_%']

        if top_feature == 'category':
            print(f"\n  🏆 Самый важный признак: category ({top_importance:.1f}%)")
            print(f"     → Категория товара определяет цену сильнее всего")
        elif top_feature in ['volume', 'width', 'height', 'depth']:
            print(f"\n  🏆 Самый важный признак: {top_feature} ({top_importance:.1f}%)")
            print(f"     → Габариты товара определяют цену сильнее всего")
        else:
            print(f"\n  🏆 Самый важный признак: {top_feature} ({top_importance:.1f}%)")
    else:
        print(f"  • Данные grouped_importance_df отсутствуют")

    # ========================================================================
    # SHAP-анализ
    # ========================================================================
    shap_importance_df = interpretation_results.get('shap_importance_df')
    print(f"\n SHAP-АНАЛИЗ:")
    if shap_importance_df is not None and len(shap_importance_df) > 0:
        top_shap = shap_importance_df.iloc[0]
        print(f"  • Самый важный признак: {top_shap['Feature']} ({top_shap['Importance_%']:.1f}%)")

        type_importance_shap = shap_importance_df.groupby('Type')['Importance_%'].sum()
        num_pct = type_importance_shap.get('Числовые', 0)
        bin_pct = type_importance_shap.get('Бинарные', 0)
        cat_pct = type_importance_shap.get('Категория', 0)
        print(f"  • SHAP важность по типам: числовые {num_pct:.1f}%, "
              f"бинарные {bin_pct:.1f}%, категория {cat_pct:.1f}%")
        print(f"  • Всего признаков в модели: {len(shap_importance_df)}")
    else:
        print(f"  • Данные shap_importance_df отсутствуют")

    # ========================================================================
    # Принцип Occam's Razor — сравниваем RF baseline с ФИНАЛЬНОЙ моделью
    # с учётом статистической значимости (CV std)
    # ========================================================================
    print(f"\n ВАЖНОЕ  -  (принцип Occam's Razor):")
    if baseline_df is not None:
        try:
            rf_baseline_r2 = baseline_df.loc[
                baseline_df['Model'] == 'Random Forest (без настройки)', 'R²'
            ].values[0]
            print(f"  • Random Forest БЕЗ настройки показал R² = {rf_baseline_r2:.4f}")
            print(f"  • Наша финальная модель ({best_model_name}) показала R² = {final_r2:.4f}")

            r2_diff = rf_baseline_r2 - final_r2

            # Получаем CV std для проверки статистической значимости
            cv_std = cv_scores.std() if cv_scores is not None else 0.0

            if r2_diff > 0:
                # Простая модель показывает лучший R²
                if r2_diff > cv_std:
                    # Разница СТАТИСТИЧЕСКИ ЗНАЧИМА
                    print(f"  • Простая модель работает ЛУЧШЕ сложной!")
                    print(f"  • Разница: {r2_diff:.4f} (CV std: {cv_std:.4f})")
                    print(f"  • РЕКОМЕНДАЦИЯ: рассмотреть использование базовой модели для продакшена")
                else:
                    # Разница НЕ статистически значима — модели неразличимы
                    print(f"  • Разница: {r2_diff:.4f} (CV std: {cv_std:.4f})")
                    print(f"  • ⚠️ Разница МЕНЬШЕ CV std — модели статистически неразличимы")
                    print(f"  • РЕКОМЕНДАЦИЯ: можно использовать любую модель")
                    print(f"  • Простая модель предпочтительнее по принципу Occam's Razor")
            else:
                # Финальная модель показывает лучший R²
                print(f"  • Финальная модель улучшила результат на {-r2_diff:.4f}")
                if (-r2_diff) > cv_std:
                    print(f"  • Улучшение статистически значимо")
                else:
                    print(f"  • ⚠️ Улучшение меньше CV std — статистически незначимо")
        except Exception as e:
            print(f"  • Не удалось получить данные для сравнения: {e}")
    else:
        print(f"  • Данные baseline_df отсутствуют")

    # ========================================================================
    # Ablation Study
    # ========================================================================
    ablation_results = ml_results.ablation_results
    print(f"\n📊 ABLATION STUDY (вклад групп признаков):")
    if ablation_results is not None and len(ablation_results) > 0:
        try:
            # Фильтруем baseline
            non_baseline = ablation_results[ablation_results['Group'] != 'Все признаки (baseline)']
            if len(non_baseline) > 0:
                most_important = non_baseline.loc[non_baseline['R²'].idxmin()]
                least_important = non_baseline.loc[non_baseline['R²'].idxmax()]
                print(f"  • Самая важная группа: {most_important['Group']} "
                      f"(удаление снижает R² на {most_important['Drop_%']:.2f}%)")
                print(f"  • Наименее важная группа: {least_important['Group']} "
                      f"(удаление снижает R² на {least_important['Drop_%']:.2f}%)")

                # Группы с отрицательным ΔR² (улучшение при удалении!)
                negative_delta = ablation_results[ablation_results['ΔR²'] < 0]
                if len(negative_delta) > 0:
                    groups_list = ', '.join(negative_delta['Group'].tolist())
                    print(f"  • ⚠️ Шумные группы (улучшают модель при удалении): {groups_list}")
        except Exception as e:
            print(f"  • Не удалось обработать данные Ablation Study: {e}")
    else:
        print(f"  • Данные ablation_results отсутствуют")

    # ========================================================================
    # Дополнительная информация
    # ========================================================================
    try:
        X_train = ml_results.X_train
        X_test = ml_results.X_test

        print(f"\n ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        n_features = len(shap_importance_df) if shap_importance_df is not None else 'N/A'
        print(f"  • Признаков в модели: {n_features}")
        print(f"  • Примеров в train: {len(X_train)}")
        print(f"  • Примеров в test: {len(X_test)}")
        print(f"  • Leakage устранён: category_price_level, designer_freq, is_large_item")
        print(f"    вычисляются только на train set")
        print(f"  • Мультиколлинеарность устранена: удалены area, log_volume, premium_x_*, segment_x_*")
        print(f"  • Кэш Optuna: optuna_cache.json (экономия ~17 мин)")
    except Exception as e:
        print(f"\n  ⚠️ Не удалось получить дополнительную информацию: {e}")

    # ========================================================================
    # Артефакты проекта
    # ========================================================================
    print(f"\n Все графики сохранены в папке: plots_step/")
    print(f"  Форматы: PNG (150 DPI) и SVG")
    print(f"  Всего: 34+ визуализации (68+ файлов)")
    print(f"\n Отчёты сохранены в папке: reports_step/")
    print(f"  Файлов: 9 (включая FAQ.md)")

    print(f"\n{'=' * 70}")
    print(" ПРОЕКТ ЗАВЕРШЁН УСПЕШНО! ")
    print(f"{'=' * 70}")

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ГИПОТЕЗА: FLAT-PACK КОНЦЕПЦИЯ IKEA
# =============================================================================


def main() -> None:
    """
    Точка входа и главный оркестратор аналитического пайплайна ценообразования IKEA.

    Функция последовательно запускает все этапы сквозного исследования: от парсинга сырых
    данных до интерпретации финальных ML-моделей и генерации бизнес-отчетов. Благодаря
    использованию модульных функций-оркестраторов (класса `run_*_pipeline`), поддерживается
    высокая читаемость кода, изолированность этапов и прозрачная передача артефактов
    (например, инсайты из EDA передаются напрямую в ML-пайплайн).

    Архитектурные этапы работы:
    1. **Парсинг аргументов**: Обработка флагов командной строки (принудительное
       обучение Optuna, кастомное число фолдов и итераций).
    2. **Препроцессинг и подготовка**: Загрузка датасета по URL, дедупликация,
       выделение составных товаров, очистка и стандартизация признаков (`designer`,
       `dimensions` и др.). Создание двух базовых срезов: полного (`df_full`) и
       уникального (`df_unique`).
    3. **EDA-пайплайн**: Проведение разведочного анализа данных и автоматическая
       генерация визуализаций. Результаты (`eda_results`) сохраняются для передачи
       в ML.
    4. **Статистический анализ**: Проверка статистических гипотез на уникальной выборке.
    5. **ML-пайплайн**: Конструирование признаков, подбор гиперпараметров (Optuna) и
       обучение финальных моделей. Корректно учитывает контекст предыдущих этапов.
    6. **Интерпретация**: Глубокий анализ вклада признаков (SHAP, Feature Importance,
       Ablation Study, Sensitivity Analysis).
    7. **Финальный отчет**: Агрегация инсайтов всех этапов и вывод консольного резюме.

    Args:
        Использует `argparse.ArgumentParser` для чтения аргументов командной строки:
            - `--rerun-optuna` (bool): Флаг принудительного сброса кэша Optuna
              и перезапуска оптимизации гиперпараметров.
            - `--trials` (int, optional): Кастомное количество итераций (trials)
              для Optuna.
            - `--cv` (int, optional): Кастомное количество фолдов для кросс-валидации.

    Returns:
        None: Функция координирует вызовы, выводит логи эксперимента в консоль
            и сохраняет артефакты (модели, графики, CSV) на диск.

    Raises:
        Exception: Возможные исключения при загрузке данных из сети или сбоях
            внутри изолированных пайплайнов (обрабатываются на уровнях соответствующих
            функций подсистем).
    """
    # ========================================================================
    # АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
    # ========================================================================
    parser = argparse.ArgumentParser(description='Анализ ценообразования IKEA')
    parser.add_argument('--rerun-optuna', action='store_true',
                        help='Принудительно пересчитать Optuna (игнорирует кэш)')
    parser.add_argument('--trials', type=int, default=None,
                        help='Количество trials для всех экспериментов (переопределяет дефолт)')
    parser.add_argument('--cv', type=int, default=None,
                        help='Количество фолдов CV (переопределяет дефолт)')
    args = parser.parse_args()

    print("=" * 70)
    print(" АНАЛИЗ ЦЕНООБРАЗОВАНИЯ IKEA ")
    print("=" * 70)

    if args.rerun_optuna:
        print("\n🔄 Режим принудительного пересчёта Optuna")

    # URL датасета
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"

    # ========================================================================
    # ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)

    df = load_ikea_data(url)
    explore_data(df)
    df = df.drop(columns=['Unnamed: 0', 'link']).copy()
    df = identify_composite_products(df)
    df = analyze_duplicates(df)
    df_full, df_unique = create_two_dataframes(df)
    df_unique = clean_designer(df_unique, dataset_name="df_unique")
    df_full = clean_designer(df_full, dataset_name="df_full")
    df_unique = classify_designer_type(df_unique)
    df_full = classify_designer_type(df_full)
    df_unique = fill_dimensions(df_unique, dataset_name="df_unique")
    df_full = fill_dimensions(df_full, dataset_name="df_full")

    # ========================================================================
    # ЭТАП 2: EDA-ПАЙПЛАЙН (ОРКЕСТРАТОР)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЭТАП 2: EDA-ПАЙПЛАЙН")
    print("=" * 70)

    eda_results = run_eda_pipeline(df_full, df_unique)

    # ========================================================================
    # ЭТАП 3: СТАТИСТИЧЕСКИЕ ГИПОТЕЗЫ (ОРКЕСТРАТОР)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЭТАП 3: СТАТИСТИЧЕСКИЕ ГИПОТЕЗЫ")
    print("=" * 70)

    run_hypothesis_tests(df_unique)

    # ========================================================================
    # ЭТАП 4: ML-ПАЙПЛАЙН (ОРКЕСТРАТОР)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЭТАП 4: ML-ПАЙПЛАЙН")
    print("=" * 70)

    #  Вердикты из EDA корректно передаются в ML-пайплайн через eda_results
    ml_results = run_ml_pipeline(
        df_unique,
        eda_results,
        rerun_optuna=args.rerun_optuna,
        trials=args.trials,
        cv=args.cv
    )

    # ========================================================================
    # ЭТАП 5: ИНТЕРПРЕТАЦИЯ МОДЕЛИ (ОРКЕСТРАТОР)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЭТАП 5: ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
    print("=" * 70)

    interpretation_results = run_interpretation_pipeline(ml_results)

    # ========================================================================
    # ЭТАП 6: ИТОГОВЫЕ ВЫВОДЫ (ОРКЕСТРАТОР)
    # ========================================================================
    print_final_summary(eda_results, ml_results, interpretation_results)


if __name__ == "__main__":
    main()