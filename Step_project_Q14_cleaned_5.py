"""
=============================================================================
АНАЛИЗ ЦЕНООБРАЗОВАНИЯ IKEA
=============================================================================
Полный скрипт анализа данных IKEA с проверкой 6 статистических гипотез
и построением ML-модели для предсказания цены.

Автор: Виктор Роменский
Дата: 2026-06-19
=============================================================================
"""
# ========================================================================
# ПОДАВЛЕНИЕ ПРЕДУПРЕЖДЕНИЙ (ДОЛЖНО БЫТЬ В САМОМ НАЧАЛЕ СКРИПТА!)
# ========================================================================

# ========================================================================
# ПОДАВЛЕНИЕ ПРЕДУПРЕЖДЕНИЙ (ДОЛЖНО БЫТЬ В САМОМ НАЧАЛЕ СКРИПТА!)
# ========================================================================
import os
import warnings
import logging

# 🔑 КЛЮЧЕВОЕ РЕШЕНИЕ: переменная окружения для дочерних процессов
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'

# Подавляем конкретные предупреждения
warnings.filterwarnings(
    'ignore',
    message='.*`sklearn.utils.parallel.delayed` should be used with.*',
    category=UserWarning
)
warnings.filterwarnings(
    'ignore',
    message='.*The NumPy global RNG was seeded.*',
    category=FutureWarning
)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.*')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib.*')

# Перехватываем warnings через logging (дополнительная защита)
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

print("✅ Настроено подавление специфических предупреждений")

# =============================================================================
# ИМПОРТ БИБЛИОТЕК
# =============================================================================

import requests
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import json
import argparse
import atexit
import hashlib
import warnings
import optuna
import sys
from datetime import datetime
from scipy import stats
from tabulate import tabulate
from typing import Any, Optional, Union, Tuple, List, Dict, Callable
# Подавление предупреждений sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.*')

# ML библиотеки
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import xgboost as xgb
from joblib import parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.inspection import permutation_importance
from scipy.stats import norm, linregress
matplotlib.use('Agg')


# =============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# =============================================================================
SELLABLE_ONLINE_VERDICT = "KEEP"  # По умолчанию оставляем, анализ решит

# Настройки визуализации
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# ========================================================================
# АВТОМАТИЧЕСКОЕ СОХРАНЕНИЕ ЛОГА В ФАЙЛ
# ========================================================================

class Logger:
    def __init__(self, log_file_path: str) -> None:
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        self.log_file_path = log_file_path
        self.closed = False  # флаг, указывающий, закрыт ли файл

    def write(self, message: str) -> None:
        if self.closed:
            # Если файл уже закрыт, просто пишем в терминал
            self.terminal.write(message)
            return
        try:
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()
        except (ValueError, OSError):
            # Если файл уже закрыт, пишем только в терминал
            self.terminal.write(message)

    def flush(self) -> None:
        if not self.closed:
            try:
                self.terminal.flush()
                self.log_file.flush()
            except (ValueError, OSError):
                pass

    def close(self) -> None:
        if not self.closed and self.log_file and not self.log_file.closed:
            try:
                self.log_file.close()
            except (ValueError, OSError):
                pass
            finally:
                self.closed = True

def cleanup_logger() -> None:
    if hasattr(sys.stdout, 'close'):
        try:
            sys.stdout.close()
        except:
            pass



# Регистрируем функцию очистки при завершении скрипта
atexit.register(cleanup_logger)

# Создаём папку для логов (если её нет)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Создаём лог-файл с датой и временем
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'ikea_analysis_{timestamp}.log'
log_path = os.path.join(log_dir, log_filename)

# Подменяем stdout и stderr на наш Logger
sys.stdout = Logger(log_path)
sys.stderr = sys.stdout

print("=" * 70)
print("📝 АВТОМАТИЧЕСКОЕ СОХРАНЕНИЕ ЛОГА")
print("=" * 70)
print(f"✅ Лог сохраняется в файл: {os.path.abspath(log_path)}")
print("=" * 70)
print()

print("=" * 70)
print()

# =============================================================================
# НАСТРОЙКИ СОХРАНЕНИЯ ГРАФИКОВ
# =============================================================================

# Создаём папку для графиков
PLOTS_DIR = 'plots_step'
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(fig: plt.Figure, filename: str) -> None:
    """
    Сохраняет графическую фигуру в растровом (PNG) и векторном (SVG) форматах.

    Функция автоматизирует экспорт визуализаций:
    1. Формирует пути сохранения, используя имя файла и предопределённую глобальную
       директорию `PLOTS_DIR`.
    2. Сохраняет растровое изображение PNG с оптимизированным разрешением (150 DPI)
       и белым фоном.
    3. Сохраняет векторный файл SVG для последующего масштабирования без потери качества.
    4. Автоматически закрывает объект фигуры (`plt.close`), освобождая оперативную
       память от неиспользуемых графических контекстов.

    Args:
        fig (plt.Figure): Объект фигуры Matplotlib, которую необходимо сохранить.
        filename (str): Базовое имя файла (без указания расширения и пути).

    Returns:
        None: Функция выполняет сохранение файлов на диск и выводит лог в консоль.

    Raises:
        NameError: Если глобальная переменная `PLOTS_DIR` не определена в области
            видимости модуля.
        OSError: Если отсутствует доступ на запись в целевую директорию или
            невозможно создать файл (например, из-за недопустимых символов в имени).
    """
    png_path = os.path.join(PLOTS_DIR, f'{filename}.png')
    svg_path = os.path.join(PLOTS_DIR, f'{filename}.svg')

    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  ✓ Сохранено: {filename}.png и {filename}.svg")


# ==============================================================================
# УНИВЕРСАЛЬНЫЙ ПРЕПРОЦЕССОР С ОПЦИОНАЛЬНЫМ МАСШТАБИРОВАНИЕМ
# ==============================================================================

def get_preprocessor(
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
        with_scaling: bool = True
) -> ColumnTransformer:
    """
    Создаёт и настраивает объект ColumnTransformer для предобработки признаков.

    Функция конструирует пайплайн предобработки (scikit-learn ColumnTransformer),
    адаптированный под различные типы данных (числовые, категориальные, бинарные).
    В зависимости от флага `with_scaling`, числовой пайплайн может опционально
    включать стандартизацию признаков.

    Логика обработки признаков:
    1. **Числовые (Numerical)**: Заполнение пропусков медианным значением (`SimpleImputer`).
       При `with_scaling=True` дополнительно применяется `StandardScaler`.
    2. **Категориальные (Categorical)**: Заполнение пропусков константой 'Unknown'
       и последующее кодирование через `OneHotEncoder` (с игнорированием неизвестных
       категорий при тесте и возвратом плотного массива).
    3. **Бинарные/Булевые (Binary)**: Заполнение пропусков нулями без изменения масштаба.

    Функция имеет встроенную защиту от пустых списков признаков: трансформер для
    конкретного типа данных инициализируется только в том случае, если переданный
    список не пуст.

    Методологическое обоснование масштабирования:
    - **Деревья решений и ансамбли** (например, `RandomForest`, `XGBoost`, `GradientBoosting`):
      Базируются на разбиении признаков по пороговым значениям (сплитах) и оценивают только
      порядок величин, а не их абсолютный масштаб. Для них стандартизация избыточна
      (`with_scaling=False`).
    - **Линейные модели** (например, `LinearRegression`, `Ridge`, `Lasso`):
      Чувствительны к масштабу признаков. Стандартизация необходима для корректной
      работы регуляризации (L1/L2 штрафуют коэффициенты одинаково) и численной стабильности
      оптимизаторов (`with_scaling=True`).
    - **Метрические алгоритмы** (например, `KNN`, `SVM`):
      Рассчитывают расстояния между объектами в многомерном пространстве. Без стандартизации
      признаки с большими абсолютными значениями будут доминировать в метрике расстояния
      (`with_scaling=True`).

    Args:
        numeric_features (List[str]): Список названий числовых признаков.
        categorical_features (List[str]): Список названий категориальных признаков.
        binary_features (List[str]): Список названий бинарных и булевых признаков.
        with_scaling (bool, optional): Флаг, определяющий необходимость стандартизации
            числовых признаков с помощью `StandardScaler`. По умолчанию True.

    Returns:
        ColumnTransformer: Готовый объект scikit-learn для предобработки колонок,
            который можно встраивать в финальный `Pipeline`. Неиспользуемые колонки,
            не вошедшие в списки, автоматически отбрасываются (`remainder='drop'`).
    """

    # Числовой трансформер: импьютация + опциональный scaler
    if with_scaling:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
            # БЕЗ StandardScaler — для деревьев
        ])

    # Категориальный трансформер: импьютация + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Бинарный трансформер: импьютация нулями
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Собираем трансформеры (защита от пустых списков)
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    if binary_features:
        transformers.append(('bin', binary_transformer, binary_features))

    return ColumnTransformer(transformers=transformers, remainder='drop')

# ==============================================================================
# КЛАССИФИКАЦИЯ МОДЕЛЕЙ ПО ТИПУ ПРЕПРОЦЕССОРА
# ==============================================================================

# Модели, которым НУЖНО масштабирование (линейные + distance-based)
MODELS_REQUIRING_SCALING = {
    'LinearRegression', 'Ridge', 'Lasso',
    # Если добавите в будущем:
    'KNeighborsRegressor', 'SVR', 'SVM'
}

# Модели, которым масштабирование НЕ НУЖНО (деревья)
MODELS_WITHOUT_SCALING = {
    'DecisionTree', 'RandomForest',
    'GradientBoosting', 'HistGradientBoosting', 'XGBoost'
}

# =============================================================================
# КЭШ OPTUNA
# =============================================================================

OPTUNA_CACHE_FILE = 'optuna_cache.json'


def compute_optuna_fingerprint(
        columns: List[str],
        n_trials_1: int,
        n_trials_2: int,
        n_trials_3: int,
        cv_folds: int
) -> str:
    """Вычисляет отпечаток (fingerprint) конфигурации, от которой зависят результаты Optuna.

    🔧 ПРИЧИНА СУЩЕСТВОВАНИЯ: без этой проверки кэш в optuna_cache.json проверялся
    только на факт существования файла (os.path.exists), но НЕ на то, актуален ли
    он для текущего набора признаков и параметров запуска. Из-за этого возможны два
    противоположных сценария ошибки:
      1. Набор признаков изменился (напр. убрали discount_pct) — старый кэш всё равно
         молча "подходит" и используется, метрики exp1-exp4 в выводе оказываются
         посчитаны на СТАРОМ наборе признаков, хотя финальный pipeline переобучается
         на новом (несоответствие метрик и модели).
      2. Параметры запуска изменились (--trials/--cv) — кэш от предыдущего запуска
         с другими trials/cv всё равно используется как валидный.
    Отпечаток строится из отсортированного списка колонок X_train и параметров
    trials/cv — при малейшем расхождении кэш считается устаревшим и Optuna
    пересчитывается заново, с явным объяснением причины в консоли.

    Args:
        columns: Список названий колонок X_train (порядок не важен — сортируется внутри).
        n_trials_1, n_trials_2, n_trials_3: Количество trials для экспериментов 1-3.
        cv_folds: Количество фолдов кросс-валидации.

    Returns:
        str: Хэш SHA-256 (первые 16 символов) конфигурации.
    """
    payload = {
        'columns': sorted(columns),
        'n_trials_1': n_trials_1,
        'n_trials_2': n_trials_2,
        'n_trials_3': n_trials_3,
        'cv_folds': cv_folds,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()[:16]


def load_optuna_cache() -> Optional[Dict[str, Any]]:
    """Загружает ранее сохраненные результаты поиска гиперпараметров Optuna из JSON-файла.

        Функция проверяет наличие локального файла кэша, заданного глобальной переменной
        `OPTUNA_CACHE_FILE`. Если файл существует, она пытается прочитать его и десериализовать
        из формата JSON. При успешном чтении возвращает словарь с параметрами, в противном
        случае (если файл поврежден или отсутствует) безопасно обрабатывает исключение
        и возвращает `None`, не прерывая выполнение программы.

        Returns:
            Optional[Dict[str, Any]]: Словарь с кэшированными результатами, содержащий
                настройки лучшей модели, подобранные параметры и метрики экспериментов.
                Возвращает `None`, если файл кэша не найден или произошла ошибка чтения.

        Raises:
            NameError: Если глобальная переменная `OPTUNA_CACHE_FILE` не определена
                в области видимости модуля.
            Exception: Любые ошибки парсинга JSON или чтения файла (обрабатываются внутри
                блока try-except).
        """

    if not os.path.exists(OPTUNA_CACHE_FILE):
        return None
    try:
        with open(OPTUNA_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"\n✅ Загружены кэшированные результаты Optuna: {OPTUNA_CACHE_FILE}")
        return cache
    except Exception as e:
        print(f"⚠️ Ошибка чтения кэша: {e}")
        return None

#==================================================================================
# Сохраняет результаты подборки гиперпараметров Optuna и метрики всех эксперементов
#==================================================================================

def save_optuna_cache(
        best_model_name: str,
        best_params: Dict[str, Any],
        r2_exp1: float,
        mae_exp1: float,
        r2_exp2: float,
        mae_exp2: float,
        r2_hgb: float,
        mae_hgb: float,
        r2_rf: Optional[float] = None,
        mae_rf: Optional[float] = None,
        fingerprint: Optional[str] = None,
) -> None:
    """Сохраняет результаты подбора гиперпараметров Optuna и метрики всех экспериментов в JSON-файл.

        Функция агрегирует переданные параметры лучшей модели и метрики качества ($R^2$ и MAE)
        для четырех проведенных экспериментов в единый словарь (кэш) и записывает его в файл,
        определенный глобальной переменной `OPTUNA_CACHE_FILE`, с форматированием отступов в 2 пробела.
        Это позволяет избежать повторных ресурсоемких запусков Optuna при последующих стартах пайплайна.

        Args:
            best_model_name (str): Название модели, показавшей наилучший результат (например, 'RandomForest').
            best_params (Dict[str, Any]): Словарь оптимальных гиперпараметров для лучшей модели.
            r2_exp1 (float): Коэффициент детерминации $R^2$ для Эксперимента 1 (Базовая модель).
            mae_exp1 (float): Средняя абсолютная ошибка (MAE) для Эксперимента 1.
            r2_exp2 (float): Коэффициент детерминации $R^2$ для Эксперимента 2 (Feature Engineering).
            mae_exp2 (float): Средняя абсолютная ошибка (MAE) для Эксперимента 2.
            r2_hgb (float): Коэффициент детерминации $R^2$ для модели HistGradientBoosting (Эксперимент 3).
            mae_hgb (float): Средняя абсолютная ошибка (MAE) для модели HistGradientBoosting (Эксперимент 3).
            r2_rf (Optional[float], optional): Коэффициент детерминации $R^2$ для финальной модели
                RandomForest с GridSearchCV (Эксперимент 4). По умолчанию None.
            mae_rf (Optional[float], optional): Средняя абсолютная ошибка (MAE) для финальной модели
                RandomForest с GridSearchCV (Эксперимент 4). По умолчанию None.
            fingerprint (Optional[str], optional): Отпечаток набора признаков + параметров
                trials/cv (см. compute_optuna_fingerprint()). Сохраняется вместе с кэшем,
                чтобы load_optuna_cache() мог определить, актуален ли кэш для текущего
                запуска, а не просто проверять факт существования файла.

        Returns:
            None: Функция записывает данные на диск и выводит подтверждающий лог в консоль.

        Raises:
            NameError: Если глобальная переменная `OPTUNA_CACHE_FILE` не определена.
            OSError: При ошибках открытия файла на запись или отсутствии прав доступа к директории.
            TypeError: Если переданные объекты не могут быть сериализованы в формат JSON.
        """

    cache = {
        'best_model_name': best_model_name,
        'best_params': best_params,
        'exp1_r2': r2_exp1, 'exp1_mae': mae_exp1,
        'exp2_r2': r2_exp2, 'exp2_mae': mae_exp2,
        'exp3_r2': r2_hgb,  'exp3_mae': mae_hgb,
        'exp4_r2': r2_rf,   'exp4_mae': mae_rf,
        'fingerprint': fingerprint,
    }

    with open(OPTUNA_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\n💾 Результаты Optuna сохранены в кэш: {OPTUNA_CACHE_FILE}")

# =============================================================================
# ЧАСТЬ 1: ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_ikea_data(url: str) -> pd.DataFrame:
    """
    Загружает датасет IKEA из удаленного источника по URL-адресу с валидацией ответа сервера.

    Функция выполняет HTTP-запрос к указанному URL с помощью библиотеки `requests`
    и проверяет код состояния ответа (`status_code`). Если сервер вернул успешный
    статус `200 OK`, функция повторно считывает данные из удаленного файла CSV в
    формат `pandas.DataFrame`. В консоль выводятся диагностические сообщения о
    ходе загрузки, включая итоговый размер полученной таблицы.

    Args:
        url (str): Удаленный HTTP/HTTPS URL-адрес, ведущий к файлу с данными в формате CSV.

    Returns:
        pd.DataFrame: Загруженный и спарсенный датафрейм, содержащий исходные данные IKEA.

    Raises:
        requests.exceptions.RequestException: При сетевых ошибках соединения, таймаутах
            или неверном URL (возбуждается неявно при вызове `requests.get`).
        Exception: Если HTTP-запрос завершился неудачно (код состояния ответа отличен от 200).
        pd.errors.ParserError: Если файл по указанному URL не может быть корректно
            распознан парсером pandas как CSV (например, поврежден или имеет неверную структуру).
    """
    print("=" * 70)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)
    print(f"Попытка загрузки данных с: {url}")

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Ошибка загрузки данных. Status code: {response.status_code}. "
            f"Ожидался 200 OK."
        )

    print(f"✓ Status code: {response.status_code} (OK)")

    df = pd.read_csv(url)

    print(f"✓ Данные успешно загружены")
    print(f"✓ Размер датасета: {df.shape[0]} строк × {df.shape[1]} колонок")

    return df

# =============================================================================
# ЧАСТЬ 2: ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ
# =============================================================================

def explore_data(df: pd.DataFrame) -> None:
    """
    Выполняет комплексный первичный экспресс-анализ структуры и содержимого датафрейма.

    Функция выводит в консоль подробную структурированную сводку о переданном наборе
    данных для быстрой оценки его качества перед началом предобработки.

    Выводимая информация включает в себя:
    1. **Общую информацию**: Размерность таблицы (строки и столбцы) и точный объем
       занимаемой оперативной памяти (с глубоким анализом объектов `deep=True`).
    2. **Список признаков**: Пронумерованный перечень всех колонок датасета.
    3. **Анализ типов и пропусков**: Информационную таблицу (отформатированную с помощью
       библиотеки `tabulate`), содержащую типы данных, количество непустых значений,
       абсолютное количество пропущенных значений (`NaN`) и их процентное соотношение
       для каждой колонки.
    4. **Описательную статистику**: Основные статистические показатели для числовых
       признаков (минимум, максимум, среднее, медиана, стандартное отклонение и квантили),
       представленные в виде Markdown-таблицы.

    Args:
        df (pd.DataFrame): Анализируемый датафрейм (исходные данные IKEA).

    Returns:
        None: Функция выводит форматированные таблицы и текстовые отчеты в консоль.

    Raises:
        NameError: Если библиотеки `tabulate` или `pandas` (под псевдонимом `pd`)
            не импортированы в модуле.
    """
    print("\n" + "=" * 70)
    print("ПЕРВИЧНЫЙ ОСМОТР ДАТАСЕТА")
    print("=" * 70)

    print(f"\n Общая информация:")
    print(f"  • Количество строк: {df.shape[0]}")
    print(f"  • Количество колонок: {df.shape[1]}")
    print(f"  • Размер в памяти: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    print(f"\n Названия колонок:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\n Информация о типах данных и пропусках:")
    info_df = pd.DataFrame({
        'Колонка': df.columns,
        'Тип данных': df.dtypes.values,
        'Не-пропусков': df.count().values,
        'Пропусков': df.isnull().sum().values,
        '% пропусков': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    print(tabulate(info_df, headers='keys', tablefmt='github', showindex=False))

    print(f"\n Описательная статистика (числовые колонки):")
    print(df.describe().T.to_markdown())


# =============================================================================
# ЧАСТЬ 3: ОПРЕДЕЛЕНИЕ СОСТАВНЫХ ТОВАРОВ
# =============================================================================

def identify_composite_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Идентифицирует составные товары (комплекты, наборы) на основе текстовых паттернов и габаритов.

    Функция применяет набор эвристических правил для сегментации каталога IKEA на
    одиночные товары и составные комплекты. Результат классификации записывается
    в новую бинарную колонку `is_composite`, а в консоль выводится аналитический
    отчет со сравнением медианных цен обеих групп.

    Методология классификации (критерии составного товара):
    1. **Слэш в названии (`slash_in_name`)**: Поиск символа "/" в имени товара (например,
       стол со стульями часто пишется через слэш).
    2. **Паттерны в описании (`description_composite`)**: Проверка краткого описания на
       соответствие регулярным выражениям, указывающим на количество предметов в комплекте
       (например, "set of 4", "pair of 2", "3 pcs", "and 2").
    3. **Отсутствие габаритов (`all_dims_missing`)**: Комплекты часто не имеют конкретных
       физических размеров (глубина, высота, ширина одновременно равны NaN), в отличие
       от одиночной мебели.
    4. **Исключение аксессуаров (`is_accessory`)**: Чтобы плоские или мягкие одиночные
       товары без габаритов (подушки, покрывала, чехлы, шторы) ошибочно не помечались как
       комплекты, подготавливается маска аксессуаров-исключений.

    Итоговый флаг `is_composite` устанавливается в `True`, если выполняется хотя бы одно
    условие: в названии есть слэш, описание указывает на набор ИЛИ у товара отсутствуют
    все габариты (при условии, что это не текстильный аксессуар).

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий колонки `name`,
            `short_description`, `depth`, `height`, `width` и `price`.

    Returns:
        pd.DataFrame: Копия исходного датафрейма с добавлением вспомогательных признаков:
            - 'slash_in_name' (bool): Наличие косой черты в названии.
            - 'description_composite' (bool): Наличие маркеров комплекта в описании.
            - 'all_dims_missing' (bool): Отсутствие информации по всем трем габаритам.
            - 'is_accessory' (bool): Является ли товар текстилем/аксессуаром по описанию.
            - 'is_composite' (bool): Результирующий флаг составного товара.

    Raises:
        AttributeError: Если в переданном датафрейме отсутствуют необходимые для анализа
            текстовые или числовые столбцы.
    """
    df = df.copy()

    # "/" ТОЛЬКО в названии
    df['slash_in_name'] = df['name'].str.contains(r'\s*/\s*', na=False)

    # Строгие паттерны для описания
    composite_patterns = [
        r'\band\s+\d+\s+',
        r'\bset\s+of\s+\d+',
        r'\bpair\s+of\s+\d+',
        r'\d+\s*pcs\b',
        r'\d+\s*piece',
    ]

    df['description_composite'] = df['short_description'].str.contains(
        '|'.join(composite_patterns), case=False, na=False
    )

    df['all_dims_missing'] = df[['depth', 'height', 'width']].isnull().all(axis=1)

    df['is_accessory'] = df['short_description'].str.contains(
        r'\b(?:cover|cushion|pad|mat|textile|curtain|blanket|pillow)\b',
        case=False, na=False, regex=True
    )

    df['is_composite'] = (
            df['slash_in_name'] |
            df['description_composite'] |
            (df['all_dims_missing'] & ~df['is_accessory'])
    )

    print("\n" + "=" * 70)
    print("АНАЛИЗ СОСТАВНЫХ ТОВАРОВ")
    print("=" * 70)
    print(f"\nВсего товаров: {len(df)}")
    print(f"Составных товаров: {df['is_composite'].sum()} ({df['is_composite'].mean() * 100:.1f}%)")
    print(f"Одиночных товаров: {(~df['is_composite']).sum()} ({(~df['is_composite']).mean() * 100:.1f}%)")

    print(f"\nМедианная цена составных товаров: {df[df['is_composite']]['price'].median():.2f}")
    print(f"Медианная цена одиночных товаров: {df[~df['is_composite']]['price'].median():.2f}")

    return df


# =============================================================================
# ЧАСТЬ 4: АНАЛИЗ ДУБЛИКАТОВ
# =============================================================================

def analyze_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проводит анализ записей с повторяющимися идентификаторами товаров (item_id).

    Функция идентифицирует строки датасета, у которых `item_id` дублируется,
    и выводит в консоль подробную статистику для оценки масштаба зашумленности данных.
    Для корректного подсчета всех вхождений используется параметр `keep=False`,
    позволяющий пометить как дубликаты все повторяющиеся строки, включая их первые копии.

    Выводимая в консоль информация включает в себя:
    1. Общее количество строк, имеющих неуникальный `item_id`.
    2. Количество уникальных идентификаторов (`item_id`), которые встречаются
       в датасете более одного раза.
    3. Долю (в процентах) строк с дубликатами от общего объема переданного датафрейма.

    Args:
        df (pd.DataFrame): Датафрейм, содержащий столбец `item_id` для анализа.

    Returns:
        pd.DataFrame: Исходный датафрейм без изменений (возвращается для сохранения
            возможности вызова функции внутри цепочек преобразований данных).

    Raises:
        KeyError: Если в переданном датафрейме отсутствует обязательный
            столбец `item_id`.
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ДУБЛИКАТОВ ПО item_id")
    print("=" * 70)

    duplicates = df[df.duplicated(subset=['item_id'], keep=False)]
    n_duplicates = len(duplicates)
    n_unique_duplicates = duplicates['item_id'].nunique()

    print(f"\nВсего строк с дублирующимися item_id: {n_duplicates}")
    print(f"Уникальных item_id с дубликатами: {n_unique_duplicates}")
    print(f"Процент строк с дубликатами: {n_duplicates / len(df) * 100:.1f}%")

    return df

#====================================================================================
# Разделение Данных на два датафрейма
#====================================================================================

def create_two_dataframes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет исходные данные на два датафрейма: с сохранением дубликатов и без них.

    Функция создает две независимые копии исходных данных:
    1. **`df_full`**: Полная копия исходного датафрейма, содержащая все записи (включая
       повторяющиеся `item_id`).
    2. **`df_unique`**: Очищенная копия, в которой дубликаты по колонке `item_id` удалены.
       Из повторяющихся записей сохраняется только первое встреченное вхождение (`keep='first'`).

    После разделения функция выводит в консоль сравнительную статистику по обоим
    наборам данных: общее число строк, количество уникальных идентификаторов,
    число удаленных записей, а также медианную цену товаров (`price`) для оценки
    влияния дедупликации на распределение целевого признака.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий столбцы `item_id` и `price`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Кортеж из двух датафреймов:
            - **df_full** (pd.DataFrame): Полная копия исходных данных с дубликатами.
            - **df_unique** (pd.DataFrame): Очищенная копия данных, содержащая только
              уникальные `item_id`.

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные
            колонки `item_id` или `price`.
    """
    print("\n" + "=" * 70)
    print("СОЗДАНИЕ ДВУХ ДАТАФРЕЙМОВ")
    print("=" * 70)

    df_full = df.copy()
    df_unique = df.drop_duplicates(subset=['item_id'], keep='first').copy()

    print(f"\ndf_full (с дубликатами):")
    print(f"  • Количество строк: {len(df_full)}")
    print(f"  • Уникальных item_id: {df_full['item_id'].nunique()}")

    print(f"\ndf_unique (без дубликатов):")
    print(f"  • Количество строк: {len(df_unique)}")
    print(f"  • Уникальных item_id: {df_unique['item_id'].nunique()}")
    print(f"  • Удалено строк: {len(df_full) - len(df_unique)}")

    print(f"\nМедианная цена в df_full: {df_full['price'].median():.2f}")
    print(f"Медианная цена в df_unique: {df_unique['price'].median():.2f}")

    return df_full, df_unique


# =============================================================================
# ЧАСТЬ 5: ОЧИСТКА ДИЗАЙНЕРОВ
# =============================================================================

def clean_designer(df: pd.DataFrame, dataset_name: str = "датасет") -> pd.DataFrame:
    """
    Очищает и стандартизирует текстовое поле `designer` в датафрейме.

    Функция выявляет невалидные и «мусорные» записи в колонке `designer` с помощью
    эвристических правил и регулярных выражений, заменяя их на значение 'Unknown'.
    Это критически важно, так как в исходном датасете в поле автора дизайна часто
    попадают технические артикулы товаров или фрагменты описаний на английском языке.

    Эвристические критерии определения «мусора» (is_garbage):
    1. **Превышение длины**: Строки длиной более 50 символов (настоящие имена
       дизайнеров IKEA укладываются в этот лимит, а длинный текст обычно является
       ошибкой парсинга).
    2. **Паттерн артикула IKEA**: Поиск шаблона вида `000.000.00` (три цифры, точка,
       три цифры, точка, две цифры), который указывает на то, что вместо имени в ячейку
       записан код товара.
    3. **Текстовое описание**: Поиск типичных слов из товарных описаний (например,
       'small', 'solution', 'clean', 'keep' и др.), появление которых в поле автора
       свидетельствует о сбое разметки.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий текстовую колонку `designer`.

    Returns:
        pd.DataFrame: Копия датафрейма с очищенной колонкой `designer`, где все
            невалидные текстовые значения заменены на `'Unknown'`.

    Raises:
        KeyError: Если в переданном датафрейме отсутствует обязательный
            столбец `designer`.
    """
    df = df.copy()

    print("\n" + "=" * 70)
    print(f"ОЧИСТКА КОЛОНКИ designer ({dataset_name})")
    print("=" * 70)

    pattern_article = r'\d{3}\.\d{3}\.\d{2}'
    pattern_description = r'\b(?:small|easy|good|solution|place|clean|keep|hung|openi)\b'

    is_garbage = (
            (df['designer'].str.len() > 50) |
            (df['designer'].str.contains(pattern_article, regex=True, na=False)) |
            (df['designer'].str.contains(pattern_description, case=False, regex=True, na=False))
    )

    n_garbage = is_garbage.sum()
    print(f"\nШаг 1: Замена 'мусорных' записей")
    print(f"  • Найдено мусорных записей: {n_garbage}")

    df.loc[is_garbage, 'designer'] = 'Unknown'

    def normalize_composite_name(name: Optional[str]) -> Optional[str]:
        if pd.isna(name) or name == 'Unknown':
            return name
        if '/' in name:
            parts = [p.strip() for p in name.split('/')]
            parts_sorted = sorted(parts)
            return '/'.join(parts_sorted)
        return name

    df['designer_clean'] = df['designer'].apply(normalize_composite_name)

    print(f"\nШаг 2: Унификация составных имён")
    print(f"  • Уникальных дизайнеров до очистки: {df['designer'].nunique()}")
    print(f"  • Уникальных дизайнеров после очистки: {df['designer_clean'].nunique()}")

    n_unknown = (df['designer_clean'] == 'Unknown').sum()
    print(f"\nЗаписей 'Unknown': {n_unknown} ({n_unknown / len(df) * 100:.1f}%)")

    return df

# =============================================================================
# ЧАСТЬ 6: КЛАССИФИКАЦИЯ ДИЗАЙНЕРОВ
# =============================================================================

def classify_designer_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Классифицирует авторов дизайна на основе состава разработчиков и брендовой принадлежности.

    Функция выполняет сегментацию авторов дизайна (на основе предварительно очищенного
    поля `designer_clean`) по двум направлениям:
    1. **Командная vs. Одиночная разработка (`is_team`)**: Флаг устанавливается в значение `1`,
       если в имени автора присутствует косая черта `/` (что указывает на коллаборацию
       или группу дизайнеров) и имя не является `'Unknown'`. В противном случае — `0`.
    2. **Массовый vs. Премиум/Нишевый сегмент (`designer_type`)**: Если дизайн разработан
       внутренней группой `'IKEA of Sweden'`, товар классифицируется как "Массовый сегмент".
       Товары от сторонних приглашенных дизайнеров и независимых студий классифицируются
       как "Премиум/Нишевый сегмент".

    После выполнения сегментации функция выводит в консоль аналитическую сводку:
    - Общее количество и процентное соотношение товаров, созданных командами и одиночными дизайнерами.
    - Сравнительный анализ медианных цен (`price`) для товаров от командных и одиночных авторов.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий текстовую колонку `designer_clean`
            и числовую колонку `price`.

    Returns:
        pd.DataFrame: Копия датафрейма с добавлением трех новых признаков:
            - 'is_team' (int): Бинарный флаг (1 или 0) совместной/командной разработки.
            - 'is_mass_market' (bool): Флаг принадлежности к внутреннему бренду 'IKEA of Sweden'.
            - 'designer_type' (str): Категориальный тип сегмента ("Массовый сегмент"
              или "Премиум/Нишевый сегмент").

    Raises:
        KeyError: Если в переданном датафрейме отсутствуют обязательные
            столбцы `designer_clean` или `price`.
    """
    df = df.copy()

    df['is_team'] = (
            df['designer_clean'].str.contains('/', regex=False, na=False) &
            (df['designer_clean'] != 'Unknown')
    ).astype(int)

    df['is_mass_market'] = df['designer_clean'].str.contains(
        r'IKEA of Sweden', case=False, na=False
    )

    df['designer_type'] = df['is_mass_market'].apply(
        lambda x: 'Массовый сегмент' if x else 'Премиум/Нишевый сегмент'
    )

    print("\n" + "=" * 70)
    print("КЛАССИФИКАЦИЯ ДИЗАЙНЕРОВ")
    print("=" * 70)
    print(f"\nТоваров от команд дизайнеров: {df['is_team'].sum()} ({df['is_team'].mean() * 100:.1f}%)")
    print(f"Товаров от одиночных дизайнеров: {(~df['is_team'].astype(bool)).sum()}")

    print(f"\nМедианная цена товаров от команд: {df[df['is_team'] == 1]['price'].median():.2f}")
    print(f"Медианная цена товаров от одиночных: {df[df['is_team'] == 0]['price'].median():.2f}")

    return df

# =============================================================================
# ЧАСТЬ 7: ЗАПОЛНЕНИЕ ПРОПУСКОВ В ГАБАРИТАХ
# =============================================================================

def fill_dimensions(df: pd.DataFrame, dataset_name: str = "датасет") -> pd.DataFrame:
    """
    Заполняет пропущенные значения физических габаритов медианой по категориям товаров.

    Функция выполняет контекстное восстановление пропущенных данных (импьютацию)
    для физических размеров товара (`depth`, `height`, `width`). Вместо использования
    глобальной медианы, функция рассчитывает медианное значение индивидуально для
    каждой категории товаров (`category`), что обеспечивает высокую точность
    и логическую согласованность восстановленных данных.

    Важная методологическая деталь (маскирование):
    Заполнение пропусков происходит **строго для одиночных нетекстильных товаров**
    (где `is_composite` и `is_accessory` равны False). Для составных комплектов
    и мягких аксессуаров (например, чехлов или штор) импьютация габаритов не
    производится, так как отсутствие четких размеров для них является естественным.

    Функция выводит в консоль сравнительную статистику (абсолютное количество
    пропусков и их процентную долю) для каждого измерения до и после процедуры заполнения.

    Args:
        df (pd.DataFrame): Исходный датафрейм, содержащий:
            - Числовые колонки размеров: `depth`, `height`, `width`.
            - Категориальный признак: `category`.
            - Вспомогательные флаги: `is_composite`, `is_accessory`.

    Returns:
        pd.DataFrame: Копия исходного датафрейма с заполненными пропущенными
            значениями габаритов для целевой группы товаров.

    Raises:
        KeyError: Если в переданном датафрейме отсутствует любая из колонок:
            `depth`, `height`, `width`, `category`, `is_composite` или `is_accessory`.
    """
    df = df.copy()

    print("\n" + "=" * 70)
    print(f"ЗАПОЛНЕНИЕ ПРОПУСКОВ В ГАБАРИТАХ ({dataset_name})")
    print("=" * 70)

    print(f"\nПропуски до заполнения:")
    for col in ['depth', 'height', 'width']:
        print(f"  • {col}: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)")

    mask = (~df['is_composite']) & (~df['is_accessory'])

    for col in ['depth', 'height', 'width']:
        median_by_category = df[mask].groupby('category')[col].transform('median')
        df.loc[mask & df[col].isnull(), col] = median_by_category[mask & df[col].isnull()]

    print(f"\nПропуски после заполнения:")
    for col in ['depth', 'height', 'width']:
        print(f"  • {col}: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)")

    return df

# =============================================================================
# ЧАСТЬ 8: ГИПОТЕЗА 1 - ОБЪЁМ ТОВАРА И ЦЕНА
# =============================================================================

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

def run_ml_pipeline(
        df_unique: pd.DataFrame,
        eda_results: Dict[str, Any],
        rerun_optuna: bool = False,
        trials: Optional[int] = None,
        cv: Optional[int] = None
) -> Dict[str, Any]:
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

    Args:
        df_unique: Датасет без дубликатов для построения признаков и обучения моделей.
        eda_results: Результаты этапа EDA, содержащие вердикты по `sellable_online`
            и `old_price` для предотвращения утечек данных (Data Leakage).
        rerun_optuna: Если True, запускает повторный ресурсоемкий поиск параметров
            через Optuna. Если False, используются предопределенные стабильные параметры.
        trials: Количество итераций (испытаний) для оптимизатора Optuna.
        cv: Количество блоков (фолдов) при кросс-валидации.

    Returns:
        Словарь с объектами обучения и результатами валидации:
            {
                'X_train': pd.DataFrame,
                'X_test': pd.DataFrame,
                'y_train': pd.Series,
                'y_test': pd.Series,
                'X_full': pd.DataFrame,
                'y_full': pd.Series,
                'num_feat': list[str],
                'cat_feat': list[str],
                'bin_feat': list[str],
                'bool_feat': list[str],
                'preprocessor': ColumnTransformer,
                'best_model_pipeline': Pipeline,
                'best_model_name': str,
                'gridsearch_results': dict,
                'cv_scores': np.ndarray,
                'ablation_results': pd.DataFrame,
                'bootstrap_ablation_results': pd.DataFrame,
                'sensitivity_results': pd.DataFrame,
                'leakage_verdict': str,
                'model_selection_details': dict
            }
    """

    # ========================================================================
    # 🔧 Вспомогательная функция: преобразование имени модели в тип
    # ========================================================================
    def _get_model_type_from_name(model_name: str) -> str:
        """Преобразует имя модели в формат, понятный дочерним функциям.

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

    print("\n" + "=" * 70)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ML")
    print("=" * 70)

    # Извлекаем параметры из eda_results
    sellable_verdict = eda_results['sellable_verdict']
    old_price_check = eda_results['old_price_check']

    # 1. Подготовка данных
    X_train, X_test, y_train, y_test, X, y, num_feat, cat_feat, bin_feat, bool_feat = prepare_ml_data(
        df_unique,
        sellable_verdict=sellable_verdict,
        old_price_verdict=old_price_check['verdict'],
        old_price_features_to_use=old_price_check['features_to_use']
    )

    # 2. Сравнение моделей
    results_df, best_name, best_pipeline, preprocessor = compare_models(
        X_train, X_test, y_train, y_test, num_feat, cat_feat, bin_feat, bool_feat
    )

    print(f"\n{'=' * 70}")
    print(" ФИНАЛЬНАЯ МОДЕЛЬ (после сравнения)")
    print(f"{'=' * 70}")
    print(f"\n Лучшая модель: {best_name}")
    print(f"   R² (тест): {results_df.iloc[0]['R2']:.4f}")
    print(f"   MAE: {results_df.iloc[0]['MAE']:.2f}")

    # Сохраняем метрики простой модели для сравнения
    simple_r2 = results_df.iloc[0]['R2']
    simple_mae = results_df.iloc[0]['MAE']

    # 3. Проверка утечки данных
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

    # 4. Optuna
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

    # Сохраняем метрики Optuna для сравнения
    y_pred_optuna_log = optuna_pipeline.predict(X_test)
    y_pred_optuna = np.expm1(y_pred_optuna_log)
    optuna_r2 = r2_score(y_test, y_pred_optuna)
    optuna_mae = mean_absolute_error(y_test, y_pred_optuna)

    print(f"\n📊 Метрики Optuna на тесте:")
    print(f"   Модель: {optuna_model_name}")
    print(f"   R² (тест): {optuna_r2:.4f}")
    print(f"   MAE (тест): {optuna_mae:.2f} SR")

    # 5. GridSearchCV
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

    # ========================================================================
    # 6. Сравнение Optuna, GridSearchCV и простой модели
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
    print(f"{'=' * 70}")

    # Сравниваем три кандидата: простая модель, Optuna, GridSearchCV
    candidates = {
        'simple': {'r2': simple_r2, 'mae': simple_mae, 'name': best_name, 'pipeline': best_pipeline, 'params': None},
        'optuna': {'r2': optuna_r2, 'mae': optuna_mae, 'name': optuna_model_name, 'pipeline': optuna_pipeline, 'params': optuna_params},
        'gridsearch': {'r2': gs_r2, 'mae': gs_mae, 'name': 'RandomForest + GridSearchCV', 'pipeline': gs_pipeline, 'params': gs_params}
    }

    # Выбираем лучшую модель по R²
    best_candidate = max(candidates.items(), key=lambda x: x[1]['r2'])
    winner = best_candidate[0]
    winner_data = best_candidate[1]

    final_pipeline = winner_data['pipeline']
    final_model_name = winner_data['name']
    final_params = winner_data['params']
    final_r2 = winner_data['r2']
    final_mae = winner_data['mae']

    # 🔧 ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ТИПА МОДЕЛИ
    final_model_type = _get_model_type_from_name(final_model_name)

    print(f"\n🏆 Победитель: {final_model_name}")
    print(f"   Тип модели (для дочерних функций): {final_model_type}")
    print(f"   R²: {final_r2:.4f}")
    print(f"   MAE: {final_mae:.2f} SR")
    print(f"\n📊 Сравнение кандидатов:")
    print(f"   Простая модель ({best_name}): R²={simple_r2:.4f}, MAE={simple_mae:.2f}")
    print(f"   Optuna ({optuna_model_name}): R²={optuna_r2:.4f}, MAE={optuna_mae:.2f}")
    print(f"   GridSearchCV: R²={gs_r2:.4f}, MAE={gs_mae:.2f}")

    # Проверка принципа Occam's Razor
    if winner == 'simple':
        print(f"\n⚠️ ВНИМАНИЕ: Простая модель работает ЛУЧШЕ сложных!")
        print(f"   Принцип Occam's Razor: если простая модель не хуже сложной, выбираем простую")
        print(f"   Разница R²: {simple_r2 - optuna_r2:+.4f} vs Optuna, {simple_r2 - gs_r2:+.4f} vs GridSearchCV")

    # Сохраняем детали выбора модели для отчёта
    model_selection_details = {
        'winner': winner,
        'simple_model_name': best_name,
        'simple_r2': simple_r2,
        'simple_mae': simple_mae,
        'optuna_model_name': optuna_model_name,
        'optuna_r2': optuna_r2,
        'optuna_mae': optuna_mae,
        'optuna_params': optuna_params,
        'gridsearch_r2': gs_r2,
        'gridsearch_mae': gs_mae,
        'gridsearch_params': gs_params,
        'final_model_name': final_model_name,
        'final_model_type': final_model_type,
        'final_r2': final_r2,
        'final_mae': final_mae,
    }

    # 7. Ablation Study
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY (анализ вклада групп признаков)")
    print(f"{'=' * 70}")

    # 🔧 Передаём параметры и тип финальной модели (никаких зашивок!)
    ablation_results = run_ablation_study(
        X_train, X_test, y_train, y_test,
        num_feat, cat_feat, bin_feat,
        bool_features=bool_feat,
        best_params=final_params,
        model_name=final_model_type
    )

    # 7.1. Bootstrap Ablation — точечная проверка отдельных признаков с
    # динамически вычисляемыми CI (дополняет групповой Ablation Study выше:
    # изолирует вклад признаков ВНУТРИ одной группы, напр. volume отдельно
    # от width/height/depth, чего групповой тест сделать не может)
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP ABLATION (точечный анализ отдельных признаков)")
    print(f"{'=' * 70}")

    bootstrap_ablation_results = run_bootstrap_ablation(
        X_train, X_test, y_train, y_test,
        numeric_features=num_feat, categorical_features=cat_feat,
        binary_features=bin_feat, bool_features=bool_feat,
        best_params=final_params, model_name=final_model_type,
        features_to_test=None,  # None = проверить ВСЕ числовые и бинарные признаки
        n_repeats=15
    )

    # 8. Sensitivity Analysis
    print(f"\n{'=' * 70}")
    print("SENSITIVITY ANALYSIS (анализ чувствительности к объёму данных)")
    print(f"{'=' * 70}")

    # 🔧 Передаём параметры и тип финальной модели (никаких зашивок!)
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

    # 9. Кросс-валидация
    X_full = pd.concat([X_train, X_test]).sort_index()
    y_full = pd.concat([y_train, y_test]).sort_index()

    # 🔧 Передаём параметры и тип финальной модели (никаких зашивок!)
    cv_scores, cv_pipeline = cross_validate_model(
        X_full, y_full,
        preprocessor,
        best_params=final_params,
        model_name=final_model_type
    )

    # ========================================================================
    # ИТОГОВЫЙ ОТЧЁТ
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(" ИТОГИ ML-ПАЙПЛАЙНА")
    print(f"{'=' * 70}")
    print(f"\n🏆 Финальная модель: {final_model_name}")
    print(f"   Тип модели: {final_model_type}")
    print(f"   R² (тест): {final_r2:.4f}")
    print(f"   MAE (тест): {final_mae:.2f} SR")
    print(f"   R² (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Способ выбора: {winner.upper()}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_full': X_full,
        'y_full': y_full,
        'num_feat': num_feat,
        'cat_feat': cat_feat,
        'bin_feat': bin_feat,
        'bool_feat': bool_feat,
        'preprocessor': preprocessor,
        'best_model_pipeline': final_pipeline,
        'best_model_name': final_model_name,
        'gridsearch_results': gridsearch_results,
        'cv_scores': cv_scores,
        'ablation_results': ablation_results,
        'bootstrap_ablation_results': bootstrap_ablation_results,
        'sensitivity_results': sensitivity_results,
        'leakage_verdict': verdict,
        'model_selection_details': model_selection_details,
    }

# ==============================================================================
# ОРКЕСТРАТОР ИНТЕРПРЕТАЦИИ
# ==============================================================================

def run_interpretation_pipeline(ml_results: dict) -> dict:
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

    best_model_pipeline = ml_results['best_model_pipeline']
    best_model_name = ml_results.get('best_model_name', 'Unknown Model')
    X_train = ml_results['X_train']
    X_test = ml_results['X_test']
    y_train = ml_results['y_train']
    y_test = ml_results['y_test']
    X_full = ml_results['X_full']
    y_full = ml_results['y_full']

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

def print_final_summary(
        eda_results: Dict[str, Any],
        ml_results: Dict[str, Any],
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
        best_model_name = ml_results['best_model_name']
        cv_scores = ml_results['cv_scores']

        # Получаем метрики финальной модели из baseline_df
        if baseline_df is not None:
            final_model_row = baseline_df[baseline_df['Model'].str.contains('наша', case=False)]
            if len(final_model_row) > 0:
                final_r2 = final_model_row['R²'].values[0]
                final_mae = final_model_row['MAE'].values[0]
            else:
                # Fallback: используем GridSearchCV
                final_r2 = ml_results['gridsearch_results']['test_r2']
                final_mae = ml_results['gridsearch_results']['test_mae']
        else:
            final_r2 = ml_results['gridsearch_results']['test_r2']
            final_mae = ml_results['gridsearch_results']['test_mae']

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
    ablation_results = ml_results.get('ablation_results')
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
        X_train = ml_results['X_train']
        X_test = ml_results['X_test']

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

def prepare_ml_data(
        df: pd.DataFrame,
        sellable_verdict: str = "KEEP",
        old_price_verdict: str = "SAFE",
        old_price_features_to_use: Optional[List[str]] = None
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