"""
ikea_core.py — конфигурация, константы, dataclass результата, логгер,
служебные функции (save_plot, get_preprocessor) и кэш Optuna.
Часть модульного разбиения Step_project_Q14 (было: один файл 9946 строк).
"""
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
import dataclasses
from dataclasses import dataclass, field

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

# =============================================================================
# СТРУКТУРЫ ДАННЫХ (dataclass вместо словаря — рекомендация куратора)
# =============================================================================

@dataclass
class MLPipelineResult:
    """Результат работы run_ml_pipeline() — типобезопасная замена словаря.

    🔧 РЕФАКТОРИНГ: раньше run_ml_pipeline() возвращал plain dict с 17 ключами,
    доступ к которым (ml_results['key']) не проверяется на этапе написания
    кода — опечатка в имени ключа превращается в KeyError только во время
    выполнения, и то не сразу, а в момент первого обращения к этому ключу
    (может быть далеко от места самой опечатки). Dataclass даёт то же самое
    удобство передачи данных одним объектом, но с проверкой имён атрибутов
    через автодополнение IDE и понятный AttributeError сразу в месте опечатки.

    Все поля обязательны (без значений по умолчанию) — ровно те же 17 полей,
    что были ключами словаря, просто теперь это `.attribute`, а не `['key']`.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series
    num_feat: List[str]
    cat_feat: List[str]
    bin_feat: List[str]
    bool_feat: List[str]
    preprocessor: ColumnTransformer
    best_model_pipeline: Pipeline
    best_model_name: str
    gridsearch_results: Dict[str, Any]
    cv_scores: np.ndarray
    ablation_results: pd.DataFrame
    bootstrap_ablation_results: pd.DataFrame
    sensitivity_results: pd.DataFrame
    leakage_verdict: str
    model_selection_details: Dict[str, Any]

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

