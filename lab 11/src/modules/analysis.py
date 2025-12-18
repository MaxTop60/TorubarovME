"""
Экспериментальное исследование алгоритмов поиска подстроки.
Сравнение времени выполнения, анализ худшего и лучшего случаев.
Визуализация результатов.
"""

import timeit
import random
import string
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable

# Импортируем наши алгоритмы
from prefix_function import prefix_function
from kmp_search import kmp_search
from z_function import z_function, search_with_z_function
from string_matching import boyer_moore_search, boyer_moore_search_with_good_suffix


def generate_random_string(length: int, alphabet: str = None) -> str:
    """Генерация случайной строки заданной длины."""
    if alphabet is None:
        alphabet = string.ascii_lowercase
    return "".join(random.choice(alphabet) for _ in range(length))


def measure_time(func: Callable, *args, number: int = 100) -> float:
    """Измерение времени выполнения функции."""
    timer = timeit.Timer(lambda: func(*args))
    return timer.timeit(number=number) / number


def compare_by_text_length():
    """Сравнение времени выполнения от длины текста."""
    print("=" * 60)
    print("СРАВНЕНИЕ ПО ДЛИНЕ ТЕКСТА")
    print("=" * 60)

    # Фиксированный паттерн
    pattern = "abcde"

    # Длины текстов для тестирования
    text_lengths = [100, 500, 1000, 2500, 5000, 10000]

    # Алгоритмы для сравнения
    algorithms = [
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Boyer-Moore (с суффиксом)", boyer_moore_search_with_good_suffix),
        ("Z-функция", search_with_z_function),
    ]

    results = {name: [] for name, _ in algorithms}

    print("\nИзмерение времени выполнения...")

    for length in text_lengths:
        print(f"Длина текста: {length}")
        text = generate_random_string(length)

        for name, algorithm in algorithms:
            try:
                time_taken = measure_time(algorithm, text, pattern, number=10)
                results[name].append(time_taken)
                print(f"  {name}: {time_taken:.6f} сек")
            except Exception as e:
                results[name].append(float("inf"))
                print(f"  {name}: ошибка - {e}")

    # Построение графика
    plt.figure(figsize=(12, 6))

    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (name, values) in enumerate(results.items()):
        plt.plot(
            text_lengths,
            values,
            marker="o",
            linewidth=2,
            label=name,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Длина текста")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Зависимость времени выполнения от длины текста")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("text_length_comparison.png", dpi=300)
    print("\nГрафик сохранен как 'text_length_comparison.png'")


def compare_by_pattern_length():
    """Сравнение времени выполнения от длины паттерна."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПО ДЛИНЕ ПАТТЕРНА")
    print("=" * 60)

    # Фиксированный текст
    text = generate_random_string(10000)

    # Длины паттернов для тестирования
    pattern_lengths = [1, 5, 10, 20, 50, 100]

    # Алгоритмы для сравнения
    algorithms = [
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Boyer-Moore (с суффиксом)", boyer_moore_search_with_good_suffix),
        ("Z-функция", search_with_z_function),
    ]

    results = {name: [] for name, _ in algorithms}

    print("\nИзмерение времени выполнения...")

    for length in pattern_lengths:
        print(f"Длина паттерна: {length}")
        pattern = generate_random_string(length)

        for name, algorithm in algorithms:
            try:
                time_taken = measure_time(algorithm, text, pattern, number=10)
                results[name].append(time_taken)
                print(f"  {name}: {time_taken:.6f} сек")
            except Exception as e:
                results[name].append(float("inf"))
                print(f"  {name}: ошибка - {e}")

    # Построение графика
    plt.figure(figsize=(12, 6))

    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (name, values) in enumerate(results.items()):
        plt.plot(
            pattern_lengths,
            values,
            marker="s",
            linewidth=2,
            label=name,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Длина паттерна")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Зависимость времени выполнения от длины паттерна")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pattern_length_comparison.png", dpi=300)
    print("\nГрафик сохранен как 'pattern_length_comparison.png'")


def analyze_worst_case():
    """Анализ худшего случая для алгоритмов."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ ХУДШЕГО СЛУЧАЯ")
    print("=" * 60)

    # Худший случай для наивного алгоритма и KMP
    # Текст: aaaa...aaa
    # Паттерн: aaa...aaab (где последний символ отличается)

    text_length = 1000
    pattern_length = 100

    text_worst = "a" * text_length
    pattern_worst = "a" * (pattern_length - 1) + "b"

    algorithms = [
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Z-функция", search_with_z_function),
    ]

    print("\nХудший случай (много совпадений, но нет полных):")
    print(f"Текст: {text_length} символов 'a'")
    print(f"Паттерн: {pattern_length-1} символов 'a' + 'b'")

    results = []
    for name, algorithm in algorithms:
        try:
            time_taken = measure_time(algorithm, text_worst, pattern_worst, number=10)
            results.append((name, time_taken))
            print(f"  {name}: {time_taken:.6f} сек")
        except Exception as e:
            print(f"  {name}: ошибка - {e}")

    # Построение графика
    plt.figure(figsize=(10, 6))

    names = [r[0] for r in results]
    times = [r[1] for r in results]

    bars = plt.bar(names, times, color=["blue", "red", "green", "orange"])

    plt.xlabel("Алгоритм")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Время выполнения в худшем случае")
    plt.grid(True, alpha=0.3, axis="y")

    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.00001,
            f"{time_val:.6f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("worst_case_analysis.png", dpi=300)
    print("\nГрафик сохранен как 'worst_case_analysis.png'")


def analyze_best_case():
    """Анализ лучшего случая для алгоритмов."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЛУЧШЕГО СЛУЧАЯ")
    print("=" * 60)

    # Лучший случай для Boyer-Moore
    # Первый символ паттерна не совпадает с первым символом текста
    # Это позволяет делать большие сдвиги

    text_length = 1000
    pattern_length = 100

    text_best = "z" + "a" * (text_length - 1)  # начинается с z
    pattern_best = "b" + "a" * (pattern_length - 1)  # начинается с b

    algorithms = [
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Boyer-Moore (с суффиксом)", boyer_moore_search_with_good_suffix),
        ("Z-функция", search_with_z_function),
    ]

    print("\nЛучший случай (быстрые сдвиги):")
    print(f"Текст: 'z' + {text_length-1} символов 'a'")
    print(f"Паттерн: 'b' + {pattern_length-1} символов 'a'")

    results = []
    for name, algorithm in algorithms:
        try:
            time_taken = measure_time(algorithm, text_best, pattern_best, number=10)
            results.append((name, time_taken))
            print(f"  {name}: {time_taken:.6f} сек")
        except Exception as e:
            print(f"  {name}: ошибка - {e}")

    # Построение графика
    plt.figure(figsize=(10, 6))

    names = [r[0] for r in results]
    times = [r[1] for r in results]

    bars = plt.bar(names, times, color=["blue", "red", "green", "orange", "purple"])

    plt.xlabel("Алгоритм")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Время выполнения в лучшем случае")
    plt.grid(True, alpha=0.3, axis="y")

    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.00001,
            f"{time_val:.6f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("best_case_analysis.png", dpi=300)
    print("\nГрафик сохранен как 'best_case_analysis.png'")


def visualize_prefix_and_z_functions():
    """Визуализация префикс-функции и Z-функции."""
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ПРЕФИКС-ФУНКЦИИ И Z-ФУНКЦИИ")
    print("=" * 60)

    # Примеры строк для визуализации
    examples = [
        ("abacaba", "Периодическая строка 'abacaba'"),
        ("aaaaa", "Строка из одинаковых символов"),
        ("abcabc", "Периодический паттерн"),
        ("ababab", "Чередующиеся символы"),
    ]

    for s, description in examples:
        print(f"\nСтрока: '{s}' - {description}")

        # Вычисляем функции
        pi = prefix_function(s)
        z = z_function(s)

        print(f"Префикс-функция: {pi}")
        print(f"Z-функция: {z}")

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Префикс-функция
        indices = list(range(len(s)))
        ax1.bar(indices, pi, color="blue", alpha=0.7)
        ax1.set_xlabel("Индекс")
        ax1.set_ylabel("Значение π[i]")
        ax1.set_title(f'Префикс-функция для строки "{s}"')
        ax1.grid(True, alpha=0.3)

        # Добавляем символы на график
        for i, char in enumerate(s):
            ax1.text(i, pi[i] + 0.1, char, ha="center", va="bottom")

        # Z-функция
        ax2.bar(indices, z, color="red", alpha=0.7)
        ax2.set_xlabel("Индекс")
        ax2.set_ylabel("Значение z[i]")
        ax2.set_title(f'Z-функция для строки "{s}"')
        ax2.grid(True, alpha=0.3)

        # Добавляем символы на график
        for i, char in enumerate(s):
            ax2.text(i, z[i] + 0.1, char, ha="center", va="bottom")

        plt.tight_layout()
        filename = f"prefix_z_function_{s[:10]}.png".replace(" ", "_")
        plt.savefig(filename, dpi=300)
        print(f"  График сохранен как '{filename}'")


def analyze_periodic_patterns():
    """Анализ производительности на периодических паттернах."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ ПЕРИОДИЧЕСКИХ ПАТТЕРНОВ")
    print("=" * 60)

    text = "ab" * 1000  # Периодический текст
    pattern_lengths = [2, 4, 8, 16, 32]

    algorithms = [
        ("KMP", kmp_search),
        ("Boyer-Moore", boyer_moore_search),
        ("Z-функция", search_with_z_function),
    ]

    results = {name: [] for name, _ in algorithms}

    print("\nИзмерение времени на периодическом тексте 'ab'*1000:")

    for length in pattern_lengths:
        print(f"Длина паттерна: {length}")
        pattern = "ab" * (length // 2)
        if length % 2:
            pattern += "a"

        for name, algorithm in algorithms:
            try:
                time_taken = measure_time(algorithm, text, pattern, number=10)
                results[name].append(time_taken)
                print(f"  {name}: {time_taken:.6f} сек")
            except Exception as e:
                results[name].append(float("inf"))
                print(f"  {name}: ошибка - {e}")

    # Построение графика
    plt.figure(figsize=(12, 6))

    colors = ["blue", "red", "green"]
    for i, (name, values) in enumerate(results.items()):
        plt.plot(
            pattern_lengths,
            values,
            marker="d",
            linewidth=2,
            label=name,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Длина паттерна")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Производительность на периодических паттернах")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("periodic_patterns_analysis.png", dpi=300)
    print("\nГрафик сохранен как 'periodic_patterns_analysis.png'")


def run_complete_analysis():
    """Запуск полного анализа всех алгоритмов."""
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ АЛГОРИТМОВ ПОИСКА ПОДСТРОКИ")
    print("=" * 80)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Устанавливаем seed для воспроизводимости
    random.seed(42)

    # Выполняем все анализы
    compare_by_text_length()
    compare_by_pattern_length()
    analyze_worst_case()
    analyze_best_case()
    visualize_prefix_and_z_functions()
    analyze_periodic_patterns()

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("Все графики сохранены в текущей директории")
    print("=" * 80)


if __name__ == "__main__":
    run_complete_analysis()
