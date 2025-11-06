import timeit
from sorts import (bubble_sort, selection_sort, insertion_sort, merge_sort,
                   quick_sort)
from generate_data import (
    generate_random_array,
    generate_sorted_array,
    generate_reversed_array,
    generate_almost_sorted_array,
)

# Список алгоритмов для тестирования
algorithms = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick": quick_sort,
}


def test_one_algorithm(algo_name, algo_func, arr):
    """Тестирует один алгоритм на одном массиве"""
    timer = timeit.Timer(lambda: algo_func(arr.copy()))
    time_taken = timer.timeit(number=1)
    return time_taken


def run_all_tests():
    """Запускает все тесты"""
    sizes = [100, 500, 1000, 2000]
    results = {}

    for algo_name, algo_func in algorithms.items():
        print(f"Тестируем {algo_name}...")
        results[algo_name] = {}

        for size in sizes:
            # Генерируем тестовые данные
            random_arr = generate_random_array(size)
            sorted_arr = generate_sorted_array(size)
            reversed_arr = generate_reversed_array(size)
            almost_arr = generate_almost_sorted_array(size)

            # Тестируем на всех типах данных
            time_random = test_one_algorithm(algo_name, algo_func, random_arr)
            time_sorted = test_one_algorithm(algo_name, algo_func, sorted_arr)
            time_reversed = test_one_algorithm(algo_name, algo_func,
                                               reversed_arr)
            time_almost = test_one_algorithm(algo_name, algo_func, almost_arr)

            results[algo_name][size] = {
                "random": time_random,
                "sorted": time_sorted,
                "reversed": time_reversed,
                "almost": time_almost,
            }

            print(f"  size {size}: {time_random:.4f}s")

    return results


def print_results(results):
    """Печатает результаты в виде таблицы"""
    print("\nРЕЗУЛЬТАТЫ:")
    print(
        f"{'Algorithm':<10} | {'Size':<4} | {'Random':<10} | {'Sorted':<10} | {'Reversed':<10} | {'Almost':<10}"
    )
    print("-" * 75)

    for algo_name in algorithms:
        for size in [100, 500, 1000, 2000]:
            times = results[algo_name][size]
            print(
                f"{algo_name:<10} | {size:<4} | {times['random']:8.4f}s | {times['sorted']:8.4f}s | {times['reversed']:8.4f}s | {times['almost']:8.4f}s"
            )


def analyze_theoretical_vs_practical(results: dict):
    """
    Анализ соответствия теоретических оценок практическим результатам
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ: ТЕОРЕТИЧЕСКИЕ ОЦЕНКИ VS ПРАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    theoretical_complexity = {
        "bubble": {"best": "O(n)", "avg": "O(n²)", "worst": "O(n²)"},
        "selection": {"best": "O(n²)", "avg": "O(n²)", "worst": "O(n²)"},
        "insertion": {"best": "O(n)", "avg": "O(n²)", "worst": "O(n²)"},
        "merge": {"best": "O(n log n)", "avg": "O(n log n)", "worst": "O(n log n)"},
        "quick": {"best": "O(n log n)", "avg": "O(n log n)", "worst": "O(n²)"},
    }

    for algo_name in results.keys():
        print(f"\n{algo_name}:")
        print(f"  Теоретическая сложность: {theoretical_complexity[algo_name]}")

        # Проверка поведения на отсортированных данных
        if (
            1000 in results[algo_name]
            and "sorted" in results[algo_name][1000]
            and "random" in results[algo_name][1000]
        ):

            sorted_time = results[algo_name][1000]["sorted"]
            random_time = results[algo_name][1000]["random"]
            ratio = sorted_time / random_time if random_time > 0 else 0

            print(f"  Отношение время(отсорт)/время(случ) при n=1000: {ratio:.3f}")

            if ratio < 0.5:
                print("  → Хорошо работает на отсортированных данных")
            elif ratio > 2:
                print("  → Плохо работает на отсортированных данных")
