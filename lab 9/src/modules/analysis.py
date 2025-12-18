import timeit
import random
import matplotlib.pyplot as plt
from dynamic_programming import (
    knapsack_01,
    knapsack_01_optimized,
    knapsack_01_with_items,
    longest_common_subsequence,
    levenshtein_distance,
    levenshtein_distance_optimized,
)

# ====================== Вспомогательные функции ======================


def print_dp_table(dp, title="Таблица ДП"):
    """Выводит таблицу DP на экран."""
    print(f"\n{title}")
    print("-" * 50)

    if not dp or not dp[0]:
        print("Таблица пуста")
        return

    rows = len(dp)
    cols = len(dp[0])

    for i in range(rows):
        row_str = ""
        for j in range(cols):
            row_str += f"{dp[i][j]:3d} "
        print(f"Строка {i:2d}: {row_str}")
    print("-" * 50)


def generate_knapsack_data(n, max_weight=50, max_value=100):
    """Генерирует данные для задачи о рюкзаке."""
    weights = [random.randint(1, max_weight) for _ in range(n)]
    values = [random.randint(1, max_value) for _ in range(n)]
    capacity = random.randint(max_weight, max_weight * 3)
    return weights, values, capacity


def generate_string_pair(n, alphabet="abcdefghijklmnopqrstuvwxyz"):
    """Генерирует пару строк для LCS и Левенштейна."""
    str1 = "".join(random.choice(alphabet) for _ in range(n))
    str2 = "".join(random.choice(alphabet) for _ in range(n))
    return str1, str2


# ====================== Визуализация таблиц ДП ======================


def visualize_knapsack_dp():
    """Визуализирует процесс заполнения таблицы ДП для задачи о рюкзаке."""
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ТАБЛИЦЫ ДП ДЛЯ ЗАДАЧИ О РЮКЗАКЕ")
    print("=" * 70)

    # Простой пример для наглядности
    weights = [2, 3, 4]
    values = [3, 4, 5]
    capacity = 5

    print(f"\nВходные данные:")
    print(f"Веса предметов: {weights}")
    print(f"Стоимости предметов: {values}")
    print(f"Вместимость рюкзака: {capacity}")

    n = len(weights)

    # Создаем и заполняем таблицу вручную для наглядности
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    print(f"\nНачальное состояние таблицы (все нули):")
    print_dp_table(dp, "Таблица ДП (начало)")

    # Поэтапное заполнение
    for i in range(1, n + 1):
        print(
            f"\nОбработка предмета {i} (вес={weights[i-1]}, стоимость={
                values[i-1]}):"
        )

        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]

        print_dp_table(dp, f"Таблица после предмета {i}")

    # Восстановление решения
    print(f"\nВосстановление решения:")
    selected_items = []
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
            print(f"  Предмет {i} выбран, осталось места: {w}")

    selected_items.reverse()
    max_value = dp[n][capacity]

    print(f"\nРезультат:")
    print(f"Максимальная стоимость: {max_value}")
    print(f"Выбранные предметы (индексы): {selected_items}")

    return max_value, selected_items


def visualize_lcs_dp():
    """Визуализирует процесс заполнения таблицы ДП для LCS."""
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ТАБЛИЦЫ ДП ДЛЯ LCS")
    print("=" * 70)

    str1 = "ABCD"
    str2 = "ACBD"

    print(f"\nВходные данные:")
    print(f"Строка 1: {str1}")
    print(f"Строка 2: {str2}")

    m, n = len(str1), len(str2)

    # Создаем и заполняем таблицу
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    print(f"\nНачальное состояние таблицы:")
    print_dp_table(dp)

    # Поэтапное заполнение
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    print(f"\nФинальное состояние таблицы:")
    print_dp_table(dp)

    # Восстановление LCS
    lcs_length, lcs = longest_common_subsequence(str1, str2)

    print(f"\nРезультат:")
    print(f"Длина LCS: {lcs_length}")
    print(f"LCS: {lcs}")

    return lcs_length, lcs


# ===== Экспериментальное исследование масштабируемости =====


def measure_knapsack_scalability():
    """Измеряет масштабируемость алгоритмов для задачи о рюкзаке."""
    print("\n" + "=" * 70)
    print("ИССЛЕДОВАНИЕ МАСШТАБИРУЕМОСТИ АЛГОРИТМОВ ДЛЯ ЗАДАЧИ О РЮКЗАКЕ")
    print("=" * 70)

    # Размеры задач для тестирования
    problem_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    times_standard = []
    times_optimized = []
    times_with_items = []

    print("\nЗамеры времени выполнения:")
    print("-" * 70)
    print("n  | Стандартный ДП | Оптимизированный | С восстановлением")
    print("-" * 70)

    random.seed(42)  # Для воспроизводимости

    for n in problem_sizes:
        # Генерация данных
        weights, values, capacity = generate_knapsack_data(n)

        # Замер времени для стандартного ДП
        standard_time = (
            timeit.timeit(lambda: knapsack_01(
                weights,
                values,
                capacity)[0],
                number=10)
            / 10
        )

        # Замер времени для оптимизированного ДП
        optimized_time = (
            timeit.timeit(
                lambda: knapsack_01_optimized(
                    weights,
                    values,
                    capacity
                ), number=10
            )
            / 10
        )

        # Замер времени для ДП с восстановлением
        items_time = (
            timeit.timeit(
                lambda: knapsack_01_with_items(
                    weights,
                    values,
                    capacity
                )[0], number=10
            )
            / 10
        )

        times_standard.append(standard_time)
        times_optimized.append(optimized_time)
        times_with_items.append(items_time)

        print(
            f"{n:2d} | {standard_time:12.6f} | {optimized_time:14.6f} | {
                items_time:15.6f}"
        )

    # Построение графиков
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(problem_sizes, times_standard, "bo-", label="Стандартный ДП",
             linewidth=2)
    plt.plot(
        problem_sizes, times_optimized, "ro-", label="Оптимизированный ДП",
        linewidth=2
    )
    plt.plot(
        problem_sizes,
        times_with_items,
        "go-",
        label="ДП с восстановлением",
        linewidth=2,
    )
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Масштабируемость алгоритмов для задачи о рюкзаке")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(problem_sizes, times_standard, "bo-", linewidth=2)
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Стандартный ДП (O(n*W))")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(problem_sizes, times_optimized, "ro-", linewidth=2)
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Оптимизированный ДП (O(n*W))")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(problem_sizes, times_with_items, "go-", linewidth=2)
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("ДП с восстановлением (O(n*W))")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("knapsack_scalability.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Анализ роста времени
    print("\n" + "-" * 70)
    print("АНАЛИЗ РОСТА ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
    print("-" * 70)

    print("\nОтношение времени при увеличении n в 2 раза:")
    print("n  -> 2n  | Стандартный | Оптимизированный | С восстановлением")
    print("-" * 70)

    for i in range(len(problem_sizes) // 2):
        n1 = problem_sizes[i]
        n2 = problem_sizes[i * 2]

        if n2 <= problem_sizes[-1]:
            idx1 = i
            idx2 = i * 2

            ratio_std = (
                times_standard[idx2] / times_standard[idx1]
                if times_standard[idx1] > 0
                else 0
            )
            ratio_opt = (
                times_optimized[idx2] / times_optimized[idx1]
                if times_optimized[idx1] > 0
                else 0
            )
            ratio_items = (
                times_with_items[idx2] / times_with_items[idx1]
                if times_with_items[idx1] > 0
                else 0
            )

            print(
                f"{n1:2d} -> {n2:3d} | {ratio_std:11.2f} | {
                    ratio_opt:15.2f} | {ratio_items:16.2f}"
            )


def measure_lcs_scalability():
    """Измеряет масштабируемость алгоритмов для LCS."""
    print("\n" + "=" * 70)
    print("ИССЛЕДОВАНИЕ МАСШТАБИРУЕМОСТИ АЛГОРИТМОВ ДЛЯ LCS")
    print("=" * 70)

    # Размеры строк для тестирования
    string_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times_lcs = []

    print("\nЗамеры времени выполнения для LCS:")
    print("-" * 50)
    print("Длина строк | Время выполнения")
    print("-" * 50)

    random.seed(42)

    for n in string_lengths:
        str1, str2 = generate_string_pair(n)

        # Замер времени для LCS
        lcs_time = (
            timeit.timeit(lambda: longest_common_subsequence(
                str1,
                str2)[0],
                number=10
            )
            / 10
        )

        times_lcs.append(lcs_time)

        print(f"{n:10d} | {lcs_time:15.6f}")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(string_lengths, times_lcs, "bo-", linewidth=2)
    plt.xlabel("Длина строк (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Масштабируемость алгоритма LCS (O(n²))")
    plt.grid(True, alpha=0.3)

    # Добавление теоретической кривой O(n²)
    theoretical = [0.000001 * n * n for n in string_lengths]
    plt.plot(
        string_lengths,
        theoretical,
        "r--",
        label="O(n²) теоретическая",
        linewidth=2
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("lcs_scalability.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Анализ квадратичного роста
    print("\n" + "-" * 50)
    print("АНАЛИЗ КВАДРАТИЧНОГО РОСТА:")
    print("-" * 50)

    print("\nПроверка квадратичного роста (время / n²):")
    for n, time_val in zip(string_lengths, times_lcs):
        if n > 0:
            ratio = time_val / (n * n)
            print(f"n={n:3d}: время/n² = {ratio:.10f}")


def measure_levenshtein_scalability():
    """Измеряет масштабируемость алгоритмов для расстояния Левенштейна."""
    print("\n" + "=" * 70)
    print(
        "ИССЛЕДОВАНИЕ МАСШТАБИРУЕМОСТИ АЛГОРИТМОВ ДЛЯ РАССТОЯНИЯ ЛЕВЕНШТЕЙНА"
        )
    print("=" * 70)

    # Размеры строк для тестирования
    string_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times_standard = []
    times_optimized = []

    print("\nЗамеры времени выполнения:")
    print("-" * 70)
    print("Длина строк | Стандартный | Оптимизированный")
    print("-" * 70)

    random.seed(42)

    for n in string_lengths:
        str1, str2 = generate_string_pair(n)

        # Замер времени для стандартного алгоритма
        standard_time = (
            timeit.timeit(lambda: levenshtein_distance(
                str1,
                str2
            ), number=10) / 10
        )

        # Замер времени для оптимизированного алгоритма
        optimized_time = (
            timeit.timeit(lambda: levenshtein_distance_optimized(
                str1,
                str2
            ), number=10)
            / 10
        )

        times_standard.append(standard_time)
        times_optimized.append(optimized_time)

        print(f"{n:10d} | {standard_time:11.6f} | {optimized_time:14.6f}")

    # Построение графиков
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(string_lengths, times_standard, "bo-", label="Стандартный",
             linewidth=2)
    plt.plot(
        string_lengths, times_optimized, "ro-", label="Оптимизированный",
        linewidth=2
    )
    plt.xlabel("Длина строк (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Сравнение алгоритмов Левенштейна")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Отношение времени выполнения
    ratios = []
    for std, opt in zip(times_standard, times_optimized):
        if opt > 0:
            ratios.append(std / opt)
        else:
            ratios.append(0)

    plt.plot(string_lengths, ratios, "go-", linewidth=2)
    plt.xlabel("Длина строк (n)")
    plt.ylabel("Отношение (стандартный/оптимизированный)")
    plt.title("Эффективность оптимизации")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("levenshtein_scalability.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "-" * 70)
    print("ВЫВОДЫ ПО МАСШТАБИРУЕМОСТИ:")
    print("-" * 70)
    print(
        """
1. Все алгоритмы ДП имеют квадратичную или псевдополиномиальную сложность.
2. Оптимизированные версии алгоритмов работают быстрее за счет экономии памяти.
3. Время выполнения растет пропорционально квадрату размера входных данных.
4. Для больших задач необходимо использовать эвристики или
   приближенные алгоритмы.
5. Оптимизация памяти часто приводит к ускорению выполнения за счет лучшей
   локализации данных.
    """
    )


# ====================== Сравнительный анализ алгоритмов ======================


def compare_dp_algorithms():
    """Сравнительный анализ различных алгоритмов ДП."""
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 70)

    # Тестовые данные
    print("\nТестовые данные для сравнения:")
    print("-" * 50)

    # 1. Рюкзак
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    print(f"\n1. Задача о рюкзаке:")
    print(f"   Веса: {weights}")
    print(f"   Стоимости: {values}")
    print(f"   Вместимость: {capacity}")

    result_std = knapsack_01(weights, values, capacity)
    result_opt = knapsack_01_optimized(weights, values, capacity)
    result_items = knapsack_01_with_items(weights, values, capacity)

    print(f"   Стандартный ДП: {result_std[0]} (предметы: {result_std[1]})")
    print(f"   Оптимизированный ДП: {result_opt}")
    print(f"   ДП с восстановлением: {result_items[0]} (предметы: {
        result_items[1]})")

    # 2. LCS
    str1 = "ABCDGH"
    str2 = "AEDFHR"

    print(f"\n2. Наибольшая общая подпоследовательность:")
    print(f"   Строка 1: {str1}")
    print(f"   Строка 2: {str2}")

    lcs_length, lcs = longest_common_subsequence(str1, str2)
    print(f"   Длина LCS: {lcs_length}")
    print(f"   LCS: {lcs}")

    # 3. Левенштейн
    str3 = "kitten"
    str4 = "sitting"

    print(f"\n3. Расстояние Левенштейна:")
    print(f"   Строка 1: {str3}")
    print(f"   Строка 2: {str4}")

    lev_std = levenshtein_distance(str3, str4)
    lev_opt = levenshtein_distance_optimized(str3, str4)

    print(f"   Стандартный алгоритм: {lev_std}")
    print(f"   Оптимизированный алгоритм: {lev_opt}")

    # Сводная таблица производительности
    print("\n" + "-" * 70)
    print("СВОДНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 70)

    print(
        "\nАлгоритм             | Временная сложность | Пространственная сложность"
    )
    print("-" * 70)
    print("Рюкзак (стандартный) | O(n*W)             | O(n*W)")
    print("Рюкзак (оптимизир.)  | O(n*W)             | O(W)")
    print("LCS                  | O(m*n)             | O(m*n)")
    print("Левенштейн (станд.)  | O(m*n)             | O(m*n)")
    print("Левенштейн (оптим.)  | O(m*n)             | O(min(m,n))")


# ====================== Основная функция ======================


def main():
    """Основная функция для запуска анализа."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ АЛГОРИТМОВ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 70)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Визуализация таблиц ДП
    visualize_knapsack_dp()
    visualize_lcs_dp()

    # Исследование масштабируемости
    measure_knapsack_scalability()
    measure_lcs_scalability()
    measure_levenshtein_scalability()

    # Сравнительный анализ
    compare_dp_algorithms()

    print("\n" + "=" * 70)
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print("\nВсе графики сохранены в файлы:")
    print("  - knapsack_scalability.png")
    print("  - lcs_scalability.png")
    print("  - levenshtein_scalability.png")


if __name__ == "__main__":
    main()

