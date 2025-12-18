import timeit
import random
from dynamic_programming import (
    fibonacci_memoization,
    fibonacci_tabular,
    knapsack_01,
    knapsack_01_optimized,
)

# ====================== Вспомогательные функции ======================


def fractional_knapsack(capacity, items):
    """
    Жадный алгоритм для непрерывного рюкзака.
    items: список кортежей (вес, стоимость)
    """
    # Сортируем по удельной стоимости (стоимость/вес) по убыванию
    sorted_items = sorted(items, key=lambda x: x[1] / x[0], reverse=True)

    total_value = 0.0
    taken_items = []

    for weight, value in sorted_items:
        if capacity == 0:
            break

        # Берем столько, сколько можем
        take_weight = min(weight, capacity)
        fraction = take_weight / weight
        total_value += value * fraction

        taken_items.append((weight, value, fraction))
        capacity -= take_weight

    return total_value, taken_items


# ================== Сравнение подходов для чисел Фибоначчи ==================


def compare_fibonacci():
    """
        Сравнение времени работы и потребления памяти для разных подходов
        к вычислению чисел Фибоначчи.
    """
    print("=" * 80)
    print("СРАВНЕНИЕ ПОДХОДОВ ДЛЯ ЧИСЕЛ ФИБОНАЧЧИ")
    print("=" * 80)

    # Тестовые значения
    test_values = [10, 20, 30, 40, 50, 100, 200, 300]

    print("\n" + "-" * 80)
    print("РЕЗУЛЬТАТЫ ИЗМЕРЕНИЯ ВРЕМЕНИ:")
    print("-" * 80)

    for n in test_values:
        print(f"\nn = {n}")

        # Мемоизация (нисходящий подход)
        try:
            memo_time = (
                timeit.timeit(
                    lambda: fibonacci_memoization(n), number=100
                ) / 100
            )
            memo_success = True
        except RecursionError:
            memo_time = float("inf")
            memo_success = False

        # Табличный подход (восходящий)
        tabular_time = timeit.timeit(
            lambda: fibonacci_tabular(n), number=1000
        ) / 1000

        print(f"  Мемоизация (нисходящий):")
        if memo_success:
            print(f"    Время: {memo_time:.8f} секунд")
            print(f"    Результат: {fibonacci_memoization(n)}")
        else:
            print(f"    ОШИБКА: превышена глубина рекурсии")

        print(f"  Табличный (восходящий):")
        print(f"    Время: {tabular_time:.8f} секунд")
        print(f"    Результат: {fibonacci_tabular(n)}")

        if memo_success:
            # Проверка корректности
            memo_result = fibonacci_memoization(n)
            tabular_result = fibonacci_tabular(n)
            print(f"  Результаты совпадают: {memo_result == tabular_result}")

    # Тестирование производительности
    print("\n" + "-" * 80)
    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 80)

    performance_values = [100, 500, 1000, 2000, 5000]

    print("\nТабличный подход (восходящий) для очень больших n:")
    for n in performance_values:
        tabular_time = timeit.timeit(
            lambda: fibonacci_tabular(n), number=100
        ) / 100

        result = fibonacci_tabular(n)
        result_str = str(result)
        if len(result_str) > 10:
            result_display = f"...{result_str[-10:]} (длина: {
                len(result_str)} цифр)"
        else:
            result_display = f"{result}"

        print(f"  n={n:5d}: время={tabular_time:.6f} сек, результат={
            result_display}")

    print("\n" + "-" * 80)
    print("ВЫВОДЫ ПО ЧИСЛАМ ФИБОНАЧЧИ:")
    print("-" * 80)
    print(
        """
1. Нисходящий подход (мемоизация):
   - Плюсы: простая реализация, близка к математическому определению
   - Минусы: риск переполнения стека для больших n, накладные
     расходы на рекурсию
   - Время: O(n), Память: O(n) для мемоизации + O(n) для стека вызовов

2. Восходящий подход (табличный):
   - Плюсы: нет риска переполнения стека, более эффективное
     использование памяти
   - Минусы: менее интуитивная реализация
   - Время: O(n), Память: O(1)

3. Для n > 1000 табличный подход - единственный рабочий вариант.
4. Табличный подход в 2-3 раза быстрее для средних значений n.
    """
    )


# ===== Сравнение жадного алгоритма и ДП для задачи о рюкзаке =====


def compare_knapsack_algorithms():
    """
        Сравнение жадного алгоритма для непрерывного рюкзака
        и ДП для 0-1 рюкзака.
    """
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ АЛГОРИТМОВ ДЛЯ ЗАДАЧИ О РЮКЗАКЕ")
    print("=" * 80)

    # Тестовые примеры
    test_cases = [
        {
            "name": "Пример 1: Жадный алгоритм оптимален",
            "capacity": 10,
            "items": [(5, 50), (4, 40), (3, 30)],
            "description": "Все предметы имеют одинаковую удельную стоимость",
        },
        {
            "name": "Пример 2: Жадный алгоритм не оптимален",
            "capacity": 10,
            "items": [(6, 60), (5, 49), (5, 49)],
            "description": "Жадный берет предмет 1, но лучше взять предметы 2 и 3",
        },
        {
            "name": "Пример 3: Классический пример",
            "capacity": 50,
            "items": [(20, 60), (30, 90), (10, 30)],
            "description": "Жадный дает 120, оптимально 90",
        },
        {
            "name": "Пример 4: Большой разрыв",
            "capacity": 100,
            "items": [(60, 120), (50, 100), (50, 100)],
            "description": "Жадный: 120, Оптимально: 200",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 60)
        print(f"Описание: {test_case['description']}")

        capacity = test_case["capacity"]
        items = test_case["items"]

        print(f"\nВместимость рюкзака: {capacity}")
        print("Предметы (вес, стоимость, удельная стоимость):")
        for j, (w, v) in enumerate(items, 1):
            ratio = v / w
            print(
                f"  Предмет {j}: вес={w}, стоимость={v}, удельная стоимость={
                    ratio:.2f}"
            )

        # Жадный алгоритм (непрерывный)
        print("\n--- Жадный алгоритм (непрерывный рюкзак) ---")
        greedy_start = timeit.default_timer()
        greedy_value, greedy_items = fractional_knapsack(capacity, items)
        greedy_time = timeit.default_timer() - greedy_start

        print(f"Максимальная стоимость: {greedy_value:.2f}")
        print(f"Время выполнения: {greedy_time:.8f} секунд")
        print("Взятые предметы (вес, стоимость, доля):")
        for w, v, f in greedy_items:
            print(f"  ({w}, {v}, {f:.2f})")

        # Динамическое программирование (0-1 рюкзак)
        print("\n--- Динамическое программирование (0-1 рюкзак) ---")

        weights = [w for w, v in items]
        values = [v for w, v in items]

        dp_start = timeit.default_timer()
        dp_value, dp_items = knapsack_01(weights, values, capacity)
        dp_time = timeit.default_timer() - dp_start

        print(f"Максимальная стоимость: {dp_value:.2f}")
        print(f"Время выполнения: {dp_time:.8f} секунд")
        print(f"Взятые предметы (индексы): {dp_items}")

        # Оптимизированное ДП
        print("\n--- Оптимизированное ДП (0-1 рюкзак) ---")
        opt_start = timeit.default_timer()
        opt_value = knapsack_01_optimized(weights, values, capacity)
        opt_time = timeit.default_timer() - opt_start

        print(f"Максимальная стоимость: {opt_value:.2f}")
        print(f"Время выполнения: {opt_time:.8f} секунд")

        # Сравнение
        print("\n--- Сравнение результатов ---")
        print(f"Жадный алгоритм (непрерывный): {greedy_value:.2f}")
        print(f"ДП (0-1 рюкзак): {dp_value:.2f}")
        print(f"Оптимизированное ДП: {opt_value:.2f}")

        if abs(dp_value - opt_value) < 0.01:
            print("ДП и оптимизированное ДП дали одинаковый результат")
        else:
            print("ОШИБКА: ДП и оптимизированное ДП дали разные результаты!")

        if abs(greedy_value - dp_value) < 0.01:
            print("Жадный алгоритм дал оптимальное решение")
        else:
            print(
                f"Жадный алгоритм не оптимален, разница: {
                    dp_value - greedy_value:.2f}"
            )

        print(f"\nОтношение времени выполнения:")
        print(f"  ДП/жадный: {dp_time/greedy_time:.2f}")
        print(f"  Оптимизированное ДП/жадный: {opt_time/greedy_time:.2f}")
        print(f"  ДП/оптимизированное ДП: {dp_time/opt_time:.2f}")

    # Тестирование производительности на случайных данных
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ НА СЛУЧАЙНЫХ ДАННЫХ")
    print("=" * 80)

    random.seed(42)  # Для воспроизводимости

    test_sizes = [5, 10, 15, 20, 25, 30]

    print("\nСравнение времени выполнения (среднее за 10 запусков):")
    print("-" * 80)
    print(
        "n  | Вместимость | Жадный (сек) | ДП (сек) | Отношение | Опт.ДП (сек)"
    )
    print("-" * 80)

    greedy_times = []
    dp_times = []
    opt_times = []

    for n in test_sizes:
        capacity = n * 10
        items = []
        for _ in range(n):
            weight = random.randint(1, 30)
            value = random.randint(1, 100)
            items.append((weight, value))

        weights = [w for w, v in items]
        values = [v for w, v in items]

        # Замер времени (усреднение)
        greedy_time = (
            sum(
                timeit.timeit(
                    lambda: fractional_knapsack(capacity, items)[0], number=10
                )
                for _ in range(10)
            )
            / 100
        )

        dp_time = (
            sum(
                timeit.timeit(
                    lambda: knapsack_01(
                        weights, values, capacity
                    )[0], number=10
                )
                for _ in range(10)
            )
            / 100
        )

        opt_time = (
            sum(
                timeit.timeit(
                    lambda: knapsack_01_optimized(
                        weights, values, capacity
                    ), number=10
                )
                for _ in range(10)
            )
            / 100
        )

        greedy_times.append(greedy_time)
        dp_times.append(dp_time)
        opt_times.append(opt_time)

        print(
            f"{n:2d} | {capacity:11d} | {greedy_time:11.6f} | {
                dp_time:8.6f} | {dp_time/greedy_time:8.2f} | {opt_time:10.6f}"
        )

    # Анализ качества решений на случайных данных
    print("\n" + "-" * 80)
    print("АНАЛИЗ КАЧЕСТВА РЕШЕНИЙ НА СЛУЧАЙНЫХ ДАННЫХ:")
    print("-" * 80)

    num_tests = 100
    greedy_optimal_count = 0
    greedy_avg_ratio = 0

    print(f"\nТестируем {num_tests} случайных задач размером n=10:")

    for test_num in range(num_tests):
        capacity = 50
        items = []
        for _ in range(10):
            weight = random.randint(1, 30)
            value = random.randint(1, 100)
            items.append((weight, value))

        weights = [w for w, v in items]
        values = [v for w, v in items]

        greedy_value = fractional_knapsack(capacity, items)[0]
        dp_value = knapsack_01(weights, values, capacity)[0]

        if abs(greedy_value - dp_value) < 0.01:
            greedy_optimal_count += 1

        if dp_value > 0:
            greedy_avg_ratio += greedy_value / dp_value

    greedy_optimal_percent = (greedy_optimal_count / num_tests) * 100
    greedy_avg_ratio = (greedy_avg_ratio / num_tests) * 100

    print(
        f"Жадный алгоритм дал оптимальное решение в {greedy_optimal_percent:.1f}% случаев"
    )
    print(
        f"Среднее качество жадного алгоритма: {greedy_avg_ratio:.1f}% от оптимального"
    )

    print("\n" + "-" * 80)
    print("ВЫВОДЫ ПО ЗАДАЧЕ О РЮКЗАКЕ:")
    print("-" * 80)
    print(
        """
1. Жадный алгоритм (непрерывный рюкзак):
   - Время: O(n log n) для сортировки
   - Всегда дает оптимальное решение для непрерывной задачи
   - Может давать неоптимальное решение для 0-1 рюкзака
   - Очень быстрый, хорошо масштабируется

2. Динамическое программирование (0-1 рюкзак):
   - Время: O(n * W), где W - вместимость рюкзака
   - Память: O(n * W) для полной таблицы, O(W) для оптимизированной
   - Всегда дает точное оптимальное решение
   - Медленнее жадного, особенно при больших W

3. Оптимизированное ДП:
   - Та же временная сложность O(n * W)
   - Меньше использует памяти: O(W) вместо O(n * W)
   - На практике быстрее полной таблицы ДП
    """
    )


# ====================== Основная функция ======================


def main():
    """Основная функция для запуска сравнений."""
    print("=" * 80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 80)

    # Сравнение подходов для чисел Фибоначчи
    compare_fibonacci()

    # Сравнение алгоритмов для задачи о рюкзаке
    compare_knapsack_algorithms()

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    main()
