# =============== Размен монет (минимальное количество монет) ================

def coin_change_min_coins(coins, amount):
    """
    Находит минимальное количество монет для размена суммы.

    Args:
        coins: список доступных номиналов монет
        amount: сумма для размена

    Returns:
        кортеж (минимальное количество монет, список монет для размена)

    Сложность:
        Время: O(n * amount), где n - количество номиналов
        Память: O(amount) для DP массива
    """
    if amount == 0:
        return 0, []

    # Сортируем монеты по возрастанию
    coins_sorted = sorted(coins)

    # dp[i] - минимальное количество монет для суммы i
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    # Для восстановления ответа: prev_coin[i] - последняя монета для суммы i
    prev_coin = [-1] * (amount + 1)

    # Заполняем DP таблицу
    for coin in coins_sorted:
        for i in range(coin, amount + 1):
            if dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                prev_coin[i] = coin

    # Если невозможно разменять сумму
    if dp[amount] == float("inf"):
        return -1, []

    # Восстанавливаем набор монет
    coins_used = []
    remaining = amount

    while remaining > 0:
        coin = prev_coin[remaining]
        coins_used.append(coin)
        remaining -= coin

    return dp[amount], coins_used


def coin_change_num_ways(coins, amount):
    """
    Находит количество способов разменять сумму
    (каждая монета может использоваться много раз).

    Args:
        coins: список доступных номиналов монет
        amount: сумма для размена

    Returns:
        количество способов размена

    Сложность:
        Время: O(n * amount)
        Память: O(amount)
    """
    # dp[i] - количество способов разменять сумму i
    dp = [0] * (amount + 1)
    dp[0] = 1  # 1 способ разменять сумму 0 - не брать ни одной монеты

    # Важно: сначала итерируем по монетам, потом по суммам
    # Это гарантирует, что каждая монета может использоваться много раз
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


# ========= Наибольшая возрастающая подпоследовательность (LIS) ==============


def longest_increasing_subsequence(nums):
    """
    Находит длину наибольшей возрастающей подпоследовательности.

    Args:
        nums: список чисел

    Returns:
        кортеж (длина LIS, сама подпоследовательность)

    Сложность:
        Время: O(n²) - для наивного подхода
        Память: O(n) для DP массива
    """
    if not nums:
        return 0, []

    n = len(nums)

    # dp[i] - длина LIS, заканчивающейся на элементе i
    dp = [1] * n

    # prev[i] - индекс предыдущего элемента в LIS
    prev = [-1] * n

    # Заполняем DP таблицу
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    # Находим максимальную длину и её позицию
    max_length = max(dp)
    max_index = dp.index(max_length)

    # Восстанавливаем подпоследовательность
    lis = []
    idx = max_index

    while idx != -1:
        lis.append(nums[idx])
        idx = prev[idx]

    lis.reverse()

    return max_length, lis


def longest_increasing_subsequence_optimized(nums):
    """
    Оптимизированный алгоритм для LIS с
    использованием бинарного поиска (O(n log n)).

    Args:
        nums: список чисел

    Returns:
        кортеж (длина LIS, сама подпоследовательность)

    Сложность:
        Время: O(n log n) - с бинарным поиском
        Память: O(n) для хранения активных списков
    """
    if not nums:
        return 0, []

    n = len(nums)

    # tails[i] - наименьший возможный последний элемент LIS длины i+1
    tails = []

    # Для восстановления подпоследовательности
    # parent[i] - индекс предыдущего элемента в LIS для nums[i]
    parent = [-1] * n

    # Для каждого элемента в tails храним индекс в исходном массиве
    tails_indices = []

    for i in range(n):
        # Бинарный поиск позиции для вставки nums[i]
        left, right = 0, len(tails)

        while left < right:
            mid = (left + right) // 2
            if tails[mid] < nums[i]:
                left = mid + 1
            else:
                right = mid

        if left == len(tails):
            tails.append(nums[i])
            tails_indices.append(i)
        else:
            tails[left] = nums[i]
            tails_indices[left] = i

        # Запоминаем родителя для восстановления
        if left > 0:
            parent[i] = tails_indices[left - 1]

    # Восстанавливаем LIS
    lis_length = len(tails)
    lis = []

    # Начинаем с последнего элемента в tails
    idx = tails_indices[-1]

    for _ in range(lis_length):
        lis.append(nums[idx])
        idx = parent[idx]

    lis.reverse()

    return lis_length, lis


def longest_non_decreasing_subsequence(nums):
    """
    Находит длину наибольшей неубывающей подпоследовательности.
    Отличие от LIS: допускаются равные элементы.

    Args:
        nums: список чисел

    Returns:
        кортеж (длина LNIS, сама подпоследовательность)
    """
    if not nums:
        return 0, []

    n = len(nums)
    dp = [1] * n
    prev = [-1] * n

    for i in range(n):
        for j in range(i):
            if nums[j] <= nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    max_length = max(dp)
    max_index = dp.index(max_length)

    lnis = []
    idx = max_index

    while idx != -1:
        lnis.append(nums[idx])
        idx = prev[idx]

    lnis.reverse()

    return max_length, lnis


# ====================== Демонстрационные функции ======================


def demo_coin_change():
    """Демонстрация задачи размена монет."""
    print("=" * 70)
    print("ЗАДАЧА РАЗМЕНА МОНЕТ")
    print("=" * 70)

    test_cases = [
        {
            "coins": [1, 2, 5],
            "amount": 11,
            "description": "Стандартная система монет (1, 2, 5)",
        },
        {
            "coins": [2, 3, 5],
            "amount": 11,
            "description": "Система без монеты 1"
        },
        {"coins": [2], "amount": 3, "description": "Невозможно разменять"},
        {
            "coins": [1, 3, 4],
            "amount": 6,
            "description": "Система, где жадный алгоритм не оптимален",
        },
        {
            "coins": [1, 5, 10, 25],
            "amount": 41,
            "description": "Американская система центов",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Монеты: {test['coins']}")
        print(f"   Сумма: {test['amount']}")

        # Минимальное количество монет
        min_coins, coins_used = coin_change_min_coins(
            test["coins"], test["amount"])

        if min_coins == -1:
            print(f"   Невозможно разменять сумму")
        else:
            print(f"   Минимальное количество монет: {min_coins}")
            print(f"   Использованные монеты: {coins_used}")
            print(f"   Проверка: {sum(coins_used)} = {test['amount']}")

        # Количество способов размена
        num_ways = coin_change_num_ways(test["coins"], test["amount"])
        print(f"   Количество способов размена: {num_ways}")

    # Сравнение с жадным алгоритмом для системы [1, 3, 4]
    print("\n" + "-" * 70)
    print("СРАВНЕНИЕ С ЖАДНЫМ АЛГОРИТМОМ (монеты: [1, 3, 4]):")
    print("-" * 70)

    coins = [1, 3, 4]

    for amount in [6, 10, 13]:
        # Жадный алгоритм
        greedy_coins = []
        remaining = amount
        for coin in sorted(coins, reverse=True):
            while remaining >= coin:
                greedy_coins.append(coin)
                remaining -= coin

        # ДП алгоритм
        dp_min_coins, dp_coins_used = coin_change_min_coins(coins, amount)

        print(f"\nСумма: {amount}")
        print(f"  Жадный алгоритм: {len(greedy_coins)} монет - {greedy_coins}")
        print(f"  ДП алгоритм: {dp_min_coins} монет - {dp_coins_used}")

        if len(greedy_coins) > dp_min_coins:
            print(f"  ВНИМАНИЕ: Жадный алгоритм не оптимален!")


def demo_lis():
    """Демонстрация задачи LIS."""
    print("\n" + "=" * 70)
    print("НАИБОЛЬШАЯ ВОЗРАСТАЮЩАЯ ПОДПОСЛЕДОВАТЕЛЬНОСТЬ (LIS)")
    print("=" * 70)

    test_cases = [
        {"nums": [10, 9, 2, 5, 3, 7, 101, 18],
         "description": "Классический пример"},
        {"nums": [0, 1, 0, 3, 2, 3], "description": "Пример с повторениями"},
        {"nums": [7, 7, 7, 7, 7, 7], "description": "Все элементы одинаковые"},
        {"nums": [1, 3, 6, 7, 9, 4, 10, 5, 6], "description": "Сложный пример"},
        {"nums": [3, 10, 2, 1, 20],
         "description": "Простой возрастающий случай"},
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Последовательность: {test['nums']}")

        # Наивный подход O(n²)
        lis_length_naive, lis_naive = longest_increasing_subsequence(
            test["nums"])
        print(f"   Наивный ДП (O(n²)):")
        print(f"     Длина LIS: {lis_length_naive}")
        print(f"     LIS: {lis_naive}")

        # Оптимизированный подход O(n log n)
        lis_length_opt, lis_opt = longest_increasing_subsequence_optimized(
            test["nums"])
        print(f"   Оптимизированный (O(n log n)):")
        print(f"     Длина LIS: {lis_length_opt}")
        print(f"     LIS: {lis_opt}")

        if lis_length_naive != lis_length_opt:
            print(f"   ОШИБКА: Результаты не совпадают!")
        else:
            print(f"   Результаты совпадают")

        # Неубывающая подпоследовательность
        lnis_length, lnis = longest_non_decreasing_subsequence(test["nums"])
        print(f"   Наибольшая неубывающая подпоследовательность:")
        print(f"     Длина: {lnis_length}")
        print(f"     Подпоследовательность: {lnis}")


def demo_comparison():
    """Сравнение производительности алгоритмов."""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ АЛГОРИТМОВ LIS")
    print("=" * 70)

    import timeit
    import random

    # Генерация тестовых данных разного размера
    test_sizes = [50, 100, 200, 300, 400, 500]

    print("\nВремя выполнения (секунды):")
    print("n  | O(n²) алгоритм | O(n log n) алгоритм")
    print("-" * 50)

    random.seed(42)

    naive_times = []
    opt_times = []

    for n in test_sizes:
        # Генерируем случайную последовательность
        nums = [random.randint(1, 1000) for _ in range(n)]

        # Замер времени для наивного алгоритма
        naive_time = (
            timeit.timeit(lambda: longest_increasing_subsequence(nums)[0],
                          number=10)
            / 10
        )

        # Замер времени для оптимизированного алгоритма
        opt_time = (
            timeit.timeit(
                lambda: longest_increasing_subsequence_optimized(nums)[0],
                number=10
            )
            / 10
        )

        naive_times.append(naive_time)
        opt_times.append(opt_time)

        print(f"{n:3d} | {naive_time:12.6f} | {opt_time:15.6f}")

    # Анализ роста времени
    print("\n" + "-" * 50)
    print("АНАЛИЗ РОСТА ВРЕМЕНИ:")
    print("-" * 50)

    print("\nОтношение времени при увеличении n в 2 раза:")
    for i in range(len(test_sizes) // 2):
        n1 = test_sizes[i]
        n2 = test_sizes[i * 2]

        if n2 <= test_sizes[-1]:
            idx1 = i
            idx2 = i * 2

            ratio_naive = (
                naive_times[
                    idx2] / naive_times[idx1] if naive_times[idx1] > 0 else 0
            )
            ratio_opt = opt_times[
                idx2] / opt_times[idx1] if opt_times[idx1] > 0 else 0

            print(
                f"n={n1:3d} -> n={n2:3d}: O(n²) в {ratio_naive:.2f} раз, O(n log n) в {ratio_opt:.2f} раз"
            )


# ====================== Основная функция ======================


def main():
    """Основная функция для демонстрации задач."""
    print("=" * 70)
    print("РЕШЕНИЕ ЗАДАЧ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 70)

    # Демонстрация задачи размена монет
    demo_coin_change()

    # Демонстрация задачи LIS
    demo_lis()

    # Сравнение производительности
    demo_comparison()

    print("\n" + "=" * 70)
    print("ВСЕ ЗАДАЧИ РЕШЕНЫ")
    print("=" * 70)


if __name__ == "__main__":
    main()
