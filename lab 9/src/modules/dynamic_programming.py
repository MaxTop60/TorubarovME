# ====================== Числа Фибоначчи ======================


def fibonacci_naive(n):
    """Наивная рекурсивная реализация чисел Фибоначчи."""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


# Временная сложность: O(2^n) - экспоненциальная
# Пространственная сложность: O(n) - глубина стека вызовов


def fibonacci_memoization(n, memo=None):
    """Рекурсивная реализация чисел Фибоначчи с мемоизацией."""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoization(
        n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]


# Временная сложность: O(n) - линейная
# Пространственная сложность: O(n) - для хранения мемоизированных значений


def fibonacci_tabular(n):
    """Итеративная табличная реализация чисел Фибоначчи."""
    if n <= 1:
        return n

    prev2, prev1 = 0, 1  # F(0), F(1)

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1


# Временная сложность: O(n) - линейная
# Пространственная сложность: O(1) - константная


# ====================== Задача о рюкзаке (0-1 Knapsack) ======================


def knapsack_01(weights, values, capacity):
    """
        Решение задачи о рюкзаке (0-1) с использованием
        динамического программирования.
    """
    n = len(weights)

    # Создаем таблицу dp размером (n+1) x (capacity+1)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Заполняем таблицу
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    # Восстанавливаем выбранные предметы
    selected_items = []
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()

    return dp[n][capacity], selected_items


# Временная сложность: O(n * W), где n - количество предметов, W - вместимость
# Пространственная сложность: O(n * W) - для таблицы dp


def knapsack_01_optimized(weights, values, capacity):
    """
        Оптимизированное решение задачи о рюкзаке (0-1)
        с использованием 1D массива.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# Временная сложность: O(n * W)
# Пространственная сложность: O(W) - только один массив


def knapsack_01_with_items(weights, values, capacity):
    """
    Решение задачи о рюкзаке (0-1) с восстановлением набора предметов
    и использованием двух массивов для оптимизации памяти.
    """
    n = len(weights)

    # Используем два массива: для предыдущей и текущей строки
    prev = [0] * (capacity + 1)
    curr = [0] * (capacity + 1)

    # Матрица для хранения выбранных предметов
    # selected[i][w] = True, если предмет i включен в решение для веса w
    selected = [[False] * (capacity + 1) for _ in range(n)]

    # Заполняем таблицу
    for i in range(n):
        for w in range(capacity + 1):
            if weights[i] <= w:
                # Вариант 1: не берем текущий предмет
                not_take = prev[w]
                # Вариант 2: берем текущий предмет
                take = prev[w - weights[i]] + values[i]

                if take > not_take:
                    curr[w] = take
                    selected[i][w] = True
                else:
                    curr[w] = not_take
                    selected[i][w] = False
            else:
                curr[w] = prev[w]
                selected[i][w] = False

        # Обновляем массивы для следующей итерации
        prev, curr = curr, [0] * (capacity + 1)

    # Восстанавливаем выбранные предметы
    selected_items = []
    w = capacity

    for i in range(n - 1, -1, -1):
        if selected[i][w]:
            selected_items.append(i)
            w -= weights[i]

    selected_items.reverse()

    return prev[capacity], selected_items


# Временная сложность: O(n * W)
# Пространственная сложность: O(n * W) - для матрицы selected, O(W) для dp


# ====== Наибольшая общая подпоследовательность (LCS) ======


def longest_common_subsequence(str1, str2):
    """Нахождение длины наибольшей общей подпоследовательности (LCS)."""
    m, n = len(str1), len(str2)

    # Создаем таблицу dp размером (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Заполняем таблицу
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Восстанавливаем LCS
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()

    return dp[m][n], "".join(lcs)


# Временная сложность: O(m * n), где m и n - длины строк
# Пространственная сложность: O(m * n) - для таблицы dp


def longest_common_subsequence_optimized(str1, str2):
    """Оптимизированная версия LCS с использованием двух строк."""
    m, n = len(str1), len(str2)

    if m < n:
        str1, str2 = str2, str1
        m, n = n, m

    # Используем только два ряда вместо всей таблицы
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# Временная сложность: O(m * n)
# Пространственная сложность: O(min(m, n))


def longest_common_subsequence_with_path(str1, str2):
    """
    Нахождение LCS с восстановлением подпоследовательности
    и использованием дополнительной структуры для хранения пути.
    """
    m, n = len(str1), len(str2)

    # Таблицы для длин и направлений
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    direction = [[0] * (n + 1) for _ in range(m + 1)]

    # Заполняем таблицы
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                direction[i][j] = 1  # Диагональ - символы совпали
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                    direction[i][j] = 2  # Вверх
                else:
                    dp[i][j] = dp[i][j - 1]
                    direction[i][j] = 3  # Влево

    # Восстанавливаем LCS по направлениям
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if direction[i][j] == 1:  # Диагональ
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif direction[i][j] == 2:  # Вверх
            i -= 1
        else:  # Влево
            j -= 1

    lcs.reverse()

    return dp[m][n], "".join(lcs)


# Временная сложность: O(m * n)
# Пространственная сложность: O(m * n) - для таблиц dp и direction


# ====================== Расстояние Левенштейна ======================


def levenshtein_distance(str1, str2):
    """Вычисление расстояния Левенштейна (редакционного расстояния)."""
    m, n = len(str1), len(str2)

    # Создаем таблицу dp размером (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Инициализация: пустая строка
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Заполняем таблицу
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # удаление
                    dp[i][j - 1] + 1,  # вставка
                    dp[i - 1][j - 1] + 1,  # замена
                )

    return dp[m][n]


# Временная сложность: O(m * n)
# Пространственная сложность: O(m * n)


def levenshtein_distance_optimized(str1, str2):
    """Оптимизированная версия расстояния Левенштейна."""
    m, n = len(str1), len(str2)

    if m < n:
        str1, str2 = str2, str1
        m, n = n, m

    # Используем только два ряда
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = min(
                    prev[j] + 1,  # удаление
                    curr[j - 1] + 1,  # вставка
                    prev[j - 1] + 1,  # замена
                )
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# Временная сложность: O(m * n)
# Пространственная сложность: O(min(m, n))


def levenshtein_distance_with_operations(str1, str2):
    """
    Вычисление расстояния Левенштейна с восстановлением операций.
    Возвращает расстояние и последовательность операций.
    """
    m, n = len(str1), len(str2)

    # Таблицы для расстояний и операций
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[0] * (n + 1) for _ in range(m + 1)]

    # Инициализация
    for i in range(m + 1):
        dp[i][0] = i
        operations[i][0] = 1  # Удаление
    for j in range(n + 1):
        dp[0][j] = j
        operations[0][j] = 2  # Вставка

    # Заполняем таблицы
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = 0  # Совпадение
            else:
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1
                replace_cost = dp[i - 1][j - 1] + 1

                min_cost = min(delete_cost, insert_cost, replace_cost)
                dp[i][j] = min_cost

                if min_cost == delete_cost:
                    operations[i][j] = 1  # Удаление
                elif min_cost == insert_cost:
                    operations[i][j] = 2  # Вставка
                else:
                    operations[i][j] = 3  # Замена

    # Восстанавливаем последовательность операций
    ops_sequence = []
    i, j = m, n

    while i > 0 or j > 0:
        op = operations[i][j]

        if op == 0:  # Совпадение
            ops_sequence.append(f"сохранить '{str1[i-1]}'")
            i -= 1
            j -= 1
        elif op == 1:  # Удаление
            ops_sequence.append(f"удалить '{str1[i-1]}'")
            i -= 1
        elif op == 2:  # Вставка
            ops_sequence.append(f"вставить '{str2[j-1]}'")
            j -= 1
        elif op == 3:  # Замена
            ops_sequence.append(f"заменить '{str1[i-1]}' на '{str2[j-1]}'")
            i -= 1
            j -= 1
        else:  # Для граничных случаев
            if i > 0:
                ops_sequence.append(f"удалить '{str1[i-1]}'")
                i -= 1
            elif j > 0:
                ops_sequence.append(f"вставить '{str2[j-1]}'")
                j -= 1

    ops_sequence.reverse()

    return dp[m][n], ops_sequence


# Временная сложность: O(m * n)
# Пространственная сложность: O(m * n) - для таблиц dp и operations
