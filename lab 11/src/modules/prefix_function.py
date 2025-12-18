def prefix_function_naive(s):
    """
    Наивная реализация префикс-функции.

    Args:
        s: входная строка

    Returns:
        список π - префикс-функция для строки s

    Сложность:
        Время: O(n³) - крайне неэффективно
        Память: O(n) - для хранения результата
    """
    n = len(s)
    pi = [0] * n

    for i in range(1, n):
        for k in range(1, i + 1):
            if s[:k] == s[i - k + 1 : i + 1]:
                pi[i] = k

    return pi


def prefix_function(s):
    """
    Эффективное вычисление префикс-функции за O(n).

    Args:
        s: входная строка

    Returns:
        список π - префикс-функция для строки s

    Сложность:
        Время: O(n) - линейная сложность
        Память: O(n) - для хранения префикс-функции

    Алгоритм:
        Использует идею, что π[i] ≤ π[i-1] + 1
        и переиспользует ранее вычисленные значения
    """
    n = len(s)
    pi = [0] * n

    for i in range(1, n):
        j = pi[i - 1]

        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]

        if s[i] == s[j]:
            j += 1

        pi[i] = j

    return pi


def prefix_function_optimized(s):
    """
    Оптимизированная версия префикс-функции с явной обработкой граничных случаев.

    Args:
        s: входная строка

    Returns:
        список π - префикс-функция для строки s

    Сложность:
        Время: O(n)
        Память: O(n)
    """
    n = len(s)
    if n == 0:
        return []

    pi = [0] * n

    for i in range(1, n):
        j = pi[i - 1]

        # Пытаемся расширить предыдущий префикс
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]

        # Если символы совпадают, увеличиваем длину префикса
        if s[i] == s[j]:
            pi[i] = j + 1
        else:
            pi[i] = 0  # j уже равно 0

    return pi


def prefix_function_with_visualization(s):
    """
    Вычисление префикс-функции с выводом промежуточных шагов.

    Args:
        s: входная строка

    Returns:
        список π - префикс-функция для строки s

    Сложность:
        Время: O(n)
        Память: O(n)
    """
    n = len(s)
    pi = [0] * n

    print(f"Вычисление префикс-функции для строки: '{s}'")
    print(f"Индексы: {list(range(n))}")
    print(f"Символы: {list(s)}")
    print("-" * 50)

    for i in range(1, n):
        print(f"\ni = {i}, символ s[{i}] = '{s[i]}'")

        j = pi[i - 1]
        print(f"  Начальное j = π[{i-1}] = {j}")

        steps = 0
        while j > 0 and s[i] != s[j]:
            steps += 1
            print(f"  Шаг {steps}: j={j}, s[{i}]='{s[i]}' != s[{j}]='{s[j]}'")
            print(f"    j = π[{j-1}] = {pi[j-1]}")
            j = pi[j - 1]

        if s[i] == s[j]:
            print(f"  s[{i}]='{s[i]}' == s[{j}]='{s[j]}', увеличиваем j")
            j += 1

        pi[i] = j
        print(f"  π[{i}] = {j}")

    print(f"\nРезультат: π = {pi}")
    return pi


# Вспомогательные функции
def is_prefix_function_valid(pi, s):
    """
    Проверка корректности вычисленной префикс-функции.

    Args:
        pi: вычисленная префикс-функция
        s: исходная строка

    Returns:
        True если префикс-функция корректна, иначе False
    """
    n = len(s)
    if len(pi) != n:
        return False

    for i in range(n):
        # π[0] всегда должно быть 0
        if i == 0 and pi[i] != 0:
            return False

        # π[i] должно быть длиной префикса, который является суффиксом s[0..i]
        if pi[i] > 0:
            prefix = s[: pi[i]]
            suffix = s[i - pi[i] + 1 : i + 1]
            if prefix != suffix:
                return False

        # π[i] должно быть максимальным
        if pi[i] < n:
            for k in range(pi[i] + 1, min(i, n) + 1):
                if s[:k] == s[i - k + 1 : i + 1]:
                    return False

    return True


def find_all_prefix_occurrences(s, prefix):
    """
    Нахождение всех вхождений префикса в строку с использованием префикс-функции.

    Args:
        s: строка для поиска
        prefix: префикс для поиска

    Returns:
        Список позиций, где заканчиваются вхождения префикса
    """
    if not prefix:
        return list(range(len(s) + 1))

    # Создаем строку prefix + '#' + s для вычисления префикс-функции
    combined = prefix + "#" + s
    pi = prefix_function(combined)

    result = []
    prefix_len = len(prefix)

    # Ищем позиции, где префикс-функция равна длине префикса
    for i in range(prefix_len + 1, len(combined)):
        if pi[i] == prefix_len:
            # Вычисляем позицию в исходной строке s
            pos_in_s = i - 2 * prefix_len
            result.append(pos_in_s)

    return result
