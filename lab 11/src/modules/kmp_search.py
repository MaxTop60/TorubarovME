from prefix_function import prefix_function


def kmp_search(text, pattern):
    if not pattern:
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)

    result = []
    j = 0

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == m:
            result.append(i - m + 1)
            j = pi[j - 1]  # Для поиска перекрывающихся вхождений

    return result


def kmp_search_with_preprocessing(pattern):
    """
    Подготовка префикс-функции для pattern для многократного поиска.

    Args:
        pattern: строка для поиска

    Returns:
        Функция для поиска pattern в текстах

    Сложность подготовки: O(m)
    """
    pi = prefix_function(pattern)
    m = len(pattern)

    def search_in_text(text):
        """Поиск предварительно обработанного pattern в text."""
        n = len(text)
        result = []
        j = 0

        for i in range(n):
            while j > 0 and text[i] != pattern[j]:
                j = pi[j - 1]

            if text[i] == pattern[j]:
                j += 1

            if j == m:
                result.append(i - m + 1)
                j = pi[j - 1]

        return result

    return search_in_text


def kmp_search_with_count(text, pattern):
    """
    KMP поиск с подсчетом количества сравнений символов.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Кортеж (позиции вхождений, количество сравнений)
    """
    if not pattern:
        return (list(range(len(text) + 1)), 0)

    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)

    result = []
    j = 0
    comparisons = 0

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]
            comparisons += 1

        comparisons += 1
        if text[i] == pattern[j]:
            j += 1

        if j == m:
            result.append(i - m + 1)
            j = pi[j - 1]

    return result, comparisons


def kmp_search_all_occurrences(text, pattern):
    """
    KMP поиск всех вхождений, включая перекрывающиеся.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Список позиций всех вхождений (включая перекрывающиеся)
    """
    if not pattern:
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)

    result = []
    j = 0

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == m:
            result.append(i - m + 1)
            # Для поиска перекрывающихся вхождений используем π[m-1]
            j = pi[j - 1]

    return result


def kmp_search_with_visualization(text, pattern):
    """
    KMP поиск с визуализацией процесса.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Список позиций вхождений
    """
    print(f"Поиск '{pattern}' в '{text}' с использованием KMP")
    print("-" * 60)

    if not pattern:
        print("Пустой pattern - возвращаем все позиции")
        return list(range(len(text) + 1))

    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)

    print(f"Префикс-функция для '{pattern}': {pi}")
    print("-" * 60)

    result = []
    j = 0

    for i in range(n):
        print(f"\ni = {i}, text[{i}] = '{text[i]}'")
        print(f"  Текущее состояние: j = {j}")

        while j > 0 and text[i] != pattern[j]:
            print(
                f"  Несовпадение: text[{i}]='{text[i]}' != pattern[{j}]='{pattern[j]}'"
            )
            print(f"    Сдвигаем: j = π[{j-1}] = {pi[j-1]}")
            j = pi[j - 1]

        if text[i] == pattern[j]:
            print(f"  Совпадение: text[{i}]='{text[i]}' == pattern[{j}]='{pattern[j]}'")
            print(f"    Увеличиваем j: {j} -> {j+1}")
            j += 1
        else:
            print(f"  Несовпадение при j=0, остаемся на j=0")

        if j == m:
            position = i - m + 1
            print(f"  ✓ Найдено вхождение на позиции {position}")
            result.append(position)
            print(f"    Продолжаем с j = π[{j-1}] = {pi[j-1]}")
            j = pi[j - 1]

    print(f"\nНайдено {len(result)} вхождений: {result}")
    return result


# Вспомогательные функции для сравнения алгоритмов
def naive_string_search(text, pattern):
    """
    Наивный алгоритм поиска подстроки.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Список позиций вхождений

    Сложность:
        Время: O(n*m) в худшем случае
        Память: O(1)
    """
    n, m = len(text), len(pattern)
    result = []

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break

        if match:
            result.append(i)

    return result


def compare_kmp_vs_naive(text, pattern):
    """
    Сравнение KMP и наивного алгоритма.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Словарь с результатами сравнения
    """
    import timeit

    # KMP поиск
    kmp_time = timeit.timeit(lambda: kmp_search(text, pattern), number=100) / 100

    kmp_result = kmp_search(text, pattern)

    # Наивный поиск
    naive_time = (
        timeit.timeit(lambda: naive_string_search(text, pattern), number=100) / 100
    )

    naive_result = naive_string_search(text, pattern)

    # Проверка корректности
    correct = kmp_result == naive_result

    return {
        "kmp_time": kmp_time,
        "naive_time": naive_time,
        "kmp_result": kmp_result,
        "naive_result": naive_result,
        "correct": correct,
        "speedup": naive_time / kmp_time if kmp_time > 0 else float("inf"),
    }
