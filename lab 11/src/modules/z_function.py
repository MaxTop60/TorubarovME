def z_function_naive(s):
    n = len(s)
    z = [0] * n

    for i in range(1, n):
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

    return z


def z_function(s):
    """
    Эффективное вычисление Z-функции за O(n).

    По определению:
    z[0] = 0 (обычнsо не используется или равно 0)
    z[i] = наибольшее k такое, что s[0:k] == s[i:i+k]
    """
    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    # z[0] = 0 (по определению, обычно не используется)

    l = r = 0

    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1

    return z


def z_function_optimized(s):
    """
    Оптимизированная версия Z-функции.

    Args:
        s: входная строка

    Returns:
        список z - Z-функция для строки s

    Сложность:
        Время: O(n)
        Память: O(n)
    """
    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    z[0] = n  # По определению, z[0] = n

    l = r = 0

    for i in range(1, n):
        if i > r:
            # i вне z-блока, вычисляем с нуля
            l = r = i
            while r < n and s[r - l] == s[r]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            # i внутри z-блока
            k = i - l
            if z[k] < r - i + 1:
                # z[i] ограничено текущим блоком
                z[i] = z[k]
            else:
                # Нужно продолжить сравнение за пределами блока
                l = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1

    return z


def z_function_with_visualization(s):
    """
    Вычисление Z-функции с выводом промежуточных шагов.

    Args:
        s: входная строка

    Returns:
        список z - Z-функция для строки s

    Сложность:
        Время: O(n)
        Память: O(n)
    """
    n = len(s)
    z = [0] * n
    l = r = 0

    print(f"Вычисление Z-функции для строки: '{s}'")
    print(f"Индексы: {list(range(n))}")
    print(f"Символы: {list(s)}")
    print("-" * 60)

    for i in range(1, n):
        print(f"\ni = {i}, символ s[{i}] = '{s[i]}'")
        print(f"  Текущий z-блок: l={l}, r={r}")

        if i <= r:
            print(f"  i внутри z-блока")
            print(f"  z[i] = min(r-i+1={r-i+1}, z[i-l]=z[{i-l}]={z[i-l]})")
            z[i] = min(r - i + 1, z[i - l])
            print(f"  z[{i}] = {z[i]} (предварительно)")
        else:
            print(f"  i вне z-блока, начинаем с 0")
            z[i] = 0

        print(f"  Сравниваем символы начиная с позиции {z[i]}:")

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            print(f"    s[{z[i]}]='{s[z[i]]}' == s[{i+z[i]}]='{s[i+z[i]]}'")
            z[i] += 1

        print(f"  Финальное z[{i}] = {z[i]}")

        if i + z[i] - 1 > r:
            old_r = r
            l, r = i, i + z[i] - 1
            print(f"  Обновляем z-блок: l={l}, r={r} (было r={old_r})")

    print(f"\nРезультат: z = {z}")
    return z


def search_with_z_function(text, pattern):
    """
    Поиск подстроки с использованием Z-функции.

    Args:
        text: строка для поиска
        pattern: искомая подстрока

    Returns:
        Список позиций, где начинаются вхождения pattern в text

    Сложность:
        Время: O(n + m), где n - длина text, m - длина pattern
        Память: O(n + m)
    """
    if not pattern:
        return list(range(len(text) + 1))

    # Создаем строку pattern + '$' + text
    combined = pattern + "$" + text
    z = z_function(combined)

    result = []
    m = len(pattern)

    # Ищем позиции, где Z[i] = m (длина pattern)
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            # Пересчитываем позицию в исходном text
            pos_in_text = i - m - 1
            result.append(pos_in_text)

    return result


def find_all_palindromes(s):
    """
    Нахождение всех палиндромов в строке с использованием Z-функции.

    Args:
        s: входная строка

    Returns:
        Список кортежей (начало, длина) для всех палиндромов

    Сложность:
        Время: O(n)
        Память: O(n)
    """
    n = len(s)
    result = []

    # Для нечетных палиндромов
    s_odd = s + "#" + s[::-1]
    z_odd = z_function(s_odd)

    # Для четных палиндромов
    s_even = s + "$" + s[::-1]
    z_even = z_function(s_even)

    # Находим палиндромы с центром в каждой позиции
    for i in range(n):
        # Нечетные палиндромы (центр - символ)
        radius = z_odd[n + 1 + (n - 1 - i)]
        if radius > 0:
            start = i - radius // 2
            length = radius
            result.append((start, length))

        # Четные палиндромы (центр - между символами)
        radius = z_even[n + 1 + (n - 1 - i)]
        if radius > 0:
            start = i - radius // 2
            length = radius
            result.append((start, length))

    # Добавляем тривиальные палиндромы (одиночные символы)
    for i in range(n):
        result.append((i, 1))

    # Убираем дубликаты и сортируем
    result = list(set(result))
    result.sort()

    return result


def is_z_function_valid(z, s):
    """
    Проверка корректности вычисленной Z-функции.

    Args:
        z: вычисленная Z-функция
        s: исходная строка

    Returns:
        True если Z-функция корректна, иначе False
    """
    n = len(s)
    if len(z) != n:
        return False

    if n > 0 and z[0] != n:
        return False

    for i in range(1, n):
        # Проверяем определение Z-функции
        for j in range(z[i]):
            if i + j >= n or s[j] != s[i + j]:
                return False

        # Проверяем максимальность
        if i + z[i] < n and s[z[i]] == s[i + z[i]]:
            return False

    return True


def compare_with_prefix_function(s):
    """
    Сравнение Z-функции и префикс-функции.

    Args:
        s: входная строка

    Returns:
        Словарь с результатами сравнения
    """
    from prefix_function import prefix_function

    z = z_function(s)
    pi = prefix_function(s)

    # Z-функцию можно получить из префикс-функции и наоборот
    n = len(s)

    # Преобразование Z в π
    pi_from_z = [0] * n
    for i in range(1, n):
        for j in range(z[i]):
            pi_from_z[i + j] = max(pi_from_z[i + j], j + 1)

    # Преобразование π в Z
    z_from_pi = [0] * n
    if n > 0:
        z_from_pi[0] = n

    for i in range(1, n):
        if pi[i] > 0:
            z_from_pi[i - pi[i] + 1] = max(z_from_pi[i - pi[i] + 1], pi[i])

    return {
        "z_function": z,
        "prefix_function": pi,
        "pi_from_z": pi_from_z,
        "z_from_pi": z_from_pi,
        "z_pi_equal": pi == pi_from_z,
        "pi_z_equal": z == z_from_pi,
    }
