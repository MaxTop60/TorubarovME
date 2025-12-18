"""
Реализация алгоритма Бойера-Мура для поиска подстроки.
"""

from typing import List


def boyer_moore_search(text: str, pattern: str) -> List[int]:
    """
    Алгоритм Бойера-Мура для поиска всех вхождений подстроки в тексте.
    Исправленная версия с правильной обработкой плохого символа.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    result = []

    # 1. Таблица плохого символа (стандартная реализация)
    # Храним последнее вхождение каждого символа в паттерне
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # 2. Фаза поиска
    s = 0  # Сдвиг паттерна относительно текста
    while s <= n - m:
        # Сравниваем справа налево
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # Нашли вхождение
            result.append(s)
            # Сдвигаем на 1 для поиска следующего вхождения
            s += 1
        else:
            # Вычисляем сдвиг по правилу плохого символа
            # Если символ есть в таблице, сдвигаем так, чтобы совместить
            # последнее вхождение этого символа в паттерне
            mismatch_char = text[s + j]
            if mismatch_char in bad_char:
                bad_char_shift = max(1, j - bad_char[mismatch_char])
            else:
                # Если символа нет в паттерне, сдвигаем на j+1
                bad_char_shift = j + 1

            s += bad_char_shift

    return result


def boyer_moore_search_with_good_suffix(text: str, pattern: str) -> List[int]:
    """
    Алгоритм Бойера-Мура с использованием обеих эвристик.
    Более сложная, но более эффективная версия.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    result = []

    # 1. Таблица плохого символа
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # 2. Таблица хорошего суффикса (упрощенная)
    # Сначала вычисляем префикс-функцию для обычной и обратной строки
    def compute_prefix(s: str) -> List[int]:
        """Вычисляет префикс-функцию для строки."""
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        return pi

    # Для хорошего суффикса создаем упрощенную таблицу
    good_suffix = [0] * (m + 1)

    # Случай 1: суффикс совпадает с префиксом
    pi = compute_prefix(pattern)
    pi_rev = compute_prefix(pattern[::-1])

    for i in range(m):
        j = m - pi_rev[i]
        if good_suffix[j] == 0:
            good_suffix[j] = i - pi_rev[i] + 1

    # Случай 2: часть суффикса совпадает с префиксом
    for i in range(1, m + 1):
        if good_suffix[i] == 0:
            good_suffix[i] = good_suffix[i - 1]

    # Для полного совпадения
    if good_suffix[0] == 0:
        good_suffix[0] = 1

    # 3. Фаза поиска
    s = 0
    while s <= n - m:
        j = m - 1

        # Сравниваем справа налево
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # Нашли вхождение
            result.append(s)
            s += good_suffix[0]
        else:
            # Вычисляем оба сдвига
            mismatch_char = text[s + j]

            # Сдвиг по плохому символу
            if mismatch_char in bad_char:
                bad_char_shift = max(1, j - bad_char[mismatch_char])
            else:
                bad_char_shift = j + 1

            # Сдвиг по хорошему суффиксу
            good_suffix_shift = good_suffix[j + 1] if j + 1 <= m else 1

            # Выбираем максимальный сдвиг
            s += max(bad_char_shift, good_suffix_shift)

    return result
