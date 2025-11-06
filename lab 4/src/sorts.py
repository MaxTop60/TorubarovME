from typing import List, Callable


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Сортировка пузырьком
    """
    arr = arr.copy()  # O(n)
    n = len(arr)  # O(1)

    for i in range(n):  # O(n)
        swapped = False  # O(1)
        for j in range(0, n - i - 1):  # O(n)
            if arr[j] > arr[j + 1]:  # O(1)
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # O(1)
                swapped = True  # O(1)
        if not swapped:  # O(1)
            break  # O(1)
    return arr  # O(1)
    # Общая временная сложность: O(n²) в худшем случае, O(n) в лучшем
    # Пространственная сложность: O(1)
    # Глубина: O(1) - не рекурсивная


def selection_sort(arr: List[int]) -> List[int]:
    """
    Сортировка выбором
    """
    arr = arr.copy()  # O(n)
    n = len(arr)  # O(1)

    for i in range(n):  # O(n)
        min_idx = i  # O(1)
        for j in range(i + 1, n):  # O(n)
            if arr[j] < arr[min_idx]:  # O(1)
                min_idx = j  # O(1)
        arr[i], arr[min_idx] = arr[min_idx], arr[i]  # O(1)
    return arr  # O(1)
    # Общая временная сложность: O(n²)
    # Пространственная сложность: O(1)
    # Глубина: O(1) - не рекурсивная


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Сортировка вставками
    """
    arr = arr.copy()  # O(n)

    for i in range(1, len(arr)):  # O(n) итераций
        key = arr[i]  # O(1)
        j = i - 1  # O(1)
        while j >= 0 and arr[j] > key:  # O(n) итераций в худшем случае
            arr[j + 1] = arr[j]  # O(1)
            j -= 1  # O(1)
        arr[j + 1] = key  # O(1)
    return arr  # O(1)
    # Общая временная сложность: O(n²) в худшем случае, O(n) в лучшем
    # Пространственная сложность: O(1)
    # Глубина: O(1) - не рекурсивная


def merge_sort(arr: List[int]) -> List[int]:
    """
    Сортировка слиянием
    """
    if len(arr) <= 1:  # O(1)
        return arr.copy()  # O(n)

    mid = len(arr) // 2  # O(1)
    left = merge_sort(arr[:mid])  # O(n log n) - рекурсивный вызов
    right = merge_sort(arr[mid:])  # O(n log n) - рекурсивный вызов

    return _merge(left, right)  # O(n)


def _merge(left: List[int], right: List[int]) -> List[int]:
    """Вспомогательная функция для слияния двух отсортированных массивов"""
    result = []  # O(1)
    i = j = 0  # O(1)

    while i < len(left) and j < len(right):  # O(n) итераций
        if left[i] <= right[j]:  # O(1)
            result.append(left[i])  # O(1)
            i += 1  # O(1)
        else:
            result.append(right[j])  # O(1)
            j += 1  # O(1)

    result.extend(left[i:])  # O(n)
    result.extend(right[j:])  # O(n)
    return result  # O(1)
    # Общая временная сложность merge_sort: O(n log n)
    # Пространственная сложность: O(n)
    # Глубина рекурсии: O(log n)


def quick_sort(arr: List[int]) -> List[int]:
    """
    Быстрая сортировка
    """
    if len(arr) <= 1:  # O(1)
        return arr.copy()  # O(n)

    pivot = _median_of_three(arr)  # O(1)

    left = [x for x in arr if x < pivot]  # O(n)
    middle = [x for x in arr if x == pivot]  # O(n)
    right = [x for x in arr if x > pivot]  # O(n)

    return quick_sort(left) + middle + quick_sort(right)
    # O(n log n) в среднем, O(n²) в худшем


def _median_of_three(arr: List[int]) -> int:
    """Выбор медианы из первого, среднего и последнего элементов"""
    first = arr[0]  # O(1)
    middle = arr[len(arr) // 2]  # O(1)
    last = arr[-1]  # O(1)

    if first <= middle <= last or last <= middle <= first:  # O(1)
        return middle  # O(1)
    elif middle <= first <= last or last <= first <= middle:  # O(1)
        return first  # O(1)
    else:
        return last  # O(1)
    # Общая временная сложность quick_sort: O(n log n) в среднем,
    # O(n²) в худшем
    # Пространственная сложность: O(n) в худшем, O(log n) в среднем
    # Глубина рекурсии: O(n) в худшем, O(log n) в среднем


def is_sorted(arr: List[int]) -> bool:
    """Проверка, отсортирован ли массив"""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))  # O(n)


def test_sorting_algorithm(sort_func: Callable, arr: List[int]) -> bool:
    """Тестирование алгоритма сортировки"""
    original = arr.copy()  # O(n)
    sorted_arr = sort_func(arr)  # Зависит от алгоритма
    result = is_sorted(sorted_arr)  # O(n)

    if not result:
        print(f"Ошибка в алгоритме {sort_func.__name__}")
        print(f"Исходный: {original[:10]}...")
        print(f"Результат: {sorted_arr[:10]}...")

    return result


# Словарь всех алгоритмов для удобного тестирования
SORTING_ALGORITHMS = {
    'bubble_sort': bubble_sort,
    'selection_sort': selection_sort,
    'insertion_sort': insertion_sort,
    'merge_sort': merge_sort,
    'quick_sort': quick_sort
}
