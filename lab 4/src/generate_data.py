import random
from typing import List, Dict


def generate_random_array(size: int) -> List[int]:
    """Генерация случайного массива"""
    return [random.randint(0, size * 10) for _ in range(size)]  # O(n)


def generate_sorted_array(size: int) -> List[int]:
    """Генерация отсортированного массива"""
    return list(range(size))  # O(n)


def generate_reversed_array(size: int) -> List[int]:
    """Генерация массива, отсортированного в обратном порядке"""
    return list(range(size, 0, -1))  # O(n)


def generate_almost_sorted_array(
        size: int, swap_percentage: float = 0.05) -> List[int]:
    """Генерация почти отсортированного массива"""
    arr = list(range(size))  # O(n)
    num_swaps = int(size * swap_percentage)  # O(1)

    for _ in range(num_swaps):  # O(n) итераций
        i = random.randint(0, size - 1)  # O(1)
        j = random.randint(0, size - 1)  # O(1)
        arr[i], arr[j] = arr[j], arr[i]  # O(1)

    return arr  # O(1)
    # Общая сложность: O(n)


def generate_test_data(sizes: List[int]) -> Dict[str, Dict[str, List[int]]]:
    """
    Генерация всех тестовых данных
    Возвращает словарь: тип_данных -> размер -> массив
    """
    test_data = {
        'random': {},
        'sorted': {},
        'reversed': {},
        'almost_sorted': {}
    }

    for size in sizes:  # O(k * n) где k - количество размеров
        test_data['random'][size] = generate_random_array(size)  # O(n)
        test_data['sorted'][size] = generate_sorted_array(size)  # O(n)
        test_data['reversed'][size] = generate_reversed_array(size)  # O(n)
        test_data['almost_sorted'][
            size] = generate_almost_sorted_array(size)  # O(n)

    return test_data  # O(1)
    # Общая сложность: O(k * n)
