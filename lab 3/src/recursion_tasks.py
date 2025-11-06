import os
import sys

sys.setrecursionlimit(10000)  # Установка максимального размера стека рекурсии


# Задание 1
def task_1():
    def binary_search_recursive(arr, target, left=0, right=None):
        # Рекурсивный алгоритм бинарного поиска.

        if right is None:
            right = len(arr) - 1  # O(1)

        if left > right:
            return -1  # Элемент не найден O(1)

        mid = (left + right) // 2  # Средний индекс O(1)

        if arr[mid] == target:
            return mid  # Элемент найден O(1)
        elif arr[mid] < target:
            return binary_search_recursive(
                arr, target, mid + 1, right
            )  # Искать справа O(log n)
        else:
            return binary_search_recursive(
                arr, target, left, mid - 1
            )  # Искать слева O(log n)

    sorted_list = [1, 3, 5, 7, 9, 11, 13]
    print(binary_search_recursive(sorted_list, 7))
    print(binary_search_recursive(sorted_list, 6))


# Задание 2
def task_2():
    def tree(path, prefix="", is_last=True, depth=0):
        # Рекурсивный обход файловой системы.
        global max_depth
        if not os.path.exists(path):
            print(f"Путь '{path}' не существует.")  # O(1)
            return

        items = os.listdir(path)  # O(n)
        for i, item in enumerate(items):  # O(n)
            full_path = os.path.join(path, item)  # O(1)

            # Символы для визуального оформления дерева
            connector = "└── " if is_last and i == len(items) - 1 else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")  # O(1)

            print(prefix + connector + item)  # O(1)

            if os.path.isdir(full_path):
                if depth > max_depth:
                    max_depth = depth
                tree(full_path, new_prefix, i == len(items) - 1, depth + 1)
        # Общая сложность O(n)

    # Пример использования
    tree("D://University/3 курс")
    print(f"\nМаксимальная глубина рекурсии: {max_depth}")


# Задание 3
def task_3():
    def hanoi(n, source, target, auxiliary):
        # Рекурсивное решение задачи Ханойские башни.
        if n == 1:  # O(1)
            print(f"Переместить диск 1 с {source} на {target}")
            return

        # Шаг 1: Переместить n-1 дисков из source на auxiliary,
        # используя target как вспомогательный
        hanoi(n - 1, source, auxiliary, target)  # O(n - 1)

        # Шаг 2: Переместить n-й диск из source на target
        print(f"Переместить диск {n} с {source} на {target}")

        # Шаг 3: Переместить n-1 дисков из auxiliary на target,
        # используя source как вспомогательный
        hanoi(n - 1, auxiliary, target, source)  # O(n - 1)
        # Общая сложность O(2^n)

    hanoi(4, "A", "C", "B")


# Инициализация глобальной переменной для обхода файловой системыЦ
max_depth = 0
task_2()
