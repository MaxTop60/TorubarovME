import heapq


def interval_scheduling(intervals):
    """
    Выбирает максимальное количество непересекающихся интервалов.

    Args:
        intervals: список интервалов в формате (start, end)

    Returns:
        список выбранных интервалов
    """
    # Сортируем интервалы по времени окончания
    intervals.sort(key=lambda x: x[1])

    selected = []
    last_end = -float("inf")

    for interval in intervals:
        start, end = interval
        # Если интервал начинается после окончания последнего выбранного
        if start >= last_end:
            selected.append(interval)
            last_end = end

    return selected
    # Временная сложность: O(n log n) из-за сортировки
    # Корректность жадного алгоритма: выбор интервала,
    # который заканчивается раньше всех,
    # максимизирует оставшееся доступное время для выбора последующих
    # интервалов.
    # Это локально оптимальный выбор приводит к глобально оптимальному решению.


def fractional_knapsack(capacity, items):
    """
    Решает задачу о непрерывном рюкзаке.

    Args:
        capacity: вместимость рюкзака
        items: список предметов в формате (weight, value)

    Returns:
        кортеж (максимальная стоимость,
        список взятых предметов в формате (вес, стоимость, доля))
    """
    # Вычисляем удельную стоимость для каждого предмета
    items_with_ratio = [(
        weight,
        value,
        value / weight
        ) for weight, value in items]

    # Сортируем по удельной стоимости в порядке убывания
    items_with_ratio.sort(key=lambda x: x[2], reverse=True)

    total_value = 0.0
    taken_items = []

    for weight, value, ratio in items_with_ratio:
        if capacity == 0:
            break

        # Берем столько, сколько можем
        take_weight = min(weight, capacity)
        fraction = take_weight / weight
        total_value += value * fraction

        taken_items.append((weight, value, fraction))
        capacity -= take_weight

    return total_value, taken_items


# Временная сложность: O(n log n) из-за сортировки
# Корректность жадного алгоритма:
# выбор предметов с максимальной удельной стоимостью
# гарантирует, что каждый взятый вес приносит максимально возможную стоимость.
# Для непрерывной задачи это приводит к оптимальному решению.


class HuffmanNode:
    """Узел дерева Хаффмана"""

    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        # Для корректной работы heapq
        return self.freq < other.freq


def huffman_coding(frequencies):
    """
    Строит оптимальный префиксный код Хаффмана.

    Args:
        frequencies: словарь {символ: частота}

    Returns:
        словарь {символ: код}
    """
    if len(frequencies) == 0:
        return {}

    # Создаем начальные узлы для каждого символа
    heap = []
    for char, freq in frequencies.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)

    # Построение дерева Хаффмана
    while len(heap) > 1:
        # Извлекаем два узла с наименьшей частотой
        left = heapq.heappop(heap)  # O(log n)
        right = heapq.heappop(heap)  # O(log n)

        # Создаем новый объединенный узел
        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)  # O(log n)

    # Корень дерева
    root = heap[0]

    # Генерация кодов обходом дерева
    codes = {}

    def generate_codes(node, current_code=""):
        if node is None:
            return

        # Если это листовой узел (символ)
        if node.char is not None:
            codes[node.char] = current_code
            return

        # Рекурсивно обходим левое и правое поддерево
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")

    generate_codes(root)

    return codes

    # Временная сложность: O(n log n), где n - количество символов
    # Основные операции с кучей имеют сложность O(log n), выполняются O(n) раз
    # Корректность жадного алгоритма: на каждом шаге объединяются два символа
    # с наименьшей частотой,
    # что минимизирует ожидаемую длину кода. Это локально оптимальный выбор
    # (слияние наименее частых символов)
    # приводит к глобально оптимальному префиксному коду.
