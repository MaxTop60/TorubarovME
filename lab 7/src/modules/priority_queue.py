from heap import MinHeap


class PriorityQueue:
    """Очередь с приоритетом на основе MinHeap"""

    def __init__(self):
        self.heap = MinHeap()

    def enqueue(self, item, priority):
        """
        Добавление элемента в очередь с приоритетом
        Более низкое значение priority = более высокий приоритет
        """
        # Создаем кортеж (приоритет, элемент) для сравнения в куче
        queue_item = (priority, item)
        self.heap.insert(queue_item)

    def dequeue(self):
        """
        Извлечение элемента с наивысшим приоритетом
        (наименьшим значением priority)
        Возвращает элемент без приоритета
        """
        if self.is_empty():
            return "Очередь пуста"

        # Извлекаем кортеж (приоритет, элемент) и возвращаем только элемент
        priority, item = self.heap.extract()
        return item

    def peek(self):
        """
        Просмотр элемента с наивысшим приоритетом без извлечения
        """
        if self.is_empty():
            return "Очередь пуста"

        priority, item = self.heap.peek()
        return item

    def is_empty(self):
        """Проверка пустоты очереди"""
        return len(self.heap.tree) == 0

    def size(self):
        """Размер очереди"""
        return len(self.heap.tree)

    def __str__(self):
        """Строковое представление очереди"""
        if self.is_empty():
            return "PriorityQueue: пусто"

        items = []
        for priority, item in self.heap.tree:
            items.append(f"{item}(приоритет:{priority})")
        return "PriorityQueue: " + " <- ".join(items)
