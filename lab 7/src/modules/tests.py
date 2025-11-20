import unittest

from heap import MinHeap
from heapsort import heapsort, heapsort_inplace
from priority_queue import PriorityQueue


class TestMinHeap(unittest.TestCase):
    """Тесты для класса MinHeap"""

    def setUp(self):
        self.heap = MinHeap()

    def test_empty_heap(self):
        """Тест пустой кучи"""
        self.assertEqual(len(self.heap.tree), 0)
        self.assertEqual(self.heap.peek(), "Куча пуста")
        self.assertTrue(self.heap.tree == [])

    def test_insert_single_element(self):
        """Тест вставки одного элемента"""
        self.heap.insert(5)
        self.assertEqual(len(self.heap.tree), 1)
        self.assertEqual(self.heap.peek(), 5)

    def test_insert_multiple_elements(self):
        """Тест вставки нескольких элементов"""
        elements = [5, 3, 8, 1, 2]
        for elem in elements:
            self.heap.insert(elem)

        # Проверяем свойство Min-Heap: корень должен быть минимальным
        self.assertEqual(self.heap.peek(), 1)
        self.assertEqual(len(self.heap.tree), 5)

    def test_heap_property_after_insert(self):
        """Проверка свойства кучи после вставки"""
        test_data = [10, 5, 15, 3, 7, 12, 1]
        for value in test_data:
            self.heap.insert(value)
            self._verify_min_heap_property(self.heap.tree)

    def test_extract_min(self):
        """Тест извлечения минимального элемента"""
        elements = [5, 3, 8, 1, 2]
        for elem in elements:
            self.heap.insert(elem)

        min_val = self.heap.extract()
        self.assertEqual(min_val, 1)
        self.assertEqual(len(self.heap.tree), 4)
        self.assertEqual(self.heap.peek(), 2)

    def test_extract_all_elements(self):
        """Тест последовательного извлечения всех элементов"""
        elements = [5, 3, 8, 1, 2]
        for elem in elements:
            self.heap.insert(elem)

        extracted = []
        while self.heap.tree:
            extracted.append(self.heap.extract())

        self.assertEqual(extracted, [1, 2, 3, 5, 8])  # Отсортированные по возрастанию
        self.assertEqual(len(self.heap.tree), 0)

    def test_heap_property_after_extract(self):
        """Проверка свойства кучи после каждого извлечения"""
        test_data = [10, 5, 15, 3, 7, 12, 1, 20, 8]
        for value in test_data:
            self.heap.insert(value)

        while self.heap.tree:
            self.heap.extract()
            if self.heap.tree:  # Проверяем свойство, если куча не пуста
                self._verify_min_heap_property(self.heap.tree)

    def test_build_heap_from_array(self):
        """Тест построения кучи из массива"""
        test_array = [9, 4, 7, 1, 5, 3, 8, 2, 6]
        self.heap.build_heap(test_array)

        self.assertEqual(len(self.heap.tree), 9)
        self._verify_min_heap_property(self.heap.tree)
        self.assertEqual(self.heap.peek(), 1)  # Минимальный элемент должен быть в корне

    def test_large_heap(self):
        """Тест с большим количеством элементов"""
        large_array = list(range(100, 0, -1))  # [100, 99, 98, ..., 1]
        self.heap.build_heap(large_array)

        self.assertEqual(len(self.heap.tree), 100)
        self._verify_min_heap_property(self.heap.tree)
        self.assertEqual(self.heap.peek(), 1)

        # Извлекаем первые 10 элементов
        for i in range(1, 11):
            self.assertEqual(self.heap.extract(), i)

    def _verify_min_heap_property(self, heap_array):
        """Вспомогательный метод для проверки свойства Min-Heap"""
        n = len(heap_array)
        for i in range(n // 2):
            left = 2 * i + 1
            right = 2 * i + 2

            # Проверяем, что родитель меньше или равен левому потомку
            if left < n:
                self.assertTrue(
                    heap_array[i] <= heap_array[left],
                    f"Нарушение свойства кучи: {heap_array[i]} > {heap_array[left]} (индексы {i} -> {left})",
                )

            # Проверяем, что родитель меньше или равен правому потомку
            if right < n:
                self.assertTrue(
                    heap_array[i] <= heap_array[right],
                    f"Нарушение свойства кучи: {heap_array[i]} > {heap_array[right]} (индексы {i} -> {right})",
                )


class TestHeapsort(unittest.TestCase):
    """Тесты для функций Heapsort"""

    def test_heapsort_empty(self):
        """Тест сортировки пустого массива"""
        result = heapsort([])
        self.assertEqual(result, [])

    def test_heapsort_single_element(self):
        """Тест сортировки массива из одного элемента"""
        result = heapsort([5])
        self.assertEqual(result, [5])

    def test_heapsort_sorted_array(self):
        """Тест сортировки уже отсортированного массива"""
        input_array = [1, 2, 3, 4, 5]
        result = heapsort(input_array)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_heapsort_reverse_sorted(self):
        """Тест сортировки массива, отсортированного в обратном порядке"""
        input_array = [5, 4, 3, 2, 1]
        result = heapsort(input_array)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_heapsort_random_array(self):
        """Тест сортировки случайного массива"""
        input_array = [9, 3, 7, 1, 5, 8, 2, 4, 6]
        result = heapsort(input_array)
        self.assertEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_heapsort_duplicates(self):
        """Тест сортировки массива с дубликатами"""
        input_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        result = heapsort(input_array)
        self.assertEqual(result, [1, 1, 2, 3, 3, 4, 5, 5, 6, 9])

    def test_heapsort_inplace_empty(self):
        """Тест in-place сортировки пустого массива"""
        array = []
        heapsort_inplace(array)
        self.assertEqual(array, [])

    def test_heapsort_inplace_sorted(self):
        """Тест in-place сортировки отсортированного массива"""
        array = [1, 2, 3, 4, 5]
        heapsort_inplace(array)
        self.assertEqual(array, [5, 4, 3, 2, 1])  # Убывающий порядок для Min-Heap

    def test_heapsort_inplace_random(self):
        """Тест in-place сортировки случайного массива"""
        array = [9, 3, 7, 1, 5, 8, 2, 4, 6]
        heapsort_inplace(array)
        self.assertEqual(array, [9, 8, 7, 6, 5, 4, 3, 2, 1])  # Убывающий порядок

    def test_heapsort_correctness(self):
        """Сравнение с встроенной сортировкой Python"""
        test_arrays = [
            [1],
            [2, 1],
            [5, 3, 8, 1, 2],
            [9, 3, 7, 1, 5, 8, 2, 4, 6],
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            list(range(50, 0, -1)),
        ]

        for array in test_arrays:
            with self.subTest(array=array):
                expected = sorted(array)
                result = heapsort(array)
                self.assertEqual(result, expected)


class TestPriorityQueue(unittest.TestCase):
    """Тесты для класса PriorityQueue"""

    def setUp(self):
        self.pq = PriorityQueue()

    def test_empty_queue(self):
        """Тест пустой очереди"""
        self.assertTrue(self.pq.is_empty())
        self.assertEqual(self.pq.size(), 0)
        self.assertEqual(self.pq.dequeue(), "Очередь пуста")
        self.assertEqual(self.pq.peek(), "Очередь пуста")

    def test_enqueue_dequeue_single(self):
        """Тест добавления и извлечения одного элемента"""
        self.pq.enqueue("task1", 1)
        self.assertFalse(self.pq.is_empty())
        self.assertEqual(self.pq.size(), 1)

        item = self.pq.dequeue()
        self.assertEqual(item, "task1")
        self.assertTrue(self.pq.is_empty())

    def test_priority_ordering(self):
        """Тест правильного порядка приоритетов"""
        # Добавляем элементы в случайном порядке
        self.pq.enqueue("low", 3)
        self.pq.enqueue("high", 1)
        self.pq.enqueue("medium", 2)
        self.pq.enqueue("urgent", 0)

        # Должны извлекаться в порядке приоритета
        self.assertEqual(self.pq.dequeue(), "urgent")  # приоритет 0
        self.assertEqual(self.pq.dequeue(), "high")  # приоритет 1
        self.assertEqual(self.pq.dequeue(), "medium")  # приоритет 2
        self.assertEqual(self.pq.dequeue(), "low")  # приоритет 3

    def test_same_priority_fifo(self):
        """Тест элементов с одинаковым приоритетом"""
        self.pq.enqueue("task1", 1)
        self.pq.enqueue("task2", 1)
        self.pq.enqueue("task3", 1)

        # При одинаковом приоритете порядок может быть любым,
        # но все должны быть извлечены
        items = set()
        while not self.pq.is_empty():
            items.add(self.pq.dequeue())

        self.assertEqual(items, {"task1", "task2", "task3"})

    def test_peek_does_not_remove(self):
        """Тест что peek не удаляет элемент"""
        self.pq.enqueue("important", 1)

        self.assertEqual(self.pq.peek(), "important")
        self.assertEqual(self.pq.peek(), "important")  # Должен оставаться тем же
        self.assertEqual(self.pq.size(), 1)

        self.assertEqual(self.pq.dequeue(), "important")
        self.assertTrue(self.pq.is_empty())

    def test_mixed_operations(self):
        """Тест смешанных операций"""
        self.pq.enqueue("first", 2)
        self.pq.enqueue("urgent", 0)

        self.assertEqual(self.pq.dequeue(), "urgent")

        self.pq.enqueue("medium", 1)
        self.pq.enqueue("last", 3)

        self.assertEqual(self.pq.dequeue(), "medium")
        self.assertEqual(self.pq.dequeue(), "first")
        self.assertEqual(self.pq.dequeue(), "last")
        self.assertTrue(self.pq.is_empty())

    def test_priority_queue_property(self):
        """Проверка свойства кучи в PriorityQueue после операций"""
        priorities = [5, 1, 8, 3, 0, 2, 7, 4, 6]
        for i, priority in enumerate(priorities):
            self.pq.enqueue(f"item{i}", priority)
            # Проверяем что внутренняя куча сохраняет свойство
            self._verify_min_heap_property_for_pq(self.pq.heap.tree)

        while not self.pq.is_empty():
            self.pq.dequeue()
            if not self.pq.is_empty():
                self._verify_min_heap_property_for_pq(self.pq.heap.tree)

    def _verify_min_heap_property_for_pq(self, heap_array):
        """Проверка свойства Min-Heap для кортежей (priority, item)"""
        n = len(heap_array)
        for i in range(n // 2):
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n:
                self.assertTrue(
                    heap_array[i][0] <= heap_array[left][0],
                    f"Нарушение свойства кучи в PriorityQueue",
                )

            if right < n:
                self.assertTrue(
                    heap_array[i][0] <= heap_array[right][0],
                    f"Нарушение свойства кучи в PriorityQueue",
                )


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def test_heap_to_heapsort_integration(self):
        """Интеграционный тест: Куча -> Heapsort"""
        heap = MinHeap()
        test_data = [9, 3, 7, 1, 5, 8, 2, 4, 6]
        heap.build_heap(test_data)

        # Используем кучу для сортировки
        sorted_result = []
        while heap.tree:
            sorted_result.append(heap.extract())

        self.assertEqual(sorted_result, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_priority_queue_with_complex_items(self):
        """Тест PriorityQueue со сложными объектами"""
        pq = PriorityQueue()

        class Task:
            def __init__(self, name, duration):
                self.name = name
                self.duration = duration

            def __str__(self):
                return f"{self.name}({self.duration}min)"

        tasks = [
            Task("быстрая задача", 5),
            Task("медленная задача", 60),
            Task("средняя задача", 30),
        ]

        # Добавляем с приоритетом = длительность
        for task in tasks:
            pq.enqueue(task, task.duration)

        # Должны извлекаться в порядке возрастания длительности
        extracted = []
        while not pq.is_empty():
            extracted.append(pq.dequeue())

        self.assertEqual([t.duration for t in extracted], [5, 30, 60])


if __name__ == "__main__":
    # Запуск тестов с детальным выводом
    unittest.main(verbosity=2)
