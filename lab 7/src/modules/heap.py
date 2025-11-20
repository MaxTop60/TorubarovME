class MinHeap:
    """Класс кучи"""

    def __init__(self):
        self.tree = []

    def _get_parent(self, index):
        """Получение индекса родительского узла"""
        return (index - 1) // 2
        # Временная сложность: O(1)

    def _get_left_child(self, index):
        """Получение индекса левого потомка"""
        return 2 * index + 1
        # Временная сложность: O(1)

    def _get_right_child(self, index):
        """Получение индекса правого потомка"""
        return 2 * index + 2
        # Временная сложность: O(1)

    def _get_min_index(self, index1, index2):
        """Получение индекса минимального из двух элементов"""
        if self.tree[index1] < self.tree[index2]:
            return index1
        else:
            return index2
        # Временная сложность: O(1)

    def _sift_up(self, index):
        """Всплытие элемента"""
        if index == 0:
            return

        if self.tree[index] < self.tree[self._get_parent(index)]:
            support = self.tree[index]
            self.tree[index] = self.tree[self._get_parent(index)]
            self.tree[self._get_parent(index)] = support
            self._sift_up(self._get_parent(index))
        # Временная сложность: O(log n) - высота дерева

    def _sift_down(self, index):
        """Погружение элемента"""
        # Случай когда элемент последний в массиве
        if index == len(self.tree) - 1:
            return

        # Случай когда элемент имеет два потомка
        if self._get_left_child(
            index
        ) < len(self.tree) and self._get_right_child(
            index
        ) < len(self.tree):
            min_child_index = self._get_min_index(
                self._get_left_child(index), self._get_right_child(index)
            )
            if self.tree[index] > self.tree[min_child_index]:
                support = self.tree[index]
                self.tree[index] = self.tree[min_child_index]
                self.tree[min_child_index] = support
                self._sift_down(min_child_index)
        else:
            # Случай когда элемент имеет только левого потомка
            if self._get_left_child(index) < len(self.tree):
                if self.tree[index] > self.tree[self._get_left_child(index)]:
                    support = self.tree[index]
                    self.tree[index] = self.tree[self._get_left_child(index)]
                    self.tree[self._get_left_child(index)] = support
            # Случай когда элемент имеет только правого потомка
            elif self._get_right_child(index) < len(self.tree):
                if self.tree[index] > self.tree[self._get_right_child(index)]:
                    support = self.tree[index]
                    self.tree[index] = self.tree[self._get_right_child(index)]
                    self.tree[self._get_right_child(index)] = support
        # Временная сложность: O(log n) - высота дерева

    def insert(self, value):
        """Вставка элемента в кучу"""
        self.tree.append(value)
        self._sift_up(len(self.tree) - 1)
        # Временная сложность: O(log n)

    def extract(self):
        """Извлечение корня"""
        if len(self.tree) > 0:
            root = self.tree.pop(0)
            if len(self.tree) > 0:
                last_element = self.tree.pop()
                self.tree.insert(0, last_element)
                self._sift_down(0)
            return root
        else:
            return "Куча пуста"
        # Временная сложность: O(log n)

    def peek(self):
        """Просмотр корня"""
        if len(self.tree) > 0:
            return self.tree[0]
        else:
            return "Куча пуста"
        # Временная сложность: O(1)

    def build_heap(self, array):
        """Создание кучи из произвольного массива"""
        # Начинаем с последнего нелистового узла
        n = len(array)
        self.tree = array
        for i in range(n // 2 - 1, -1, -1):
            self._sift_down(i)
        # Временная сложность: O(n)
