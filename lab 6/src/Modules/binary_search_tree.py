from collections import deque


class TreeNode:
    """Класс узла BST"""

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return f"TreeNode({self.value})"


class BinarySearchTree:
    """Класс BST"""

    def __init__(self):
        self.root = None

    def is_empty(self):
        """Проверка дерева на наличие узлов"""
        return self.root is None
        # Временная сложность: O(1)

    def insert(self, value):
        """Метод для вставки узла"""
        if self.is_empty():
            self.root = TreeNode(value)
        else:
            self.insert_recursive(self.root, value)
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def insert_recursive(self, node, value):
        """Вспомогательный рекурсивный метод втсавки узла"""
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self.insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self.insert_recursive(node.right, value)
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def search(self, value):
        """Метод поиска узла в дереве"""
        if self.root is None:
            return "Элемент не найден, дерево пустое"
        else:
            return self.search_recursive(self.root, value)
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def search_recursive(self, node, value):
        """Вспомогательный рекурсивный метод для поиска узла"""
        if node.value == value:
            return node
        elif value < node.value:
            if node.left is None:
                return "Элемент не найден"
            else:
                return self.search_recursive(node.left, value)
        else:
            if node.right is None:
                return "Элемент не найден"
            else:
                return self.search_recursive(node.right, value)
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def delete(self, value):
        """Метод удаления узла из дерева"""
        if self.root is None:
            return "Дерево пустое, нельзя произвести удаление узла"
        else:
            self.root = self.delete_recursive(self.root, value)
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def delete_recursive(self, node, value):
        """Вспомогательный рекурсивный метод для удаления узла"""
        if node is None:
            return node

        if value < node.value:
            node.left = self.delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self.delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            successor = self.find_min(node.right)
            node.value = successor.value
            node.right = self.delete_recursive(node.right, successor.value)

        return node
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def find_min(self, node):
        """Метод для поиска минимального узла в поддереве"""
        current = node
        while current and current.left:
            current = current.left
        return current
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def find_max(self, node):
        """Метод для поиска максимального узла в поддереве"""
        current = node
        while current and current.right:
            current = current.right
        return current
        # Временная сложность: O(log n) в среднем, O(n) в худшем

    def is_valid(self):
        """Проверяет, является ли дерево валидным BST"""
        if self.root is None:
            return True  # Пустое дерево считается валидным BST

        return self._is_valid_recursive(self.root, float("-inf"), float("inf"))
        # Временная сложность: O(n) - нужно проверить все узлы

    def _is_valid_recursive(self, node, min_val, max_val):
        """
        Вспомогательный рекурсивный метод проверки валидности BST
        min_val и max_val задают допустимый диапазон значений для узла
        """
        if node is None:
            return True

        # Проверяем, что значение узла находится в допустимом диапазоне
        if not (min_val < node.value < max_val):
            return False

        # Рекурсивно проверяем левое и правое поддеревья
        return self._is_valid_recursive(
            node.left, min_val, node.value
        ) and self._is_valid_recursive(node.right, node.value, max_val)
        # Временная сложность: O(n)

    def height(self, node=None):
        """Получение высоты дерева/поддерева"""
        if node is None:
            node = self.root

        if node is None:
            return -1

        queue = deque([(node, 0)])
        max_height = 0

        while queue:
            current, level = queue.popleft()
            max_height = max(max_height, level)

            if current.left:
                queue.append((current.left, level + 1))
            if current.right:
                queue.append((current.right, level + 1))

        return max_height
