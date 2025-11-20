import unittest
from binary_search_tree import TreeNode, BinarySearchTree

from tree_traversal import (
    in_order_recursive,
    pre_order_recursive,
    post_order_recursive,
    in_order_iterative,
)


class TestBinarySearchTree(unittest.TestCase):

    def setUp(self):
        """Настройка тестового окружения перед каждым тестом"""
        self.bst = BinarySearchTree()

    def test_initial_state(self):
        """Тест начального состояния дерева"""
        self.assertTrue(self.bst.is_empty())
        self.assertIsNone(self.bst.root)

    def test_insert_into_empty_tree(self):
        """Тест вставки в пустое дерево"""
        self.bst.insert(10)
        self.assertFalse(self.bst.is_empty())
        self.assertEqual(self.bst.root.value, 10)
        self.assertIsNone(self.bst.root.left)
        self.assertIsNone(self.bst.root.right)

    def test_insert_multiple_values(self):
        """Тест вставки нескольких значений"""
        values = [8, 3, 10, 1, 6, 14, 4, 7, 13]
        for value in values:
            self.bst.insert(value)

        # Проверка структуры дерева
        self.assertEqual(self.bst.root.value, 8)
        self.assertEqual(self.bst.root.left.value, 3)
        self.assertEqual(self.bst.root.right.value, 10)
        self.assertEqual(self.bst.root.left.left.value, 1)
        self.assertEqual(self.bst.root.left.right.value, 6)

    def test_bst_property_after_insert(self):
        """Тест свойства BST после вставки"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        # Дерево должно оставаться валидным BST
        self.assertTrue(self.bst.is_valid())

    def test_search_existing_elements(self):
        """Тест поиска существующих элементов"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        for value in values:
            result = self.bst.search(value)
            self.assertIsInstance(result, TreeNode)
            self.assertEqual(result.value, value)

    def test_search_non_existing_elements(self):
        """Тест поиска несуществующих элементов"""
        values = [5, 3, 7]
        for value in values:
            self.bst.insert(value)

        result = self.bst.search(10)
        self.assertEqual(result, "Элемент не найден")

        result = self.bst.search(1)
        self.assertEqual(result, "Элемент не найден")

    def test_search_in_empty_tree(self):
        """Тест поиска в пустом дереве"""
        result = self.bst.search(5)
        self.assertEqual(result, "Элемент не найден, дерево пустое")

    def test_delete_leaf_node(self):
        """Тест удаления листового узла"""
        values = [5, 3, 7, 2, 4]
        for value in values:
            self.bst.insert(value)

        # Удаляем лист 2
        self.bst.delete(2)
        self.assertTrue(self.bst.is_valid())
        self.assertEqual(self.bst.search(2), "Элемент не найден")

        # Проверяем, что родительский узел обновился
        self.assertIsNone(self.bst.root.left.left)

    def test_delete_node_with_one_child(self):
        """Тест удаления узла с одним потомком"""
        values = [5, 3, 7, 2, 6]
        for value in values:
            self.bst.insert(value)

        # Удаляем узел 7 (имеет левого потомка 6)
        self.bst.delete(7)
        self.assertTrue(self.bst.is_valid())
        self.assertEqual(self.bst.search(7), "Элемент не найден")

        # Проверяем, что правый потомок корня теперь 6
        self.assertEqual(self.bst.root.right.value, 6)

    def test_delete_node_with_two_children(self):
        """Тест удаления узла с двумя потомками"""
        values = [5, 3, 8, 2, 4, 7, 9, 6]
        for value in values:
            self.bst.insert(value)

        # Удаляем узел 8 (имеет двух потомков)
        self.bst.delete(8)
        self.assertTrue(self.bst.is_valid())
        self.assertEqual(self.bst.search(8), "Элемент не найден")

        # Преемником должен быть 9 (наименьший в правом поддереве)
        self.assertEqual(self.bst.root.right.value, 9)

    def test_delete_root_node(self):
        """Тест удаления корневого узла"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        # Удаляем корень
        self.bst.delete(5)
        self.assertTrue(self.bst.is_valid())
        self.assertEqual(self.bst.search(5), "Элемент не найден")

        # Новый корень должен быть преемником (6)
        self.assertEqual(self.bst.root.value, 6)

    def test_delete_from_empty_tree(self):
        """Тест удаления из пустого дерева"""
        result = self.bst.delete(5)
        self.assertEqual(result, "Дерево пустое, нельзя произвести удаление узла")

    def test_delete_non_existing_element(self):
        """Тест удаления несуществующего элемента"""
        values = [5, 3, 7]
        for value in values:
            self.bst.insert(value)

        # Удаление не должно сломать дерево
        self.bst.delete(10)
        self.assertTrue(self.bst.is_valid())

        # Все существующие элементы должны остаться
        for value in values:
            result = self.bst.search(value)
            self.assertEqual(result.value, value)

    def test_find_min_max(self):
        """Тест поиска минимального и максимального элементов"""
        values = [8, 3, 10, 1, 6, 14, 4, 7, 13]
        for value in values:
            self.bst.insert(value)

        min_node = self.bst.find_min(self.bst.root)
        max_node = self.bst.find_max(self.bst.root)

        self.assertEqual(min_node.value, 1)
        self.assertEqual(max_node.value, 14)

    def test_tree_height(self):
        """Тест вычисления высоты дерева"""
        # Пустое дерево
        self.assertEqual(self.bst.height(), -1)

        # Дерево с одним узлом
        self.bst.insert(5)
        self.assertEqual(self.bst.height(), 0)

        # Сбалансированное дерево
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)
        self.assertEqual(self.bst.height(), 2)

    def test_in_order_traversal(self):
        """Тест in-order обхода"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        # In-order должен вернуть отсортированную последовательность
        expected = [2, 3, 4, 5, 6, 7, 8]

        # Проверяем рекурсивную версию
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        in_order_recursive(self.bst.root)
        sys.stdout = sys.__stdout__

        result = list(map(int, captured_output.getvalue().strip().split()))
        self.assertEqual(result, expected)

        # Проверяем итеративную версию
        captured_output = io.StringIO()
        sys.stdout = captured_output
        in_order_iterative(self.bst.root)
        sys.stdout = sys.__stdout__

        result = list(map(int, captured_output.getvalue().strip().split()))
        self.assertEqual(result, expected)

    def test_pre_order_traversal(self):
        """Тест pre-order обхода"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        expected = [5, 3, 2, 4, 7, 6, 8]

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        pre_order_recursive(self.bst.root)
        sys.stdout = sys.__stdout__

        result = list(map(int, captured_output.getvalue().strip().split()))
        self.assertEqual(result, expected)

    def test_post_order_traversal(self):
        """Тест post-order обхода"""
        values = [5, 3, 7, 2, 4, 6, 8]
        for value in values:
            self.bst.insert(value)

        expected = [2, 4, 3, 6, 8, 7, 5]

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        post_order_recursive(self.bst.root)
        sys.stdout = sys.__stdout__

        result = list(map(int, captured_output.getvalue().strip().split()))
        self.assertEqual(result, expected)

    def test_bst_property_after_complex_operations(self):
        """Тест свойства BST после сложных операций"""
        # Вставляем значения в разном порядке
        insert_sequence = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
        delete_sequence = [20, 70, 30]

        for value in insert_sequence:
            self.bst.insert(value)
            self.assertTrue(
                self.bst.is_valid(), f"Tree invalid after inserting {value}"
            )

        for value in delete_sequence:
            self.bst.delete(value)
            self.assertTrue(self.bst.is_valid(), f"Tree invalid after deleting {value}")

    def test_duplicate_insertion(self):
        """Тест вставки дубликатов"""
        self.bst.insert(5)
        self.bst.insert(5)  # Дубликат

        # Дерево должно остаться валидным
        self.assertTrue(self.bst.is_valid())

        # В дереве должен быть только один узел со значением 5
        count = 0

        def count_nodes(node, target):
            nonlocal count
            if node:
                if node.value == target:
                    count += 1
                count_nodes(node.left, target)
                count_nodes(node.right, target)

        count_nodes(self.bst.root, 5)
        self.assertEqual(count, 1)

    def test_invalid_bst_detection(self):
        """Тест обнаружения невалидного BST"""
        # Создаем невалидное BST вручную
        root = TreeNode(5)
        root.left = TreeNode(6)  # Нарушениеs: 6 > 5 в левом поддереве
        root.right = TreeNode(7)

        invalid_bst = BinarySearchTree()
        invalid_bst.root = root

        self.assertFalse(invalid_bst.is_valid())


if __name__ == "__main__":
    # Запуск тестов с детальным выводом
    unittest.main(verbosity=2)
