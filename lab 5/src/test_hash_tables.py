import unittest
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing
from hash_functions import simple_hash, polynomial_hash, djb2_hash


class TestHashFunctions(unittest.TestCase):
    """
    Тестирование хеш-функций
    """

    def test_simple_hash(self):
        """Тест простой хеш-функции - O(n)"""
        result1 = simple_hash("test", 100)  # O(4)
        result2 = simple_hash("test", 100)  # O(4)
        self.assertEqual(result1, result2)  # O(1)

        # Проверка детерминированности
        result3 = simple_hash("hello", 50)  # O(5)
        result4 = simple_hash("hello", 50)  # O(5)
        self.assertEqual(result3, result4)  # O(1)

    def test_polynomial_hash(self):
        """Тест полиномиальной хеш-функции - O(n)"""
        result1 = polynomial_hash("abc", 100)  # O(3)
        result2 = polynomial_hash("abc", 100)  # O(3)
        self.assertEqual(result1, result2)  # O(1)

        # Проверка что порядок имеет значение
        result3 = polynomial_hash("abc", 100)  # O(3)
        result4 = polynomial_hash("cba", 100)  # O(3)
        self.assertNotEqual(result3, result4)  # O(1)

    def test_djb2_hash(self):
        """Тест DJB2 хеш-функции - O(n)"""
        result1 = djb2_hash("test", 100)  # O(4)
        result2 = djb2_hash("test", 100)  # O(4)
        self.assertEqual(result1, result2)  # O(1)


class TestHashTableChaining(unittest.TestCase):
    """
    Тестирование хеш-таблицы с методом цепочек
    """

    def setUp(self):
        """Настройка тестов - O(1)"""
        self.ht = HashTableChaining(size=10)  # O(10)

    def test_insert_search(self):
        """Тест вставки и поиска - O(1) в среднем"""
        self.ht.insert("key1", "value1")  # O(1)
        self.ht.insert("key2", "value2")  # O(1)

        self.assertEqual(self.ht.search("key1"), "value1")  # O(1)
        self.assertEqual(self.ht.search("key2"), "value2")  # O(1)
        self.assertIsNone(self.ht.search("key3"))  # O(1)

    def test_update(self):
        """Тест обновления значения - O(1) в среднем"""
        self.ht.insert("key1", "value1")  # O(1)
        self.ht.insert("key1", "new_value")  # O(1)

        self.assertEqual(self.ht.search("key1"), "new_value")  # O(1)

    def test_delete(self):
        """Тест удаления - O(1) в среднем"""
        self.ht.insert("key1", "value1")  # O(1)
        self.assertTrue(self.ht.delete("key1"))  # O(1)
        self.assertIsNone(self.ht.search("key1"))  # O(1)
        self.assertFalse(self.ht.delete("key1"))  # O(1)

    def test_collisions(self):
        """Тест обработки коллизий - O(n) в худшем"""
        # Создаем коллизии
        self.ht.insert("a", 1)  # O(1)
        self.ht.insert("k", 2)  # O(1) - может быть коллизия

        self.assertEqual(self.ht.search("a"), 1)  # O(1)
        self.assertEqual(self.ht.search("k"), 2)  # O(1)


class TestHashTableOpenAddressing(unittest.TestCase):
    """
    Тестирование хеш-таблицы с открытой адресацией
    """

    def test_linear_probing(self):
        """Тест линейного пробирования - O(1/(1-α)) в среднем"""
        ht = HashTableOpenAddressing(size=10, probing_method='linear')  # O(10)
        ht.insert("key1", "value1")  # O(1)
        ht.insert("key2", "value2")  # O(1)

        self.assertEqual(ht.search("key1"), "value1")  # O(1)
        self.assertEqual(ht.search("key2"), "value2")  # O(1)

    def test_double_hashing(self):
        """Тест двойного хеширования - O(1/(1-α)) в среднем"""
        ht = HashTableOpenAddressing(size=10, probing_method='double')  # O(10)
        ht.insert("key1", "value1")  # O(1)
        ht.insert("key2", "value2")  # O(1)

        self.assertEqual(ht.search("key1"), "value1")  # O(1)
        self.assertEqual(ht.search("key2"), "value2")  # O(1)

    def test_deletion(self):
        """Тест удаления с маркером - O(1/(1-α)) в среднем"""
        ht = HashTableOpenAddressing(size=10)  # O(10)
        ht.insert("key1", "value1")  # O(1)
        ht.insert("key2", "value2")  # O(1)

        self.assertTrue(ht.delete("key1"))  # O(1)
        self.assertIsNone(ht.search("key1"))  # O(1)
        self.assertEqual(ht.search("key2"), "value2")  # O(1)


def run_tests():
    """Запуск всех тестов - O(все тесты)"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()  # O(все тесты)
