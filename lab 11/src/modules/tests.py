"""
Unit-тесты для всех алгоритмов строк.
Тестирование на строках разных типов:
- случайные строки
- периодические паттерны
- строки с повторениями
- граничные случаи
"""

import unittest
import random
import string
import timeit
from typing import List

# Импортируем все наши алгоритмы
from prefix_function import prefix_function, prefix_function_naive
from kmp_search import kmp_search, naive_string_search as kmp_naive
from z_function import z_function, z_function_naive, search_with_z_function
from string_matching import boyer_moore_search, boyer_moore_search_with_good_suffix


class TestPrefixFunction(unittest.TestCase):
    """Тесты для префикс-функции."""

    def test_prefix_function_basic(self):
        """Базовые тесты префикс-функции."""
        test_cases = [
            ("abcabcd", [0, 0, 0, 1, 2, 3, 0]),
            ("aabaaab", [0, 1, 0, 1, 2, 2, 3]),
            ("abababcaab", [0, 0, 1, 2, 3, 4, 0, 1, 1, 2]),
            ("", []),
            ("a", [0]),
            ("aa", [0, 1]),
            ("ab", [0, 0]),
        ]

        for s, expected in test_cases:
            with self.subTest(s=s):
                result = prefix_function(s)
                self.assertEqual(result, expected)

    def test_prefix_function_naive_vs_optimized(self):
        """Сравнение наивной и оптимизированной версий."""
        test_strings = [
            "abacaba",
            "abcabcabc",
            "aaaaa",
            "ababab",
            "abcdefgh" * 3,
        ]

        for s in test_strings:
            with self.subTest(s=s):
                naive = prefix_function_naive(s)
                optimized = prefix_function(s)
                self.assertEqual(naive, optimized)

    def test_prefix_function_performance(self):
        """Тест производительности префикс-функции."""
        # Большая строка
        s = "a" * 1000 + "b" + "a" * 1000

        # Измеряем время для оптимизированной версии
        timer = timeit.Timer(lambda: prefix_function(s))
        time_optimized = timer.timeit(number=100) / 100

        # Для маленькой строки измеряем наивную тоже
        s_small = "abacaba" * 10
        timer_naive = timeit.Timer(lambda: prefix_function_naive(s_small))
        time_naive = timer_naive.timeit(number=10) / 10

        timer_optimized_small = timeit.Timer(lambda: prefix_function(s_small))
        time_optimized_small = timer_optimized_small.timeit(number=1000) / 1000

        print(f"\nПроизводительность префикс-функции:")
        print(f"  Наивная (маленькая строка): {time_naive:.6f} сек")
        print(f"  Оптимизированная (маленькая): {time_optimized_small:.6f} сек")
        print(f"  Оптимизированная (большая): {time_optimized:.6f} сек")

        # Проверяем, что оптимизированная работает
        self.assertLess(time_optimized, 0.01)


class TestKMP(unittest.TestCase):
    """Тесты для алгоритма Кнута-Морриса-Пратта."""

    def test_kmp_search_basic(self):
        """Базовые тесты KMP."""
        test_cases = [
            ("abacabadabacaba", "aba", [0, 4, 8, 12]),
            ("hello world", "world", [6]),
            ("aaa", "aa", [0, 1]),
            ("", "abc", []),
            ("abc", "", [0, 1, 2, 3]),
            ("abababab", "abab", [0, 2, 4]),
            ("mississippi", "issi", [1, 4]),
        ]

        for text, pattern, expected in test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = kmp_search(text, pattern)
                self.assertEqual(result, expected)

    def test_kmp_vs_naive_consistency(self):
        """Согласованность KMP с наивным алгоритмом."""
        random_gen = random.Random(42)

        for _ in range(50):
            text_len = random_gen.randint(5, 30)
            pattern_len = random_gen.randint(1, 10)

            text = "".join(random_gen.choice("abc") for _ in range(text_len))
            pattern = "".join(random_gen.choice("abc") for _ in range(pattern_len))

            if pattern_len > text_len:
                continue

            kmp_result = kmp_search(text, pattern)
            naive_result = kmp_naive(text, pattern)

            self.assertEqual(
                kmp_result,
                naive_result,
                f"Расхождение для text='{text}', pattern='{pattern}'",
            )

    def test_kmp_performance(self):
        """Тест производительности KMP."""
        # Периодическая строка - KMP должен хорошо работать
        text = "abc" * 1000
        pattern = "abcabcabc"

        # Проверяем корректность
        result = kmp_search(text, pattern)
        self.assertTrue(len(result) > 0)

        # Измеряем время
        timer = timeit.Timer(lambda: kmp_search(text, pattern))
        time_taken = timer.timeit(number=100) / 100

        print(f"\nПроизводительность KMP:")
        print(f"  Время поиска: {time_taken:.6f} сек")
        print(f"  Найдено вхождений: {len(result)}")

        self.assertLess(time_taken, 0.01)


class TestZFunction(unittest.TestCase):
    """Тесты для Z-функции."""

    def test_z_function_basic(self):
        """Базовые тесты Z-функции."""
        test_cases = [
            ("aaaaa", [0, 4, 3, 2, 1]),  # z[0] = 0
            ("abacaba", [0, 0, 1, 0, 3, 0, 1]),  # z[0] = 0
            ("abcabc", [0, 0, 0, 3, 0, 0]),  # z[0] = 0
            ("", []),
            ("a", [0]),  # z[0] = 0
            ("ab", [0, 0]),  # z[0] = 0
        ]

        for s, expected in test_cases:
            with self.subTest(s=s):
                result = z_function(s)
                self.assertEqual(result, expected)

    def test_z_function_extended(self):
        """Расширенные тесты Z-функции."""
        test_cases = [
            ("abcde", [0, 0, 0, 0, 0]),
            ("aabaab", [0, 1, 0, 3, 1, 0]),  # Исправлено значение
            ("aaaab", [0, 3, 2, 1, 0]),
            ("ababab", [0, 0, 4, 0, 2, 0]),
        ]

        for s, expected in test_cases:
            with self.subTest(s=s):
                result = z_function(s)
                self.assertEqual(result, expected)


class TestBoyerMoore(unittest.TestCase):
    """Тесты для алгоритма Бойера-Мура."""

    def test_boyer_moore_basic(self):
        """Базовые тесты Бойера-Мура."""
        test_cases = [
            ("abacabadabacaba", "aba", [0, 4, 8, 12]),
            ("hello world", "world", [6]),
            ("aaa", "aa", [0, 1]),
            ("", "abc", []),
            ("abc", "", [0, 1, 2, 3]),
            ("mississippi", "issi", [1, 4]),
            ("abababab", "abab", [0, 2, 4]),
        ]

        for text, pattern, expected in test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = boyer_moore_search(text, pattern)
                self.assertEqual(result, expected)

    def test_boyer_moore_special_cases(self):
        """Специальные случаи для Бойера-Мура."""
        test_cases = [
            # Повторяющиеся символы
            ("aaaaa", "aaa", [0, 1, 2]),
            # Паттерн в конце
            ("abcdefg", "efg", [4]),
            # Паттерн в начале
            ("abcdefg", "abc", [0]),  # Эта проверка должна теперь работать
            # Несколько вхождений
            ("ababababab", "aba", [0, 2, 4, 6]),
            # Символы, которых нет в паттерне
            ("xyz", "abc", []),
        ]

        for text, pattern, expected in test_cases:
            with self.subTest(text=text, pattern=pattern):
                result = boyer_moore_search(text, pattern)
                self.assertEqual(result, expected)


class TestAllAlgorithmsConsistency(unittest.TestCase):
    """Тесты согласованности всех алгоритмов."""

    def setUp(self):
        self.random = random.Random(42)

    def generate_test_string(
        self, length: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    ) -> str:
        return "".join(self.random.choice(alphabet) for _ in range(length))

    @staticmethod
    def simple_naive_search(text: str, pattern: str) -> List[int]:
        """Простой наивный алгоритм поиска для внутреннего использования в тестах."""
        n, m = len(text), len(pattern)
        if m == 0:
            return list(range(n + 1))
        result = []
        for i in range(n - m + 1):
            if text[i : i + m] == pattern:
                result.append(i)
        return result

    def test_consistency_simple_cases(self):
        """Согласованность на простых случаях."""
        test_cases = [
            ("hello world", "world"),
            ("abacabadabacaba", "aba"),
            ("mississippi", "issi"),
            ("aaa", "aa"),
            ("abababab", "abab"),
        ]

        for text, pattern in test_cases:
            # Запускаем все алгоритмы
            kmp_result = kmp_search(text, pattern)
            z_result = search_with_z_function(text, pattern)
            bm_result = boyer_moore_search(text, pattern)
            bm_suffix_result = boyer_moore_search_with_good_suffix(text, pattern)
            naive_result = self.simple_naive_search(text, pattern)  # Используем self.

            # Все должны давать одинаковый результат
            algorithms = [
                ("KMP", kmp_result),
                ("Z-function", z_result),
                ("Boyer-Moore", bm_result),
                ("Boyer-Moore с суффиксом", bm_suffix_result),
                ("Naive", naive_result),
            ]

            # Используем наивный алгоритм как эталон
            reference = naive_result

            for algo_name, result in algorithms:
                with self.subTest(text=text, pattern=pattern, algorithm=algo_name):
                    self.assertEqual(
                        result, reference, f"Алгоритм {algo_name} дал другой результат"
                    )

    def test_consistency_random(self):
        """Случайное тестирование согласованности."""
        for _ in range(50):
            text_len = self.random.randint(20, 100)
            pattern_len = self.random.randint(1, 15)

            text = self.generate_test_string(text_len, "abc")
            pattern = self.generate_test_string(pattern_len, "abc")

            if pattern_len > text_len:
                continue

            # Используем простой наивный поиск как эталон
            naive_result = self.simple_naive_search(text, pattern)  # Используем self.
            kmp_result = kmp_search(text, pattern)
            z_result = search_with_z_function(text, pattern)
            bm_result = boyer_moore_search(text, pattern)

            self.assertEqual(
                kmp_result,
                naive_result,
                f"KMP vs Naive для text='{text}', pattern='{pattern}'",
            )
            self.assertEqual(
                z_result,
                naive_result,
                f"Z-search vs Naive для text='{text}', pattern='{pattern}'",
            )
            self.assertEqual(
                bm_result,
                naive_result,
                f"Boyer-Moore vs Naive для text='{text}', pattern='{pattern}'",
            )


def run_all_tests():
    """Запуск всех тестов."""
    # Создаем test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Добавляем все тестовые классы
    suite.addTests(loader.loadTestsFromTestCase(TestPrefixFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestKMP))
    suite.addTests(loader.loadTestsFromTestCase(TestZFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestBoyerMoore))
    suite.addTests(loader.loadTestsFromTestCase(TestAllAlgorithmsConsistency))

    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Выводим статистику
    print(f"\n" + "=" * 60)
    print(f"ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"  Всего тестов: {result.testsRun}")
    print(f"  Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Провалено: {len(result.failures)}")
    print(f"  Ошибок: {len(result.errors)}")

    if result.failures:
        print(f"\nПроваленные тесты:")
        for test, traceback in result.failures:
            print(f"  - {test.id()}")

    if result.errors:
        print(f"\nТесты с ошибками:")
        for test, traceback in result.errors:
            print(f"  - {test.id()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Запуск тестов для всех алгоритмов строк...")
    print("=" * 60)

    success = run_all_tests()

    # Выход с кодом ошибки
    exit(0 if success else 1)
