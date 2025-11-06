from performance_test_hash import run_all_performance_tests
from plot_hash_results import plot_all_results
from test_hash_tables import run_tests


def main():
    """
    Главная функция лабораторной работы
    Сложность: O(все компоненты)
    """
    print("ХЕШ-ФУНКЦИИ И ХЕШ-ТАБЛИЦЫ")  # O(1)
    print("=" * 60)  # O(1)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Запуск unit-тестов
    print("\n1. ЗАПУСК UNIT-ТЕСТОВ...")  # O(1)
    run_tests()  # O(все тесты)

    # Запуск тестов производительности
    print("\n2. ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ...")  # O(1)
    results = run_all_performance_tests()  # O(все тесты производительности)

    # Визуализация результатов
    print("\n3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")  # O(1)
    plot_all_results(results)  # O(все графики)


if __name__ == "__main__":
    main()  # O(все компоненты)
