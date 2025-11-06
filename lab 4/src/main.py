from performance_test import (
    run_all_tests,
    print_results,
    analyze_theoretical_vs_practical,
)
from plot_results import (plot_time_vs_size, plot_time_vs_datatype,
                          create_summary_table)


def main():
    print("АНАЛИЗ АЛГОРИТМОВ СОРТИРОВКИ")
    print("=" * 50)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Запускаем все тесты
    results = run_all_tests()

    # Выводим результаты в таблицу
    print_results(results)

    # Строим графики и анализируем
    print("\nСоздание графиков...")
    plot_time_vs_size(results, "random")
    plot_time_vs_datatype(results, 1000)

    # Сводная таблица и анализ
    create_summary_table(results)
    analyze_theoretical_vs_practical(results)


if __name__ == "__main__":
    main()
