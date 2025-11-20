from Modules.analisys import (
        demo_visualization,
        run_experiment,
        plot_results
    )
import sys

sys.setrecursionlimit(10000)

if __name__ == "__main__":
    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Демонстрация визуализации
    demo_visualization()

    # Основной эксперимент
    sizes, balanced_times, degenerate_times = run_experiment()

    # Построение графиков
    plot_results(sizes, balanced_times, degenerate_times)

    print("\n=== ЭКСПЕРИМЕНТ ЗАВЕРШЕН ===")
