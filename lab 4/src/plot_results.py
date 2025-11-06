import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_time_vs_size(results: Dict, data_type: str = "random"):
    """
    График зависимости времени от размера массива для одного типа данных
    """
    plt.figure(figsize=(12, 8))

    # Получаем размеры из первого алгоритма
    first_algo = next(iter(results.values()))
    sizes = sorted(first_algo.keys())

    for algo_name, algo_data in results.items():
        times = []
        for size in sizes:
            # Обращаемся к правильной структуре данных
            times.append(algo_data[size][data_type])
        plt.plot(sizes, times, marker="o", label=algo_name, linewidth=2)

    plt.xlabel("Размер массива")
    plt.ylabel("Время (секунды)")
    plt.title(
        f"""Зависимость времени сортировки
              от размера массива ({data_type} данные)"""
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"time_vs_size_{data_type}.png", dpi=300)
    plt.show()


def plot_time_vs_datatype(results: Dict, size: int = 1000):
    """
    График зависимости времени от типа данных для фиксированного размера
    """
    plt.figure(figsize=(12, 8))

    data_types = ["random", "sorted", "reversed", "almost"]
    x_pos = np.arange(len(data_types))
    width = 0.15

    for i, (algo_name, algo_data) in enumerate(results.items()):
        times = []
        for data_type in data_types:
            if size in algo_data:
                times.append(algo_data[size][data_type])
            else:
                times.append(0)

        plt.bar(x_pos + i * width, times, width, label=algo_name)

    plt.xlabel("Тип данных")
    plt.ylabel("Время (секунды)")
    plt.title(f"Время сортировки для разных типов данных (размер {size})")
    plt.xticks(x_pos + width * 2, data_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"time_vs_datatype_size{size}.png", dpi=300)
    plt.show()


def create_summary_table(results: Dict):
    """
    Создание сводной таблицы результатов
    """
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 90)

    data_types = ["random", "sorted", "reversed", "almost"]
    sizes = [100, 500, 1000, 2000]

    best_per_type = {}

    for data_type in data_types:
        best_per_type[data_type] = {}
        for size in sizes:
            best_time = float("inf")
            best_algo = "N/A"

            for algo_name, algo_data in results.items():
                if size in algo_data and data_type in algo_data[size]:

                    time_val = algo_data[size][data_type]
                    if time_val < best_time and time_val != float("inf"):
                        best_time = time_val
                        best_algo = algo_name

            if best_time != float("inf"):
                best_per_type[data_type][size] = (best_algo, best_time)
            else:
                best_per_type[data_type][size] = ("N/A", float("inf"))

    print(
        f"{'Тип данных':<14} | {'100':<20} | {'500':<20} | {'1000':<20} | {'2000':<20}"
    )
    print("-" * 90)

    for data_type in data_types:
        row_parts = [f"{data_type:<14}"]
        for size in sizes:
            if data_type in best_per_type and size in best_per_type[data_type]:

                best_algo, best_time = best_per_type[data_type][size]

                if best_time != float("inf"):
                    algo_short = best_algo[
                        :6] if len(best_algo) > 6 else best_algo
                    cell = f"{algo_short}:{best_time:.6f}"
                    row_parts.append(f" {cell:<19}")
                else:
                    row_parts.append(f" {'N/A':<19}")
            else:
                row_parts.append(f" {'N/A':<19}")

        print(" |".join(row_parts))

    # Анализ результатов
    print("\n" + "=" * 50)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 50)

    # Подсчитываем сколько раз каждый алгоритм был лучшим
    algo_wins = {}
    for data_type in data_types:
        for size in sizes:
            if (
                data_type in best_per_type
                and size in best_per_type[data_type]
                and best_per_type[data_type][size][0] != "N/A"
            ):

                best_algo = best_per_type[data_type][size][0]
                algo_wins[best_algo] = algo_wins.get(best_algo, 0) + 1

    print("Количество побед по алгоритмам:")
    for algo, wins in sorted(algo_wins.items(),
                             key=lambda x: x[1], reverse=True):
        print(f"  {algo}: {wins} раз")
