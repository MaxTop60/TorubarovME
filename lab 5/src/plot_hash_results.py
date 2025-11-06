import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_hash_function_comparison(results: Dict):
    """
    График сравнения хеш-функций
    Сложность: O(функции * методы)
    """
    if "hash_functions" not in results:  # O(1)
        print("Нет данных для сравнения хеш-функций")  # O(1)
        return  # O(1)

    hash_functions = list(results["hash_functions"].keys())  # O(3)
    methods = ["chaining", "open_linear", "open_double"]  # O(3)

    # График времени вставки
    plt.figure(figsize=(12, 8))  # O(1)

    for i, method in enumerate(methods):  # O(3) итераций
        insert_times = []  # O(1)
        for hf in hash_functions:  # O(3) итераций
            if method in results["hash_functions"][hf]:  # O(1)
                insert_times.append(
                    results["hash_functions"][hf][method]["avg_insert_time"]
                )  # O(1)
            else:
                insert_times.append(0)  # O(1)

        plt.bar(
            np.arange(len(hash_functions)) + i * 0.25,
            insert_times,
            width=0.25,
            label=method,
        )  # O(1)

    plt.xlabel("Хеш-функции")  # O(1)
    plt.ylabel("Время вставки (секунды)")  # O(1)
    plt.title("Сравнение времени вставки для разных хеш-функций")  # O(1)
    plt.xticks(np.arange(len(hash_functions)) + 0.25, hash_functions)  # O(1)
    plt.legend()  # O(1)
    plt.grid(True, alpha=0.3)  # O(1)
    plt.tight_layout()  # O(1)
    plt.savefig("hash_functions_insert.png", dpi=300)  # O(1)
    plt.show()  # O(1)


def plot_load_factor_impact(results: Dict):
    """
    График влияния коэффициента заполнения
    Сложность: O(нагрузки * методы)
    """
    if "load_factors" not in results:  # O(1)
        print("Нет данных для анализа коэффициента заполнения")  # O(1)
        return  # O(1)

    load_factors = list(results["load_factors"].keys())  # O(4)
    methods = ["chaining", "open_linear", "open_double"]  # O(3)

    # График времени поиска в зависимости от коэффициента заполнения
    plt.figure(figsize=(12, 8))  # O(1)

    for method in methods:  # O(3) итераций
        search_times = []  # O(1)
        for lf in load_factors:  # O(4) итераций
            if method in results["load_factors"][lf]:  # O(1)
                search_times.append(
                    results["load_factors"][lf][method]["avg_search_time"]
                )  # O(1)
            else:
                search_times.append(0)  # O(1)

        plt.plot(
            load_factors, search_times, marker="o", label=method, linewidth=2
        )  # O(1)

    plt.xlabel("Коэффициент заполнения")  # O(1)
    plt.ylabel("Время поиска (секунды)")  # O(1)
    plt.title("Влияние коэффициента заполнения на время поиска")  # O(1)
    plt.legend()  # O(1)
    plt.grid(True, alpha=0.3)  # O(1)
    plt.tight_layout()  # O(1)
    plt.savefig("load_factor_impact.png", dpi=300)  # O(1)
    plt.show()  # O(1)


def plot_collision_stats(results: Dict):
    """
    График статистики коллизий
    Сложность: O(методы)
    """
    if "collisions" not in results:  # O(1)
        print("Нет данных для анализа коллизий")  # O(1)
        return  # O(1)

    collision_data = results["collisions"]  # O(1)
    methods = list(collision_data.keys())  # O(3)

    # Подготовка данных для столбчатой диаграммы
    collision_metrics = []  # O(1)
    method_names = []  # O(1)

    for method in methods:  # O(3) итераций
        stats = collision_data[method]["stats"]  # O(1)
        if "total_collisions" in stats:  # O(1)
            # Метод цепочек
            collision_metrics.append(stats["total_collisions"])  # O(1)
        elif "average_probes" in stats:  # O(1)
            # Открытая адресация
            collision_metrics.append(
                stats["average_probes"] * 10
            )  # Масштабируем для наглядности  # O(1)
        else:
            collision_metrics.append(0)  # O(1)
        method_names.append(method)  # O(1)

    plt.figure(figsize=(10, 6))  # O(1)
    bars = plt.bar(
        method_names, collision_metrics, color=["skyblue",
                                                "lightcoral", "lightgreen"]
    )  # O(1)

    # Добавляем значения на столбцы
    for bar, value in zip(bars, collision_metrics):  # O(3) итераций
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,  # O(1)
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )  # O(1)

    plt.xlabel("Метод разрешения коллизий")  # O(1)
    plt.ylabel("Коллизии / Пробирования (масштабировано)")  # O(1)
    plt.title("Сравнение количества коллизий для разных методов")  # O(1)
    plt.grid(True, alpha=0.3)  # O(1)
    plt.tight_layout()  # O(1)
    plt.savefig("collision_stats.png", dpi=300)  # O(1)
    plt.show()  # O(1)


def create_performance_summary(results: Dict):
    """
    Создание сводной таблицы производительности
    Сложность: O(результаты)
    """
    print("\n" + "=" * 80)  # O(1)
    print("СВОДНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ")  # O(1)
    print("=" * 80)  # O(1)

    # Анализ лучших методов для разных сценариев
    print("\nОПТИМАЛЬНЫЕ МЕТОДЫ:")  # O(1)

    # Лучшая хеш-функция
    if "hash_functions" in results and results["hash_functions"]:  # O(1)
        best_hash_func = min(  # O(3)
            results["hash_functions"].items(),
            key=lambda x: x[1].get("chaining", {}).get("avg_search_time",
                                                       float("inf")),
        )[0]
        print(f"Лучшая хеш-функция: {best_hash_func}")  # O(1)
    else:
        print("Лучшая хеш-функция: данные недоступны")  # O(1)

    # Лучший метод при низкой нагрузке
    if "load_factors" in results and 0.1 in results["load_factors"]:  # O(1)
        low_load_data = results["load_factors"][0.1]  # O(1)
        best_low_load = min(  # O(3)
            low_load_data.items(),
            key=lambda x: x[1].get("avg_search_time", float("inf")),
        )[0]
        print(f"Лучший метод при низкой нагрузке (α=0.1): {best_low_load}")
    else:
        print("Лучший метод при низкой нагрузке: данные недоступны")  # O(1)

    # Лучший метод при высокой нагрузке
    if "load_factors" in results and results["load_factors"]:  # O(1)
        # Берем самый высокий доступный коэффициент заполнения
        max_load_factor = max(results["load_factors"].keys())  # O(4)
        high_load_data = results["load_factors"][max_load_factor]  # O(1)
        best_high_load = min(  # O(3)
            high_load_data.items(),
            key=lambda x: x[1].get("avg_search_time", float("inf")),
        )[0]
        print(
            f"Лучший метод при высокой нагрузке (α={max_load_factor}): {best_high_load}"
        )  # O(1)
    else:
        print("Лучший метод при высокой нагрузке: данные недоступны")  # O(1)

    # Лучший метод при коллизиях
    if "collisions" in results and results["collisions"]:  # O(1)
        collision_data = results["collisions"]  # O(1)
        best_collision = min(  # O(3)
            collision_data.items(),
            key=lambda x: x[1].get("avg_search_time", float("inf")),
        )[0]
        print(f"Лучший метод при коллизиях: {best_collision}")  # O(1)
    else:
        print("Лучший метод при коллизиях: данные недоступны")  # O(1)

    # Общие рекомендации
    print("\nРЕКОМЕНДАЦИИ:")  # O(1)
    print(
        "1. Для общего использования: метод цепочек с полиномиальной хеш-функцией"
    )  # O(1)
    print(
        "2. Для высокой производительности: открытая адресация с двойным хешированием"
    )  # O(1)
    print(
        "3. Для избежания коллизий: поддерживайте коэффициент заполнения < 0.7"
    )  # O(1)
    print(
        "4. Для строковых ключей: используйте DJB2 или полиномиальную хеш-функцию"
    )  # O(1)


def plot_all_results(results: Dict):
    """
    Построение всех графиков
    Сложность: O(все графики)
    """
    print("\nСОЗДАНИЕ ГРАФИКОВ...")  # O(1)

    plot_hash_function_comparison(results)  # O(функции * методы)
    plot_load_factor_impact(results)  # O(нагрузки * методы)
    plot_collision_stats(results)  # O(методы)
    create_performance_summary(results)  # O(результаты)

    print("Графики сохранены в файлы .png")  # O(1)


if __name__ == "__main__":
    # Для тестирования
    from performance_test_hash import run_all_performance_tests

    results = run_all_performance_tests()
    plot_all_results(results)
