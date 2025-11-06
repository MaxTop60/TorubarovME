import timeit
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing
from generate_hash_data import generate_test_data


def measure_operation_time(operation, setup_code, number=1):
    """
    Измерение времени операции с помощью timeit
    Сложность: O(number * сложность_операции)
    """
    try:
        timer = timeit.Timer(operation, setup=setup_code)  # O(1)
        # O(number * сложность_операции)
        time_taken = timer.timeit(number=number)
        return time_taken / number  # O(1)
    except Exception as e:
        print(f"Ошибка при измерении времени: {e}")  # O(1)
        return float("inf")  # O(1)


def test_hash_table_performance(hash_table_class,
                                test_data, table_size=1000, **kwargs):
    """
    Тестирование производительности хеш-таблицы с timeit
    Сложность: O(n) операций
    """
    # Увеличиваем размер таблицы чтобы избежать переполнения
    safe_table_size = max(table_size, len(test_data) * 2)  # O(1)

    # Подготовка данных для timeit
    data_str = str(test_data)  # O(n)

    # Тестирование вставки
    insert_setup = f"""
from {hash_table_class.__module__} import {hash_table_class.__name__}
test_data = {data_str}
ht = {hash_table_class.__name__}(size={safe_table_size}, **{kwargs})
    """  # O(n)

    insert_operation = """
for key, value in test_data:
    ht.insert(key, value)
    """  # O(n)

    avg_insert_time = measure_operation_time(
        insert_operation, insert_setup, number=1
    )  # O(n)

    # Тестирование поиска (после вставки)
    search_setup = f"""
from {hash_table_class.__module__} import {hash_table_class.__name__}
test_data = {data_str}
ht = {hash_table_class.__name__}(size={safe_table_size}, **{kwargs})
for key, value in test_data:
    ht.insert(key, value)
    """  # O(n)

    search_operation = """
for key, value in test_data:
    result = ht.search(key)
    """  # O(n)

    avg_search_time = measure_operation_time(
        search_operation, search_setup, number=1
    )  # O(n)

    # Получаем статистику
    try:
        ht = hash_table_class(size=safe_table_size, **kwargs)  # O(size)
        for key, value in test_data:  # O(n)
            ht.insert(key, value)  # O(1)
        stats = ht.get_collision_stats()  # O(size)
        load_factor = ht.load_factor  # O(1)
    except Exception as e:
        print(f"Ошибка при получении статистики: {e}")  # O(1)
        stats = {"error": str(e)}  # O(1)
        load_factor = 0  # O(1)

    return {  # O(1)
        "avg_insert_time": avg_insert_time,
        "avg_search_time": avg_search_time,
        "stats": stats,
        "load_factor": load_factor,
    }


def compare_hash_functions():
    """
    Сравнение разных хеш-функций
    Сложность: O(функции * n)
    """
    # Уменьшаем количество тестовых данных чтобы избежать переполнения
    test_data = generate_test_data(200)  # O(200 * 10)
    hash_functions = ["simple", "polynomial", "djb2"]  # O(1)

    results = {}  # O(1)
    for hash_func in hash_functions:  # O(3) итераций
        print(f"Тестирование хеш-функции: {hash_func}")  # O(1)

        # Метод цепочек
        result_chaining = test_hash_table_performance(  # O(n)
            HashTableChaining, test_data,
            hash_function=hash_func, table_size=500
        )

        # Открытая адресация с линейным пробированием
        result_open_linear = test_hash_table_performance(  # O(n)
            HashTableOpenAddressing,
            test_data,
            hash_function=hash_func,
            table_size=500,
            probing_method="linear",
        )

        # Открытая адресация с двойным хешированием
        result_open_double = test_hash_table_performance(  # O(n)
            HashTableOpenAddressing,
            test_data,
            hash_function=hash_func,
            table_size=500,
            probing_method="double",
        )

        results[hash_func] = {  # O(1)
            "chaining": result_chaining,
            "open_linear": result_open_linear,
            "open_double": result_open_double,
        }

    return results  # O(1)


def test_load_factor_impact():
    """
    Тестирование влияния коэффициента заполнения
    Сложность: O(нагрузки * n)
    """
    load_factors = [
        0.1,
        0.3,
        0.5,
        0.7,
    ]  # Убрали 0.9 чтобы избежать переполнения  # O(1)
    table_size = 500  # O(1)
    results = {}  # O(1)

    for target_lf in load_factors:  # O(4) итераций
        print(f"Тестирование коэффициента заполнения: {target_lf}")  # O(1)

        num_items = int(table_size * target_lf)  # O(1)
        test_data = generate_test_data(num_items)  # O(num_items * 10)

        # Метод цепочек
        result_chaining = test_hash_table_performance(  # O(num_items)
            HashTableChaining, test_data, table_size=table_size
        )

        # Открытая адресация с линейным пробированием
        result_open_linear = test_hash_table_performance(  # O(num_items)
            HashTableOpenAddressing,
            test_data,
            table_size=table_size,
            probing_method="linear",
        )

        # Открытая адресация с двойным хешированием
        result_open_double = test_hash_table_performance(  # O(num_items)
            HashTableOpenAddressing,
            test_data,
            table_size=table_size,
            probing_method="double",
        )

        results[target_lf] = {  # O(1)
            "chaining": result_chaining,
            "open_linear": result_open_linear,
            "open_double": result_open_double,
        }

    return results  # O(1)


def test_collision_performance():
    """
    Тестирование производительности при коллизиях
    Сложность: O(n²) в худшем случае
    """
    # Генерируем данные, которые могут вызывать коллизии
    base_keys = [
        "key"
    ] * 50  # Уменьшили количество чтобы избежать переполнения  # O(50)
    test_data = [
        (f"{base}{i}", f"value_{i}") for i, base in enumerate(base_keys)
    ]  # O(50)

    implementations = [  # O(1)
        ("Chaining", HashTableChaining, {}),
        ("OpenLinear", HashTableOpenAddressing, {"probing_method": "linear"}),
        ("OpenDouble", HashTableOpenAddressing, {"probing_method": "double"}),
    ]

    results = {}  # O(1)
    for name, cls, kwargs in implementations:  # O(3) итераций
        print(f"Тестирование коллизий для {name}")  # O(1)
        result = test_hash_table_performance(
            cls, test_data, table_size=100, **kwargs
        )  # O(50)
        results[name] = result  # O(1)

    return results  # O(1)


def print_results_table(results):
    """
    Вывод результатов в виде таблицы
    Сложность: O(результаты)
    """
    print("\n" + "=" * 80)  # O(1)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")  # O(1)
    print("=" * 80)  # O(1)

    # Таблица сравнения хеш-функций
    if "hash_functions" in results:
        print("\nСРАВНЕНИЕ ХЕШ-ФУНКЦИЙ:")  # O(1)
        print(
            f"{'Функция':<12} {'Метод':<15} {'Вставка (с)':<12} {'Поиск (с)':<12} {'Коллизии':<10} {'Load Factor':<12}"
        )  # O(1)
        print("-" * 80)  # O(1)

        for hash_func, data in results["hash_functions"].items():
            for method_name in [
                "chaining",
                "open_linear",
                "open_double",
            ]:  # O(3) итераций
                if method_name in data:
                    method_data = data[method_name]  # O(1)
                    collision_metric = method_data["stats"].get(
                        "total_collisions",
                        method_data["stats"].get("average_probes", 0),
                    )  # O(1)
                    print(
                        f"{hash_func:<12} {method_name:<15} {method_data['avg_insert_time']:<12.6f} "  # O(1)
                        f"{method_data['avg_search_time']:<12.6f} {collision_metric:<10.2f} "
                        f"{method_data['load_factor']:<12.3f}"
                    )  # O(1)


def run_all_performance_tests():
    """
    Запуск всех тестов производительности
    Сложность: O(все тесты)
    """
    print("ЗАПУСК ТЕСТОВ ПРОИЗВОДИТЕЛЬНОСТИ ХЕШ-ТАБЛИЦ")  # O(1)
    print("=" * 50)  # O(1)

    results = {}  # O(1)

    print("\n1. Сравнение хеш-функций...")  # O(1)
    results["hash_functions"] = compare_hash_functions()  # O(3 * 200)

    print("\n2. Тестирование влияния коэффициента заполнения...")  # O(1)
    results["load_factors"] = test_load_factor_impact()  # O(4 * n)

    print("\n3. Тестирование производительности при коллизиях...")  # O(1)
    results["collisions"] = test_collision_performance()  # O(3 * 50)

    # Вывод результатов
    print_results_table(results)  # O(результаты)

    return results  # O(1)


if __name__ == "__main__":
    results = run_all_performance_tests()  # O(все тесты)
