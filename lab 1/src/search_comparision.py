import timeit
import matplotlib.pyplot as plt
import random


def linear_search(arr, target):
    """Алгоритм линейного поиска"""
    for i in arr:  # O(n)
        if i == target:  # O(1)
            return True  # O(1)
    return False  # O(1)
    # Общая сложность O(n)


def binary_search(arr, target):
    """Алгоритм бинарного поиска"""
    while len(arr) > 1:  # O(log n)
        mid = arr[len(arr) // 2]  # O(1)
        if mid == target:  # O(1)
            return True  # O(1)
        elif mid < target:  # O(1)
            arr = arr[:mid]  # O(1)
        else:  # O(1)
            arr = arr[mid:]  # o(1)
    return False  # O(1)
    # Общая сложность O(log n)


def timer(arr, target):
    """Функция для замера времени выполнения"""
    global times_linear
    global times_binary

    # Для линейного поиска
    execution_time_linear = (
        timeit.timeit(
            lambda: linear_search(arr, target), number=10) * 1000 / 10
    )
    times_linear.append(execution_time_linear)

    # Для бинарного поиска
    execution_time_binary = (
        timeit.timeit(
            lambda: binary_search(arr, target), number=10) * 1000 / 10
    )
    times_binary.append(execution_time_binary)

    print(
        "{:>10} {:>30.4f} {:>30.4f}".format(
            len(arr), execution_time_linear, execution_time_binary
        )
    )


# Характеристики ПК
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i3-1220P @ 1.5GHz
- Оперативная память: 8 GB DDR4
- ОС: Windows 11
- Python: 3.12.10
"""
print(pc_info)


sizes = [1000, 5000, 10000, 50000, 100000, 500000]
arrs = []
times_linear = []
times_binary = []

# Создание отсортированных массивов
for size in sizes:
    arr = []
    for i in range(size):
        arr.append(random.randint(0, 1000))
    arr.sort()
    arrs.append([arr])

# Выбор целевого элемента
arrs[0].append(arrs[0][0][0])  # Первый элемент
arrs[1].append(arrs[1][0][len(arrs[1][0]) // 2])  # Средний элемент
arrs[2].append(arrs[2][0][-1])  # Последний элемент
arrs[3].append(random.randint(0, arrs[3][0][-1]))  # Случайный элемент
arrs[4].append(1001)  # Не существует
arrs[5].append(random.randint(0, arrs[3][0][-1]))  # Случайный элемент

print("Замеры времени выполнения для алгоритмов поиска:")
print(
    "{:>10} {:>30} {:>30}".format(
        "Размер (N)", "Время (мкс) - Линейный поиск",
        "Время (мкс) - Бинарный поиск"
    )
)

# Замеры времени выполнения
for arr in arrs:
    timer(arr[0], arr[1])

# График в обычном масштабе
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_linear, "bo-", label="Линейный поиск")
plt.plot(sizes, times_binary, "ro-", label="Бинарный поиск")
plt.xlabel("Размер массива (N)")
plt.ylabel("Время выполнения (мкс)")
plt.title("""Зависимость времени выполнения от размера массива
          \n(Обычный масштаб)""")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("time_complexity_plot_linear.png", dpi=300, bbox_inches="tight")
plt.show()

# График в логарифмическом масштабе
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_linear, "bo-", label="Линейный поиск")
plt.plot(sizes, times_binary, "ro-", label="Бинарный поиск")
plt.xlabel("Размер массива (N)")
plt.ylabel("Время выполнения (мкс)")
plt.title("""Зависимость времени выполнения от размера массива
          \n(Логарифмический масштаб)""")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale("log")  # Логарифмический масштаб по X
plt.yscale("log")  # Логарифмический масштаб по Y
plt.legend()
plt.tight_layout()
plt.savefig("time_complexity_plot_log.png", dpi=300, bbox_inches="tight")
plt.show()
