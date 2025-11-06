import timeit
import matplotlib.pyplot as plt


def count_calls(func):
    # Декоратор для подсчета вызовов функции
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


@count_calls
def fibonacci_memo(n, memo={}):
    # Функция вычисления n числа Фибоначи с мемоизацией
    if n in memo:  # O(1)
        return memo[n]
    if n == 1:  # O(1)
        return 1
    if n == 0:
        return 0  #

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)  # O(1)
    return memo[n]  # O(1)
    # Общая сложность: O(n)


@count_calls
def fibonacci(n):
    # Функция вычисляет n-е число Фибоначчи
    if n == 0:  # O(1)
        return 0  # O(1)
    elif n == 1:  # O(1)
        return 1  # O(1)
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)  # O(2^n)
    # Общая сложность: O(2^n)


# Характеристики ПК
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i3-1220P @ 1.5GHz
- Оперативная память: 8 GB DDR4
- ОС: Windows 11
- Python: 3.12.10
"""
print(pc_info)

# Замер времени выполнения и глубины рекурсии при n = 35
print("Замер выремени выполнения и глубины рекурсии при n = 35:")
print("{:>10} {:>25} {:>25}".format(
    "Тип функции", "Время", "Глубина рекурсии"))

time_non_memo = timeit.timeit(lambda: fibonacci(35), number=1)
count_non_memo = fibonacci.call_count
time_memo = timeit.timeit(lambda: fibonacci_memo(35), number=1)
count_memo = fibonacci_memo.call_count

print("{:>10} {:>25.4f} {:>25}".format(
    "Без мемоизации", time_non_memo, count_non_memo))
print("{:>10} {:>25.4f} {:>25}".format("С мемоизацией", time_memo, count_memo))


# Сравнение времени выполнения при разных n

# Диапазон значений n
n_values = list(range(1, 30))

# Время выполнения наивной и мемоизированной функции
print("""Замер выремени выполнения наивной
       и мемоизированной функций при разных N:""")
print("{:>10} {:>30} {:>30}".format(
    "N", "Время (мкс) - naive", "Время (мкс) - memo"))
times_non_memo = []
times_memo = []

for n in n_values:
    time_non_memo = timeit.timeit(lambda: fibonacci(n), number=10) * 1000 / 10
    times_non_memo.append(time_non_memo)
    time_memo = timeit.timeit(lambda: fibonacci_memo(n), number=10) * 1000 / 10
    times_memo.append(time_memo)
    print("{:>10} {:>30.4f} {:>30.4f}".format(n, time_non_memo, time_memo))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_non_memo, label="Без мемоизации", marker="o")
plt.plot(n_values, times_memo, label="С мемоизацией", marker="x")
plt.xlabel("N (порядковый номер числа Фибоначчи)")
plt.ylabel("Время выполнения (мкс)")
plt.title("Сравнение времени выполнения: Фибоначчи с мемоизацией и без")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fibonacci_time_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
