def factorial(n):
    # Функция вычисляет факториал числа n
    if n == 1:
        return 1  # O(1)
    else:
        return n * factorial(n - 1)  # O(n)
    # Общая сложность: O(n)
    # Глубина: O(n)


def fibonacci(n):
    # Функция вычисляет n-е число Фибоначчи
    if n == 0:  # O(1)
        return 0  # O(1)
    elif n == 1:  # O(1)
        return 1  # O(1)
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)  # O(2^n)
    # Общая сложность: O(2^n)
    # Глубина: O(2^n)


def fast_power(a, n):
    # Возведение a в n через степени двойки
    if n == 0:
        return 1  # O(1)
    if n == 1:
        return a  # O(1)

    half = fast_power(a, n // 2)  # O(log n)

    if n % 2 == 0:  # O(1)
        return half * half  # O(1)
    else:
        return half * half * a  # O(1)
    # Общая сложность: O(log n)
    # Глубина: O(log n)
