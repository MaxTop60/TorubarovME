def min_coins_greedy(amount, coins):
    """
    Жадный алгоритм для задачи о минимальном количестве монет.
    Работает корректно для канонических систем монет
    (например, 1, 2, 5, 10...).

    Args:
        amount: сумма сдачи
        coins: список доступных монет в порядке убывания номинала

    Returns:
        кортеж (минимальное количество монет, список монет для выдачи)
    """
    # Сортируем монеты по убыванию номинала
    coins_sorted = sorted(coins, reverse=True)

    coin_count = 0
    result_coins = []

    for coin in coins_sorted:
        if amount == 0:
            break

        # Берем максимально возможное количество монет данного номинала
        count = amount // coin
        if count > 0:
            coin_count += count
            result_coins.extend([coin] * count)
            amount -= count * coin

    # Если не удалось набрать точную сумму
    if amount > 0:
        raise ValueError(f"Невозможно выдать сумму {
            amount} доступными монетами")

    return coin_count, result_coins


def test_min_coins_greedy():
    """
    Тестирование жадного алгоритма для задачи о минимальном количестве монет.
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЖАДНОГО АЛГОРИТМА ДЛЯ ЗАДАЧИ О МОНЕТАХ")
    print("=" * 60)

    # Тест 1: Стандартная российская система монет
    print(
        "\nТест 1: Российская система монет [1, 2, 5, 10, 50, 100, 200, 500]")
    coins1 = [1, 2, 5, 10, 50, 100, 200, 500]
    test_cases1 = [
        (28, "28 руб."),
        (137, "137 руб."),
        (543, "543 руб."),
        (1024, "1024 руб."),
    ]

    for amount, description in test_cases1:
        try:
            count, coins_used = min_coins_greedy(amount, coins1)
            print(f"\n{description}:")
            print(f"  Количество монет: {count}")
            print(f"  Монеты: {coins_used}")

            # Проверка
            if sum(coins_used) == amount and len(coins_used) == count:
                print(f"  Корректно")
            else:
                print(f"  Ошибка: сумма {sum(coins_used)} != {amount}")
        except ValueError as e:
            print(f"\n{description}: {e}")

    # Тест 2: Американская система центов
    print("\n" + "-" * 60)
    print("Тест 2: Американская система [1, 5, 10, 25] (центы)")
    coins2 = [1, 5, 10, 25]
    test_cases2 = [
        (41, "41 цент"),
        (63, "63 цента"),
        (99, "99 центов"),
    ]

    for amount, description in test_cases2:
        try:
            count, coins_used = min_coins_greedy(amount, coins2)
            print(f"\n{description}:")
            print(f"  Количество монет: {count}")
            print(f"  Монеты: {coins_used}")

            # Проверка
            if sum(coins_used) == amount and len(coins_used) == count:
                print(f"  Корректно")
            else:
                print(f"  Ошибка: сумма {sum(coins_used)} != {amount}")
        except ValueError as e:
            print(f"\n{description}: {e}")

    # Пример работы алгоритма по шагам
    print("\n" + "=" * 60)
    print("ПОШАГОВЫЙ ПРИМЕР РАБОТЫ АЛГОРИТМА")
    print("=" * 60)

    amount = 137
    coins = [500, 200, 100, 50, 10, 5, 2, 1]

    print(f"\nСумма: {amount}")
    print(f"Монеты (по убыванию): {coins}")
    print("\nШаги алгоритма:")

    remaining = amount
    total_coins = 0
    result = []

    for coin in coins:
        if remaining == 0:
            break

        count = remaining // coin
        if count > 0:
            total_coins += count
            for _ in range(count):
                result.append(coin)
            remaining -= count * coin
            print(f"  Берём {count} монет(у) номиналом {coin}. Осталось: {
                remaining}")

    print(f"\nИтого: {total_coins} монет")
    print(f"Монеты: {result}")
    print(f"Проверка суммы: {sum(result)} = {amount}")


if __name__ == "__main__":
    test_min_coins_greedy()
