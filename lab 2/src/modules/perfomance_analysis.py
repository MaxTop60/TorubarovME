from .linked_list import LinkedList
import timeit
import random
import collections
import matplotlib.pyplot as plt


# Функция для добавления элементов в начало списка
def list_prepend(list, size):
    for i in range(size):  # O(n)
        list.insert(0, random.randint(0, 1000))  # O(n)
    # Общая сложность: O(n^2)


def linked_list_prepend(linked_list, size):
    for i in range(size):  # O(n)
        linked_list.insert_at_start(random.randint(0, 1000))  # O(1)
    # Общая сложность: O(n)


def comparision(sizes):
    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    list = []
    linked_list = LinkedList()
    times_list_insert = []
    times_linked_list_insert = []
    times_list_pop = []
    times_deque_pop = []

    print(
        """Замеры времени выполнения для list и
        linked_list (добавление N элементов в начало):"""
    )
    print(
        "{:>10} {:>19} {:>30}".format(
            "N", "Время (мкс) - list", "Время (мкс) - linked_list"
        )
    )

    # Замеры времени выполнения при сравнении list и linked_list
    for size in sizes:
        time_1 = timeit.timeit(
            lambda: list_prepend(list, size), number=10) * 1000 / 10
        times_list_insert.append(time_1)
        time_2 = (
            timeit.timeit(
                lambda: linked_list_prepend(linked_list, size), number=10)
            * 1000
            / 10
        )
        times_linked_list_insert.append(time_2)
        print(f"{size:>10} {time_1:>19.4f} {time_2:>30.4f}")

    print(
        """Замеры времени выполнения для list и deque
        (удаление из начала N колличества элементов):"""
    )
    print("{:>19} {:>30}".format("Время (мкс) - list", "Время (мкс) - deque"))

    deque = collections.deque()
    # Замеры времени выполнения при сравнении list и deque
    for size in sizes:
        list = []
        for i in range(size):  # Заполнение списка
            deque.appendleft(random.randint(0, size))
            list.append(random.randint(0, size))

        time_1 = timeit.timeit(lambda: list.pop(0), number=1) * 1000
        times_list_pop.append(time_1)
        time_2 = timeit.timeit(lambda: deque.popleft(), number=1) * 1000
        times_deque_pop.append(time_2)

        print(f"{size:>10} {time_1:>19.4f} {time_2:>30.4f}")

    # График сравнения list и linked_list
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_list_insert, "bo-", label="list")
    plt.plot(sizes, times_linked_list_insert, "ro-", label="linked_list")
    plt.xlabel("Колличество элементов (N)")
    plt.ylabel("Время выполнения (мкс)")
    plt.title(
        """Зависимость времени выполнения от колличества элементов
        \n(Сравнение list и linked_list)"""
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "time_complexity_plot_linked.png", dpi=300, bbox_inches="tight")
    plt.show()

    # График сравнения list и deque
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_list_insert, "bo-", label="list")
    plt.plot(sizes, times_linked_list_insert, "ro-", label="deque")
    plt.xlabel("Колличество элементов (N)")
    plt.ylabel("Время выполнения (мкс)")
    plt.title(
        """Зависимость времени выполнения от колличества элементов
        \n(Сравнение list и deque)"""
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_complexity_plot_deque.png", dpi=300, bbox_inches="tight")
    plt.show()
