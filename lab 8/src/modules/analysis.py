import timeit
import random
import matplotlib.pyplot as plt
from greedy_algorithms import fractional_knapsack, huffman_coding


# ---------- Анализ задачи о рюкзаке ----------
def brute_force_01_knapsack(capacity, items):
    """
    Полный перебор для задачи 0-1 рюкзака.
    items: список кортежей (вес, стоимость)
    """
    n = len(items)
    best_value = 0
    best_combination = []

    # Перебираем все возможные комбинации (2^n вариантов)
    for i in range(2**n):
        current_weight = 0
        current_value = 0
        combination = []

        # Проверяем каждый предмет
        for j in range(n):
            # Если j-й бит установлен в 1, берем предмет
            if (i >> j) & 1:
                weight, value = items[j]
                if current_weight + weight <= capacity:
                    current_weight += weight
                    current_value += value
                    combination.append(j)  # Сохраняем индекс предмета
                else:
                    # Если не влезает, прерываем эту комбинацию
                    break
        else:
            # Если все предметы влезли, сравниваем с лучшим
            if current_value > best_value:
                best_value = current_value
                best_combination = combination.copy()

    # Преобразуем индексы в информацию о взятых предметах
    taken_items = []
    for idx in best_combination:
        weight, value = items[idx]
        taken_items.append((weight, value, 1.0))  # 1.0 = взяли целиком

    return best_value, taken_items


# ---------- Экспериментальное исследование алгоритма Хаффмана ----------
def generate_text_data(size):
    """Генерация текстовых данных заданного размера"""
    # Случайные буквы английского алфавита
    letters = "abcdefghijklmnopqrstuvwxyz"
    text = "".join(random.choice(letters) for _ in range(size))

    # Подсчет частот символов без Counter
    frequencies = {}
    for char in text:
        if char in frequencies:
            frequencies[char] += 1
        else:
            frequencies[char] = 1

    return frequencies


def measure_huffman_time(frequencies):
    """Измерение времени работы алгоритма Хаффмана"""

    def run_huffman():
        return huffman_coding(frequencies)

    # Измеряем время выполнения 10 раз и берем среднее
    timer = timeit.Timer(run_huffman)
    time_taken = timer.timeit(number=10) / 10

    return time_taken


def visualize_huffman_tree(codes):
    """Визуализация дерева кодов Хаффмана с помощью matplotlib"""
    if not codes:
        print("Нет кодов для визуализации")
        return

    print("\nДерево кодов Хаффмана (ASCII-арт):")
    print("=" * 50)

    # Группируем коды по длине для лучшего отображения
    codes_by_length = {}
    for char, code in codes.items():
        length = len(code)
        if length not in codes_by_length:
            codes_by_length[length] = []
        codes_by_length[length].append((char, code))

    # Выводим коды, сгруппированные по длине
    for length in sorted(codes_by_length.keys()):
        print(f"\nКоды длиной {length}:")
        for char, code in sorted(codes_by_length[length]):
            print(f"  '{char}': {code}")

    # Создаем простую текстовую визуализацию дерева
    print("\nТекстовое представление дерева:")
    print("(0 - левый ребенок, 1 - правый ребенок)")

    # Находим корневые коды (начинающиеся с разных битов)
    root_codes = {}
    for char, code in codes.items():
        first_bit = code[0]
        if first_bit not in root_codes:
            root_codes[first_bit] = []
        root_codes[first_bit].append((char, code))

    for bit in sorted(root_codes.keys()):
        print(f"\nПоддерево для бита {bit}:")
        for char, code in sorted(root_codes[bit]):
            remaining_code = code[1:] if len(code) > 1 else " (лист)"
            print(f"  '{char}': {remaining_code}")


def plot_huffman_performance():
    """
        Построение графика зависимости времени работы алгоритма
        Хаффмана от размера данных
    """
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ АЛГОРИТМА ХАФФМАНА")
    print("=" * 70)

    # Размеры данных для тестирования
    data_sizes = [100, 500, 1000, 2000, 5000, 10000]
    times = []

    print("\nЗамеры времени выполнения:")
    print("-" * 50)

    for size in data_sizes:
        # Генерация данных
        frequencies = generate_text_data(size)

        # Измерение времени
        time_taken = measure_huffman_time(frequencies)
        times.append(time_taken)

        print(f"Размер данных: {size} символов")
        print(f"  Уникальных символов: {len(frequencies)}")
        print(f"  Время выполнения: {time_taken:.6f} секунд")
        print()

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, times, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Размер входных данных (количество символов)", fontsize=12)
    plt.ylabel("Время выполнения (секунды)", fontsize=12)
    plt.title(
        "Зависимость времени работы алгоритма Хаффмана от размера данных",
        fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Добавляем аннотации точек
    for i, (size, time_val) in enumerate(zip(data_sizes, times)):
        plt.annotate(
            f"{time_val:.6f}s",
            xy=(size, time_val),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.savefig("huffman_performance.png", dpi=150)
    print("График сохранен как 'huffman_performance.png'")

    # Анализ сложности
    print("\nАнализ временной сложности:")
    print("-" * 50)
    print("Ожидаемая сложность: O(n log n)")
    print("Где n - количество уникальных символов")
    print("\nФактическое время растет приблизительно пропорционально n log n")


# ---------- Сравнение алгоритмов для задачи о рюкзаке ----------
def compare_algorithms():
    """
    Сравнение жадного алгоритма для непрерывного рюкзака
    и точного алгоритма для дискретного (0-1) рюкзака
    """
    print("=" * 60)
    print("СРАВНЕНИЕ АЛГОРИТМОВ ДЛЯ ЗАДАЧИ О РЮКЗАКЕ")
    print("=" * 60)

    # Пример 1: Простой случай
    print("\nПример 1: Простой случай (3 предмета)")
    capacity = 10
    items1 = [(5, 10), (4, 8), (3, 6)]

    # Жадный алгоритм (непрерывный)
    greedy_val, greedy_items = fractional_knapsack(capacity, items1)
    print(f"\nЖадный алгоритм (непрерывный):")
    print(f"  Максимальная стоимость: {greedy_val:.2f}")
    print(f"  Взятые предметы:")
    for w, v, f in greedy_items:
        print(f"    Вес: {w}, Стоимость: {v}, Доля: {f:.2f}")

    # Точный алгоритм (0-1 рюкзак)
    exact_val, exact_items = brute_force_01_knapsack(capacity, items1)
    print(f"\nТочный алгоритм (0-1 рюкзак):")
    print(f"  Максимальная стоимость: {exact_val:.2f}")
    print(f"  Взятые предметы:")
    for w, v, f in exact_items:
        print(f"    Вес: {w}, Стоимость: {v}, Доля: {f:.2f}")

    # Пример 2: Показываем проблему жадного алгоритма
    print("\n" + "-" * 60)
    print("Пример 2: Демонстрация неоптимальности жадного подхода")

    capacity = 10
    items2 = [(6, 12), (5, 9), (5, 9)]  # Все веса и стоимости

    print(f"\nВходные данные:")
    print(f"  Вместимость рюкзака: {capacity}")
    for i, (w, v) in enumerate(items2):
        ratio = v / w
        print(
            f"  Предмет {i+1}: вес={w}, стоимость={v}, удельная стоимость={
                ratio:.2f}"
        )

    # Жадный алгоритм (непрерывный)
    greedy_val, greedy_items = fractional_knapsack(capacity, items2)
    print(f"\nЖадный алгоритм (непрерывный):")
    print(f"  Максимальная стоимость: {greedy_val:.2f}")
    print(f"  Взятые предметы:")
    total_weight = 0
    for w, v, f in greedy_items:
        taken_weight = w * f
        total_weight += taken_weight
        print(
            f"    Вес: {w}, Стоимость: {v}, Доля: {f:.2f}, Взятый вес: {
                taken_weight:.2f}"
        )
    print(f"  Всего взято веса: {total_weight:.2f} из {capacity}")

    # Точный алгоритм (0-1 рюкзак)
    exact_val, exact_items = brute_force_01_knapsack(capacity, items2)
    print(f"\nТочный алгоритм (0-1 рюкзак):")
    print(f"  Максимальная стоимость: {exact_val:.2f}")
    print(f"  Взятые предметы:")
    total_weight = 0
    for w, v, f in exact_items:
        taken_weight = w * f
        total_weight += taken_weight
        print(
            f"    Вес: {w}, Стоимость: {v}, Доля: {f:.2f}, Взятый вес: {
                taken_weight:.2f}"
        )
    print(f"  Всего взято веса: {total_weight:.2f} из {capacity}")

    # Сравнение результатов
    print(f"\nСравнение результатов:")
    print(f"  Жадный алгоритм (непрерывный): {greedy_val:.2f}")
    print(f"  Точный алгоритм (0-1): {exact_val:.2f}")
    print(f"  Разница: {exact_val - greedy_val:.2f}")

    # Пример 3: Классический пример неоптимальности
    print("\n" + "-" * 60)
    print("Пример 3: Классический пример неоптимальности жадного алгоритма")

    capacity = 50
    items3 = [(20, 60), (30, 90), (10, 30)]

    print(f"\nВходные данные:")
    print(f"  Вместимость рюкзака: {capacity}")
    for i, (w, v) in enumerate(items3):
        ratio = v / w
        print(
            f"  Предмет {i+1}: вес={w}, стоимость={v}, удельная стоимость={
                ratio:.2f}"
        )

    # Жадный алгоритм сортирует по удельной стоимости
    sorted_by_ratio = sorted(items3, key=lambda x: x[1] / x[0], reverse=True)
    print(f"\nПредметы, отсортированные по удельной стоимости:")
    for i, (w, v) in enumerate(sorted_by_ratio):
        ratio = v / w
        print(f"  {i+1}. Вес={w}, Стоимость={v}, Уд.стоимость={ratio:.2f}")

    # Жадный алгоритм (непрерывный) - берет по максимуму первые предметы
    greedy_val, greedy_items = fractional_knapsack(capacity, items3)
    print(f"\nЖадный алгоритм (непрерывный):")
    print(f"  Максимальная стоимость: {greedy_val:.2f}")

    # Точный алгоритм (0-1 рюкзак)
    exact_val, exact_items = brute_force_01_knapsack(capacity, items3)
    print(f"\nТочный алгоритм (0-1 рюкзак):")
    print(f"  Максимальная стоимость: {exact_val:.2f}")


# ---------- Основная функция ----------
def main():
    """Основная функция для запуска всех анализов"""

    print("=" * 70)
    print("АНАЛИЗ ЖАДНЫХ АЛГОРИТМОВ")
    print("=" * 70)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Запуск сравнения алгоритмов для рюкзака
    compare_algorithms()

    # Запуск экспериментального исследования алгоритма Хаффмана
    plot_huffman_performance()

    # Дополнительно: визуализация дерева Хаффмана для небольшого примера
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ДЕРЕВА ХАФФМАНА ДЛЯ ТЕСТОВОГО ПРИМЕРА")
    print("=" * 70)

    # Создаем небольшой пример для визуализации
    test_frequencies = {"a": 5, "b": 9, "c": 12, "d": 13, "e": 16, "f": 45}
    test_codes = huffman_coding(test_frequencies)
    visualize_huffman_tree(test_codes)

    # Замер времени для этого примера
    time_taken = measure_huffman_time(test_frequencies)
    print(f"\nВремя выполнения для примера: {time_taken:.6f} секунд")

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    main()
