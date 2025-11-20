import timeit
import matplotlib.pyplot as plt
import random
from modules.heap import MinHeap
from modules.heapsort import heapsort


def build_heap_sequential_insert(arr):
    """Построение кучи последовательной вставкой O(n log n)"""
    heap = MinHeap()
    for item in arr:
        heap.insert(item)
    return heap


def build_heap_algorithm(arr):
    """Построение кучи алгоритмом build_heap O(n)"""
    heap = MinHeap()
    heap.build_heap(arr)
    return heap


def quicksort(arr):
    """Быстрая сортировка для сравнения"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def mergesort(arr):
    """Сортировка слиянием для сравнения"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    """Слияние для сортировки слиянием"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def measure_heap_construction_time():
    """Измерение времени построения кучи разными методами"""
    print("=== ИЗМЕРЕНИЕ ВРЕМЕНИ ПОСТРОЕНИЯ КУЧИ ===")

    sizes = [100, 500, 1000, 2000, 5000, 10000]
    sequential_times = []
    build_heap_times = []

    for size in sizes:
        # Генерируем случайный массив
        test_array = [random.randint(1, 10000) for _ in range(size)]

        # Измеряем время последовательной вставки
        sequential_time = (
            timeit.timeit(lambda: build_heap_sequential_insert(test_array), number=10)
            / 10
        )

        # Измеряем время алгоритма build_heap
        build_heap_time = (
            timeit.timeit(lambda: build_heap_algorithm(test_array), number=10) / 10
        )

        sequential_times.append(sequential_time)
        build_heap_times.append(build_heap_time)

        print(
            f"Размер: {size:5d} | "
            f"Последовательная вставка: {sequential_time:.6f} сек | "
            f"Build_Heap: {build_heap_time:.6f} сек | "
            f"Ускорение: {sequential_time/build_heap_time:.2f}x"
        )

    return sizes, sequential_times, build_heap_times


def measure_sorting_algorithms_time():
    """Сравнение времени работы алгоритмов сортировки"""
    print("\n=== СРАВНЕНИЕ АЛГОРИТМОВ СОРТИРОВКИ ===")

    sizes = [100, 500, 1000, 2000, 5000]
    heapsort_times = []
    quicksort_times = []
    mergesort_times = []

    for size in sizes:
        # Генерируем случайный массив
        test_array = [random.randint(1, 10000) for _ in range(size)]

        # Heapsort
        heapsort_time = timeit.timeit(lambda: heapsort(test_array[:]), number=10) / 10

        # Quicksort
        quicksort_time = timeit.timeit(lambda: quicksort(test_array[:]), number=10) / 10

        # Mergesort
        mergesort_time = timeit.timeit(lambda: mergesort(test_array[:]), number=10) / 10

        heapsort_times.append(heapsort_time)
        quicksort_times.append(quicksort_time)
        mergesort_times.append(mergesort_time)

        print(
            f"Размер: {size:5d} | "
            f"Heapsort: {heapsort_time:.6f} сек | "
            f"Quicksort: {quicksort_time:.6f} сек | "
            f"Mergesort: {mergesort_time:.6f} сек"
        )

    return sizes, heapsort_times, quicksort_times, mergesort_times


def visualize_heap_array(heap, title="Куча"):
    """Визуализация кучи как массива с правильными связями"""
    if not heap.tree:
        print(f"{title}: пустая")
        return

    print(f"\n{title} (массив с связями):")

    for i, val in enumerate(heap.tree):
        parent_idx = (i - 1) // 2
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2

        # Формируем информацию о связях
        connections = []

        if i == 0:
            connections.append("корень")
        else:
            connections.append(f"родитель[{parent_idx}]={heap.tree[parent_idx]}")

        if left_idx < len(heap.tree):
            connections.append(f"левый[{left_idx}]={heap.tree[left_idx]}")

        if right_idx < len(heap.tree):
            connections.append(f"правый[{right_idx}]={heap.tree[right_idx]}")

        connections_str = ", ".join(connections)
        print(f"  [{i:2d}]: {val:2d}  ({connections_str})")


def demo_heap_visualization():
    """Демонстрация визуализации куч"""
    print("\n=== ВИЗУАЛИЗАЦИЯ КУЧ ===")

    # Создаем тестовую кучу
    test_data = [5, 3, 8, 1, 2, 7, 4, 6]

    # Куча построенная последовательной вставкой
    heap_seq = build_heap_sequential_insert(test_data)
    visualize_heap_array(heap_seq, "Куча (последовательная вставка)")

    # Куча построенная алгоритмом build_heap
    heap_build = build_heap_algorithm(test_data)
    visualize_heap_array(heap_build, "Куча (build_heap)")

    # Проверяем что обе кучи имеют одинаковые элементы (возможно в разном порядке)
    print(f"\nЭлементы совпадают: {sorted(heap_seq.tree) == sorted(heap_build.tree)}")


def plot_performance_graphs(
    sizes_heap,
    sequential_times,
    build_heap_times,
    sizes_sort,
    heapsort_times,
    quicksort_times,
    mergesort_times,
):
    """Построение графиков производительности"""

    plt.figure(figsize=(15, 8))

    # График 1: Построение кучи
    plt.subplot(2, 2, 1)
    plt.plot(
        sizes_heap,
        sequential_times,
        "o-",
        label="Последовательная вставка",
        linewidth=2,
    )
    plt.plot(
        sizes_heap, build_heap_times, "s-", label="Build_Heap алгоритм", linewidth=2
    )
    plt.xlabel("Размер массива")
    plt.ylabel("Время (секунды)")
    plt.title("Время построения кучи")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # График 2: Отношение времени построения
    plt.subplot(2, 2, 2)
    ratios = [seq / build for seq, build in zip(sequential_times, build_heap_times)]
    plt.plot(sizes_heap, ratios, "o-", color="red", linewidth=2)
    plt.xlabel("Размер массива")
    plt.ylabel("Отношение времен (последовательная/build_heap)")
    plt.title("Эффективность Build_Heap алгоритма")
    plt.grid(True, alpha=0.3)

    # График 3: Сравнение алгоритмов сортировки
    plt.subplot(2, 2, 3)
    plt.plot(sizes_sort, heapsort_times, "o-", label="Heapsort", linewidth=2)
    plt.plot(sizes_sort, quicksort_times, "s-", label="Quicksort", linewidth=2)
    plt.plot(sizes_sort, mergesort_times, "^-", label="Mergesort", linewidth=2)
    plt.xlabel("Размер массива")
    plt.ylabel("Время (секунды)")
    plt.title("Сравнение алгоритмов сортировки")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # График 4: Отношение Heapsort к другим алгоритмам
    plt.subplot(2, 2, 4)
    heapsort_vs_quicksort = [ht / qt for ht, qt in zip(heapsort_times, quicksort_times)]
    heapsort_vs_mergesort = [ht / mt for ht, mt in zip(heapsort_times, mergesort_times)]

    plt.plot(
        sizes_sort,
        heapsort_vs_quicksort,
        "o-",
        label="Heapsort / Quicksort",
        linewidth=2,
    )
    plt.plot(
        sizes_sort,
        heapsort_vs_mergesort,
        "s-",
        label="Heapsort / Mergesort",
        linewidth=2,
    )
    plt.xlabel("Размер массива")
    plt.ylabel("Отношение времени")
    plt.title("Heapsort относительно других алгоритмов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_comprehensive_analysis():
    """Запуск полного анализа"""
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ АЛГОРИТМОВ КУЧИ И СОРТИРОВКИ")
    print("=" * 70)

    # Демонстрация визуализации
    demo_heap_visualization()

    # Измерение времени построения кучи
    sizes_heap, sequential_times, build_heap_times = measure_heap_construction_time()

    # Измерение времени сортировки
    sizes_sort, heapsort_times, quicksort_times, mergesort_times = (
        measure_sorting_algorithms_time()
    )

    # Построение графиков
    plot_performance_graphs(
        sizes_heap,
        sequential_times,
        build_heap_times,
        sizes_sort,
        heapsort_times,
        quicksort_times,
        mergesort_times,
    )

    # Вывод статистики
    print("\n=== СТАТИСТИКА ===")
    avg_heap_ratio = sum(sequential_times) / sum(build_heap_times)
    print(f"Среднее ускорение Build_Heap: {avg_heap_ratio:.2f}x")

    print("\nЛучшие алгоритмы сортировки:")
    for i, size in enumerate(sizes_sort):
        times = {
            "Heapsort": heapsort_times[i],
            "Quicksort": quicksort_times[i],
            "Mergesort": mergesort_times[i],
        }
        best = min(times, key=times.get)
        print(f"  Размер {size}: {best} ({times[best]:.6f} сек)")
