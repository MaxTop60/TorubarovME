import timeit
import matplotlib.pyplot as plt
import random
from Modules.binary_search_tree import BinarySearchTree


def build_balanced_tree(size):
    """Построение сбалансированного дерева (случайный порядок вставки)"""
    bst = BinarySearchTree()
    numbers = list(range(size))
    random.shuffle(numbers)  # Перемешиваем для балансировки
    for num in numbers:
        bst.insert(num)
    return bst


def build_degenerate_tree(size):
    """Построение вырожденного дерева (отсортированный порядок вставки)"""
    bst = BinarySearchTree()
    for num in range(size):
        bst.insert(num)  # Вставка в отсортированном порядке
    return bst


def measure_search_time(tree, search_operations=1000):
    """Измерение времени выполнения операций поиска"""

    def search_wrapper():
        # Выполняем search_operations случайных поисков
        for _ in range(search_operations):
            target = random.randint(0, tree.find_max(tree.root).value - 1)
            tree.search(target)

    # Используем timeit для точного измерения
    timer = timeit.Timer(search_wrapper)
    return timer.timeit(number=1)  # Выполняем один раз и возвращаем время


def visualize_tree_simple(tree, node=None, level=0, prefix="Root: "):
    """Простая текстовая визуализация дерева"""
    if node is None:
        node = tree.root

    if node is None:
        return "Empty tree"

    result = ""
    if level == 0:
        result += f"{prefix}{node.value}\n"
    else:
        result += "    " * (level - 1) + f"└── {node.value}\n"

    if node.left:
        result += visualize_tree_simple(tree, node.left, level + 1, "L: ")
    if node.right:
        result += visualize_tree_simple(tree, node.right, level + 1, "R: ")

    return result


def run_experiment():
    """Запуск полного эксперимента"""
    print("=== ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ BST ===")

    # Размеры деревьев для тестирования
    sizes = [100, 500, 1000, 2000, 5000]
    search_operations = 1000

    balanced_times = []
    degenerate_times = []

    print(f"\nИзмерение времени {search_operations} операций поиска:")
    print("Размер | Сбалансированное | Вырожденное")
    print("-" * 45)

    for size in sizes:
        print(f"{size:6} |", end="")

        # Сбалансированное дерево
        balanced_tree = build_balanced_tree(size)
        balanced_time = measure_search_time(balanced_tree, search_operations)
        balanced_times.append(balanced_time)
        print(f" {balanced_time:8.4f} сек    |", end="")

        # Вырожденное дерево
        degenerate_tree = build_degenerate_tree(size)
        degenerate_time = measure_search_time(degenerate_tree,
                                              search_operations)
        degenerate_times.append(degenerate_time)
        print(f" {degenerate_time:8.4f} сек")

    return sizes, balanced_times, degenerate_times


def plot_results(sizes, balanced_times, degenerate_times):
    """Построение графиков результатов"""
    plt.figure(figsize=(12, 5))

    # График 1: Время выполнения
    plt.subplot(1, 2, 1)
    plt.plot(sizes, balanced_times, "o-", label="Сбалансированное дерево",
             linewidth=2)
    plt.plot(sizes, degenerate_times, "s-", label="Вырожденное дерево",
             linewidth=2)
    plt.xlabel("Количество элементов в дереве")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Время 1000 операций поиска")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Отношение времен
    plt.subplot(1, 2, 2)
    ratios = [deg / bal for bal, deg in zip(balanced_times, degenerate_times)]
    plt.plot(sizes, ratios, "o-", color="red", linewidth=2)
    plt.xlabel("Количество элементов в дереве")
    plt.ylabel("Отношение времен (вырожденное/сбалансированное)")
    plt.title("Эффективность сбалансированного дерева")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bst_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Вывод статистики
    print(f"\n=== СТАТИСТИКА ===")
    print(f"Максимальное ускорение: {max(ratios):.2f}x")
    print(f"Минимальное ускорение: {min(ratios):.2f}x")
    print(f"Среднее ускорение: {sum(ratios)/len(ratios):.2f}x")


def demo_visualization():
    """Демонстрация визуализации деревьев"""
    print("\n=== ВИЗУАЛИЗАЦИЯ ДЕРЕВЬЕВ ===")

    # Сбалансированное дерево (маленькое для наглядности)
    print("Сбалансированное дерево (10 элементов):")
    balanced_small = build_balanced_tree(10)
    print("Текстовая визуализация:")
    print(visualize_tree_simple(balanced_small))

    print("\n" + "=" * 50 + "\n")

    # Вырожденное дерево (маленькое для наглядности)
    print("Вырожденное дерево (10 элементов):")
    degenerate_small = build_degenerate_tree(10)
    print("Текстовая визуализация:")
    print(visualize_tree_simple(degenerate_small))
