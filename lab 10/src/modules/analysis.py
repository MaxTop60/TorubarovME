# Файл: analysis.py
import timeit
import random
import math
import matplotlib.pyplot as plt
from graph_representation import AdjacencyMatrix, AdjacencyList
from graph_traversal import (
    bfs_adjacency_matrix,
    bfs_adjacency_list,
    dfs_iterative_adjacency_matrix,
    dfs_iterative_adjacency_list,
)
from shortest_path import (
    dijkstra_adjacency_matrix,
    dijkstra_adjacency_list,
    connected_components_bfs,
    topological_sort_kahn,
)

# ====================== Генерация тестовых графов ======================


def generate_random_graph_matrix(
    num_vertices, edge_probability, directed=False, weighted=False
):
    """
    Генерация случайного графа в виде матрицы смежности.

    Args:
        num_vertices: количество вершин
        edge_probability: вероятность существования ребра
        directed: ориентированный ли граф
        weighted: взвешенный ли граф
    """
    graph = AdjacencyMatrix(num_vertices, directed, weighted)

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j or directed:  # Для неориентированных пропускаем петли
                if random.random() < edge_probability:
                    weight = random.randint(1, 10) if weighted else 1
                    graph.add_edge(i, j, weight)

    return graph


def generate_random_graph_list(
    num_vertices, edge_probability, directed=False, weighted=False
):
    """
    Генерация случайного графа в виде списка смежности.
    """
    graph = AdjacencyList(directed, weighted)

    # Добавляем все вершины
    for i in range(num_vertices):
        graph.add_vertex(i)

    # Добавляем случайные ребра
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j or directed:
                if random.random() < edge_probability:
                    weight = random.randint(1, 10) if weighted else 1
                    graph.add_edge(i, j, weight)

    return graph


def generate_sparse_graph(
    num_vertices, edges_per_vertex, directed=False, weighted=False
):
    """
    Генерация разреженного графа.
    """
    if directed:
        graph_list = AdjacencyList(directed=True, weighted=weighted)
        graph_matrix = AdjacencyMatrix(num_vertices, directed=True, weighted=weighted)
    else:
        graph_list = AdjacencyList(directed=False, weighted=weighted)
        graph_matrix = AdjacencyMatrix(num_vertices, directed=False, weighted=weighted)

    # Добавляем вершины
    for i in range(num_vertices):
        graph_list.add_vertex(i)

    # Добавляем ребра
    for i in range(num_vertices):
        for _ in range(edges_per_vertex):
            j = random.randint(0, num_vertices - 1)
            if i != j:
                weight = random.randint(1, 10) if weighted else 1
                graph_list.add_edge(i, j, weight)
                graph_matrix.add_edge(i, j, weight)

    return graph_list, graph_matrix


def generate_dense_graph(num_vertices, directed=False, weighted=False):
    """
    Генерация плотного графа (почти полного).
    """
    if directed:
        graph_list = AdjacencyList(directed=True, weighted=weighted)
        graph_matrix = AdjacencyMatrix(num_vertices, directed=True, weighted=weighted)
    else:
        graph_list = AdjacencyList(directed=False, weighted=weighted)
        graph_matrix = AdjacencyMatrix(num_vertices, directed=False, weighted=weighted)

    # Добавляем все вершины
    for i in range(num_vertices):
        graph_list.add_vertex(i)

    # Добавляем много ребер
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                if random.random() < 0.8:  # 80% вероятность ребра
                    weight = random.randint(1, 10) if weighted else 1
                    graph_list.add_edge(i, j, weight)
                    graph_matrix.add_edge(i, j, weight)

    return graph_list, graph_matrix


# ====================== Сравнение операций для разных представлений ======================


def compare_representation_operations():
    """
    Сравнение времени выполнения операций для разных представлений графов.
    """
    print("=" * 80)
    print("СРАВНЕНИЕ ОПЕРАЦИЙ ДЛЯ РАЗНЫХ ПРЕДСТАВЛЕНИЙ ГРАФОВ")
    print("=" * 80)

    # Размеры графов для тестирования
    sizes = [50, 100, 150, 200, 250]
    results = {
        "add_edge": {"matrix": [], "list": []},
        "has_edge": {"matrix": [], "list": []},
        "get_neighbors": {"matrix": [], "list": []},
    }

    print("\nЗамер времени операций (в микросекундах):")
    print("-" * 80)
    print("Размер | Добавление ребра | Проверка ребра | Получение соседей")
    print("       | Матрица | Список | Матрица | Список | Матрица | Список")
    print("-" * 80)

    random.seed(42)  # Для воспроизводимости

    for n in sizes:
        # Создаем графы
        graph_list = AdjacencyList(directed=False, weighted=False)
        graph_matrix = AdjacencyMatrix(n, directed=False, weighted=False)

        # Добавляем вершины
        for i in range(n):
            graph_list.add_vertex(i)

        # Тест добавления ребра
        def add_edge_matrix():
            for i in range(min(100, n)):
                for j in range(min(100, n)):
                    if i != j:
                        graph_matrix.add_edge(i, j)

        def add_edge_list():
            for i in range(min(100, n)):
                for j in range(min(100, n)):
                    if i != j:
                        graph_list.add_edge(i, j)

        time_matrix_add = timeit.timeit(add_edge_matrix, number=10) / 10 * 1e6
        time_list_add = timeit.timeit(add_edge_list, number=10) / 10 * 1e6

        # Добавляем некоторые ребра для тестирования других операций
        for i in range(min(50, n)):
            for j in range(i + 1, min(50, n), 2):
                graph_matrix.add_edge(i, j)
                graph_list.add_edge(i, j)

        # Тест проверки наличия ребра
        def has_edge_matrix():
            for i in range(min(100, n)):
                for j in range(min(100, n)):
                    graph_matrix.has_edge(i, j)

        def has_edge_list():
            for i in range(min(100, n)):
                for j in range(min(100, n)):
                    graph_list.has_edge(i, j)

        time_matrix_has = timeit.timeit(has_edge_matrix, number=10) / 10 * 1e6
        time_list_has = timeit.timeit(has_edge_list, number=10) / 10 * 1e6

        # Тест получения соседей
        def get_neighbors_matrix():
            for i in range(n):
                graph_matrix.get_neighbors(i)

        def get_neighbors_list():
            for i in range(n):
                graph_list.get_neighbors(i)

        time_matrix_neighbors = (
            timeit.timeit(get_neighbors_matrix, number=10) / 10 * 1e6
        )
        time_list_neighbors = timeit.timeit(get_neighbors_list, number=10) / 10 * 1e6

        # Сохраняем результаты
        results["add_edge"]["matrix"].append(time_matrix_add)
        results["add_edge"]["list"].append(time_list_add)
        results["has_edge"]["matrix"].append(time_matrix_has)
        results["has_edge"]["list"].append(time_list_has)
        results["get_neighbors"]["matrix"].append(time_matrix_neighbors)
        results["get_neighbors"]["list"].append(time_list_neighbors)

        print(
            f"{n:6d} | {time_matrix_add:8.1f} | {time_list_add:6.1f} | "
            f"{time_matrix_has:8.1f} | {time_list_has:6.1f} | "
            f"{time_matrix_neighbors:8.1f} | {time_list_neighbors:6.1f}"
        )

    # Построение графиков
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # График 1: Добавление ребра
    axes[0].plot(
        sizes, results["add_edge"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[0].plot(sizes, results["add_edge"]["list"], "bo-", label="Список", linewidth=2)
    axes[0].set_xlabel("Количество вершин")
    axes[0].set_ylabel("Время (микросекунды)")
    axes[0].set_title("Добавление ребра")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График 2: Проверка наличия ребра
    axes[1].plot(
        sizes, results["has_edge"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[1].plot(sizes, results["has_edge"]["list"], "bo-", label="Список", linewidth=2)
    axes[1].set_xlabel("Количество вершин")
    axes[1].set_ylabel("Время (микросекунды)")
    axes[1].set_title("Проверка наличия ребра")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # График 3: Получение соседей
    axes[2].plot(
        sizes, results["get_neighbors"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[2].plot(
        sizes, results["get_neighbors"]["list"], "bo-", label="Список", linewidth=2
    )
    axes[2].set_xlabel("Количество вершин")
    axes[2].set_ylabel("Время (микросекунды)")
    axes[2].set_title("Получение соседей вершины")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("representation_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО СРАВНЕНИЮ ПРЕДСТАВЛЕНИЙ:")
    print("=" * 80)
    print(
        """
1. Матрица смежности:
   • Быстрая проверка наличия ребра: O(1)
   • Медленное получение соседей: O(V)
   • Постоянное время добавления ребра: O(1)
   • Большой расход памяти: O(V²)

2. Список смежности:
   • Быстрое получение соседей: O(1) в среднем
   • Медленная проверка наличия ребра: O(deg(v))
   • Эффективное использование памяти: O(V + E)
   • Быстрое добавление ребра: O(1) в среднем
    """
    )


# ====================== Исследование масштабируемости алгоритмов ======================


def analyze_algorithm_scalability():
    """
    Исследование масштабируемости алгоритмов на больших графах.
    """
    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ МАСШТАБИРУЕМОСТИ АЛГОРИТМОВ")
    print("=" * 80)

    # Размеры графов для тестирования
    sizes = [50, 100, 150, 200, 250, 300]
    results = {
        "bfs": {"matrix": [], "list": []},
        "dfs": {"matrix": [], "list": []},
        "dijkstra": {"matrix": [], "list": []},
        "components": {"list": []},
        "topological": {"list": []},
    }

    print("\nЗамер времени выполнения алгоритмов (в миллисекундах):")
    print("-" * 80)
    print("Размер | BFS        | DFS        | Дейкстра   | Компоненты | Топ.сорт")
    print("       | Матр | Спис| Матр | Спис| Матр | Спис| Список     | Список")
    print("-" * 80)

    random.seed(42)

    for n in sizes:
        # Генерируем тестовые графы
        graph_list, graph_matrix = generate_sparse_graph(
            n, 3, directed=False, weighted=False
        )
        weighted_list, weighted_matrix = generate_sparse_graph(
            n, 3, directed=True, weighted=True
        )
        directed_list, _ = generate_sparse_graph(n, 2, directed=True, weighted=False)

        # Тест BFS
        def bfs_matrix():
            return bfs_adjacency_matrix(graph_matrix, 0)

        def bfs_list():
            return bfs_adjacency_list(graph_list, 0)

        time_bfs_matrix = timeit.timeit(bfs_matrix, number=10) / 10 * 1000
        time_bfs_list = timeit.timeit(bfs_list, number=10) / 10 * 1000

        # Тест DFS
        def dfs_matrix():
            return dfs_iterative_adjacency_matrix(graph_matrix, 0)

        def dfs_list():
            return dfs_iterative_adjacency_list(graph_list, 0)

        time_dfs_matrix = timeit.timeit(dfs_matrix, number=10) / 10 * 1000
        time_dfs_list = timeit.timeit(dfs_list, number=10) / 10 * 1000

        # Тест Дейкстры (только для взвешенных графов)
        def dijkstra_matrix():
            return dijkstra_adjacency_matrix(weighted_matrix, 0)

        def dijkstra_list():
            return dijkstra_adjacency_list(weighted_list, 0)

        time_dijkstra_matrix = timeit.timeit(dijkstra_matrix, number=5) / 5 * 1000
        time_dijkstra_list = timeit.timeit(dijkstra_list, number=5) / 5 * 1000

        # Тест компонент связности
        def components():
            return connected_components_bfs(graph_list)

        time_components = timeit.timeit(components, number=10) / 10 * 1000

        # Тест топологической сортировки (только для ориентированных)
        def topological():
            return topological_sort_kahn(directed_list)

        time_topological = timeit.timeit(topological, number=10) / 10 * 1000

        # Сохраняем результаты
        results["bfs"]["matrix"].append(time_bfs_matrix)
        results["bfs"]["list"].append(time_bfs_list)
        results["dfs"]["matrix"].append(time_dfs_matrix)
        results["dfs"]["list"].append(time_dfs_list)
        results["dijkstra"]["matrix"].append(time_dijkstra_matrix)
        results["dijkstra"]["list"].append(time_dijkstra_list)
        results["components"]["list"].append(time_components)
        results["topological"]["list"].append(time_topological)

        print(
            f"{n:6d} | {time_bfs_matrix:5.2f} | {time_bfs_list:4.2f} | "
            f"{time_dfs_matrix:5.2f} | {time_dfs_list:4.2f} | "
            f"{time_dijkstra_matrix:5.2f} | {time_dijkstra_list:4.2f} | "
            f"{time_components:8.2f} | {time_topological:8.2f}"
        )

    # Построение графиков
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # График 1: BFS сравнение
    axes[0, 0].plot(
        sizes, results["bfs"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[0, 0].plot(sizes, results["bfs"]["list"], "bo-", label="Список", linewidth=2)
    axes[0, 0].set_xlabel("Количество вершин")
    axes[0, 0].set_ylabel("Время (мс)")
    axes[0, 0].set_title("BFS (O(V²) vs O(V+E))")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: DFS сравнение
    axes[0, 1].plot(
        sizes, results["dfs"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[0, 1].plot(sizes, results["dfs"]["list"], "bo-", label="Список", linewidth=2)
    axes[0, 1].set_xlabel("Количество вершин")
    axes[0, 1].set_ylabel("Время (мс)")
    axes[0, 1].set_title("DFS (O(V²) vs O(V+E))")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Дейкстра сравнение
    axes[0, 2].plot(
        sizes, results["dijkstra"]["matrix"], "ro-", label="Матрица", linewidth=2
    )
    axes[0, 2].plot(
        sizes, results["dijkstra"]["list"], "bo-", label="Список", linewidth=2
    )
    axes[0, 2].set_xlabel("Количество вершин")
    axes[0, 2].set_ylabel("Время (мс)")
    axes[0, 2].set_title("Дейкстра (O(V²) vs O((V+E)logV))")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # График 4: Компоненты связности
    axes[1, 0].plot(
        sizes, results["components"]["list"], "go-", label="Список", linewidth=2
    )
    axes[1, 0].set_xlabel("Количество вершин")
    axes[1, 0].set_ylabel("Время (мс)")
    axes[1, 0].set_title("Компоненты связности (O(V+E))")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # График 5: Топологическая сортировка
    axes[1, 1].plot(
        sizes, results["topological"]["list"], "mo-", label="Список", linewidth=2
    )
    axes[1, 1].set_xlabel("Количество вершин")
    axes[1, 1].set_ylabel("Время (мс)")
    axes[1, 1].set_title("Топологическая сортировка (O(V+E))")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # График 6: Сводное сравнение
    axes[1, 2].plot(sizes, results["bfs"]["list"], "b-", label="BFS", linewidth=2)
    axes[1, 2].plot(sizes, results["dfs"]["list"], "g-", label="DFS", linewidth=2)
    axes[1, 2].plot(
        sizes, results["dijkstra"]["list"], "r-", label="Дейкстра", linewidth=2
    )
    axes[1, 2].plot(
        sizes, results["components"]["list"], "c-", label="Компоненты", linewidth=2
    )
    axes[1, 2].plot(
        sizes, results["topological"]["list"], "m-", label="Топ.сорт", linewidth=2
    )
    axes[1, 2].set_xlabel("Количество вершин")
    axes[1, 2].set_ylabel("Время (мс)")
    axes[1, 2].set_title("Сравнение алгоритмов (список смежности)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("algorithm_scalability.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Анализ роста времени
    print("\n" + "=" * 80)
    print("АНАЛИЗ РОСТА ВРЕМЕНИ:")
    print("=" * 80)

    print("\nОтношение времени при увеличении размера в 2 раза:")
    print("Алгоритм         | n=100 | n=200 | Ожидаемая сложность")
    print("-" * 60)

    for i, (size_small, size_large) in enumerate([(100, 200), (150, 300)]):
        if size_small in sizes and size_large in sizes:
            idx_small = sizes.index(size_small)
            idx_large = sizes.index(size_large)

            ratio_bfs_matrix = (
                results["bfs"]["matrix"][idx_large]
                / results["bfs"]["matrix"][idx_small]
            )
            ratio_bfs_list = (
                results["bfs"]["list"][idx_large] / results["bfs"]["list"][idx_small]
            )
            ratio_dijkstra_list = (
                results["dijkstra"]["list"][idx_large]
                / results["dijkstra"]["list"][idx_small]
            )

            print(
                f"BFS (матрица)     | {ratio_bfs_matrix:5.2f} | O(V²) -> ожидается ~4.0"
            )
            print(
                f"BFS (список)      | {ratio_bfs_list:5.2f} | O(V+E) -> ожидается ~2.0"
            )
            print(
                f"Дейкстра (список) | {ratio_dijkstra_list:5.2f} | O((V+E)logV) -> ожидается ~2.5"
            )


# ====================== Визуализация графов ======================


def visualize_small_graph():
    """
    Визуализация небольшого графа и результатов алгоритмов.
    """
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ МАЛЕНЬКОГО ГРАФА")
    print("=" * 80)

    # Создаем небольшой граф для визуализации
    graph = AdjacencyList(directed=False, weighted=True)

    # Простой граф с 6 вершинами
    edges = [
        (0, 1, 4),
        (0, 2, 2),
        (1, 2, 1),
        (1, 3, 5),
        (2, 3, 8),
        (2, 4, 10),
        (3, 4, 2),
        (3, 5, 6),
        (4, 5, 2),
    ]

    for u, v, w in edges:
        graph.add_edge(u, v, w)

    print("\nГраф для визуализации:")
    print(graph)

    # Запускаем алгоритмы
    print("\nРезультаты алгоритмов:")
    print("-" * 40)

    # BFS
    print("BFS от вершины 0:")
    distances_bfs, parents_bfs = bfs_adjacency_list(graph, 0)
    print(f"  Расстояния: {distances_bfs}")

    # Дейкстра
    print("\nДейкстра от вершины 0:")
    distances_dijkstra, parents_dijkstra = dijkstra_adjacency_list(graph, 0)
    print(f"  Кратчайшие расстояния: {distances_dijkstra}")

    # Компоненты связности
    print("\nКомпоненты связности:")
    components = connected_components_bfs(graph)
    print(f"  Компоненты: {components}")

    # Создаем простую текстовую визуализацию
    print("\n" + "=" * 80)
    print("ТЕКСТОВАЯ ВИЗУАЛИЗАЦИЯ ГРАФА:")
    print("=" * 80)

    print("\nСтруктура графа:")
    print("(числа в скобках - веса ребер)")
    print("-" * 40)

    for vertex in sorted(graph.vertices):
        neighbors = graph.get_neighbors_with_weights(vertex)
        if neighbors:
            neighbors_str = ", ".join(f"{v}({w})" for v, w in neighbors)
            print(f"Вершина {vertex}: {neighbors_str}")

    print("\nМатрица смежности (весов):")
    print("-" * 40)

    # Создаем матрицу весов
    vertices = sorted(graph.vertices)
    n = len(vertices)

    # Заголовок
    header = "   " + " ".join(f"{v:3}" for v in vertices)
    print(header)
    print("   " + "-" * (n * 3 + 1))

    # Строки матрицы
    for i in vertices:
        row = f"{i:2}|"
        for j in vertices:
            weight = graph.get_edge_weight(i, j)
            if weight is not None:
                row += f"{weight:3}"
            else:
                if i == j:
                    row += "  0"
                else:
                    row += "  -"
        print(row)

    print("\nКратчайшие пути от вершины 0:")
    print("-" * 40)
    for vertex in vertices:
        if vertex != 0:
            if distances_dijkstra[vertex] < math.inf:
                # Восстанавливаем путь
                path = []
                current = vertex
                while current is not None:
                    path.append(current)
                    current = parents_dijkstra.get(current)
                path.reverse()
                print(
                    f"  0 -> {vertex}: расстояние = {distances_dijkstra[vertex]}, путь = {path}"
                )


# ====================== Сравнение плотных и разреженных графов ======================


def compare_dense_vs_sparse():
    """
    Сравнение производительности на плотных и разреженных графах.
    """
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ПЛОТНЫХ И РАЗРЕЖЕННЫХ ГРАФОВ")
    print("=" * 80)

    sizes = [50, 100, 150, 200]
    results_dense = {"bfs": [], "dfs": [], "dijkstra": []}
    results_sparse = {"bfs": [], "dfs": [], "dijkstra": []}

    print("\nВремя выполнения на графах разной плотности (в миллисекундах):")
    print("-" * 80)
    print("Размер | Плотный граф         | Разреженный граф     ")
    print("       | BFS  | DFS  | Дейкстра | BFS  | DFS  | Дейкстра")
    print("-" * 80)

    random.seed(42)

    for n in sizes:
        # Генерируем графы
        dense_list, _ = generate_dense_graph(n, directed=False, weighted=True)
        sparse_list, _ = generate_sparse_graph(n, 3, directed=False, weighted=True)

        # Тесты для плотного графа
        def bfs_dense():
            return bfs_adjacency_list(dense_list, 0)

        def dfs_dense():
            return dfs_iterative_adjacency_list(dense_list, 0)

        def dijkstra_dense():
            return dijkstra_adjacency_list(dense_list, 0)

        time_bfs_dense = timeit.timeit(bfs_dense, number=5) / 5 * 1000
        time_dfs_dense = timeit.timeit(dfs_dense, number=5) / 5 * 1000
        time_dijkstra_dense = timeit.timeit(dijkstra_dense, number=3) / 3 * 1000

        # Тесты для разреженного графа
        def bfs_sparse():
            return bfs_adjacency_list(sparse_list, 0)

        def dfs_sparse():
            return dfs_iterative_adjacency_list(sparse_list, 0)

        def dijkstra_sparse():
            return dijkstra_adjacency_list(sparse_list, 0)

        time_bfs_sparse = timeit.timeit(bfs_sparse, number=5) / 5 * 1000
        time_dfs_sparse = timeit.timeit(dfs_sparse, number=5) / 5 * 1000
        time_dijkstra_sparse = timeit.timeit(dijkstra_sparse, number=3) / 3 * 1000

        # Сохраняем результаты
        results_dense["bfs"].append(time_bfs_dense)
        results_dense["dfs"].append(time_dfs_dense)
        results_dense["dijkstra"].append(time_dijkstra_dense)
        results_sparse["bfs"].append(time_bfs_sparse)
        results_sparse["dfs"].append(time_dfs_sparse)
        results_sparse["dijkstra"].append(time_dijkstra_sparse)

        print(
            f"{n:6d} | {time_bfs_dense:5.2f} | {time_dfs_dense:5.2f} | {time_dijkstra_dense:8.2f} | "
            f"{time_bfs_sparse:5.2f} | {time_dfs_sparse:5.2f} | {time_dijkstra_sparse:8.2f}"
        )

    # Построение графиков
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # График 1: BFS
    axes[0].plot(sizes, results_dense["bfs"], "r-", label="Плотный", linewidth=2)
    axes[0].plot(sizes, results_sparse["bfs"], "b-", label="Разреженный", linewidth=2)
    axes[0].set_xlabel("Количество вершин")
    axes[0].set_ylabel("Время (мс)")
    axes[0].set_title("BFS: плотный vs разреженный")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График 2: DFS
    axes[1].plot(sizes, results_dense["dfs"], "r-", label="Плотный", linewidth=2)
    axes[1].plot(sizes, results_sparse["dfs"], "b-", label="Разреженный", linewidth=2)
    axes[1].set_xlabel("Количество вершин")
    axes[1].set_ylabel("Время (мс)")
    axes[1].set_title("DFS: плотный vs разреженный")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # График 3: Дейкстра
    axes[2].plot(sizes, results_dense["dijkstra"], "r-", label="Плотный", linewidth=2)
    axes[2].plot(
        sizes, results_sparse["dijkstra"], "b-", label="Разреженный", linewidth=2
    )
    axes[2].set_xlabel("Количество вершин")
    axes[2].set_ylabel("Время (мс)")
    axes[2].set_title("Дейкстра: плотный vs разреженный")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dense_vs_sparse.png", dpi=150, bbox_inches="tight")

    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО ПЛОТНОСТИ ГРАФОВ:")
    print("=" * 80)
    print(
        """
1. На плотных графах:
   • BFS/DFS: O(V²) для матрицы, O(V²) для списка (так как E ≈ V²)
   • Дейкстра: O(V²) для матрицы, O(V² log V) для списка
   • Матрица смежности часто быстрее
   
2. На разреженных графах:
   • BFS/DFS: O(V+E) где E << V²
   • Дейкстра: O((V+E) log V)F
   • Список смежности значительно быстрее
    """
    )


# ====================== Основная функция ======================


def main():
    """Основная функция для запуска анализа."""
    print("=" * 80)
    print("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ ГРАФОВЫХ АЛГОРИТМОВ")
    print("=" * 80)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i3-1220P @ 1.5GHz
    - Оперативная память: 8 GB DDR4
    - ОС: Windows 11
    - Python: 3.12.10
    """
    print(pc_info)

    # Сравнение операций для разных представлений
    compare_representation_operations()

    # Исследование масштабируемости алгоритмов
    analyze_algorithm_scalability()

    # Визуализация маленького графа
    visualize_small_graph()

    # Сравнение плотных и разреженных графов
    compare_dense_vs_sparse()

    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\nСозданные графики:")
    print("  1. representation_comparison.png - сравнение операций")
    print("  2. algorithm_scalability.png - масштабируемость алгоритмов")
    print("  3. dense_vs_sparse.png - сравнение плотных и разреженных графов")


if __name__ == "__main__":
    main()
