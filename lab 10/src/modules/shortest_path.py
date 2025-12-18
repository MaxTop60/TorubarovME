import heapq
import math
from collections import deque, defaultdict

# ====================== Поиск компонент связности ======================


def connected_components_bfs(graph):
    """
    Поиск компонент связности с помощью BFS.

    Args:
        graph: граф (объект AdjacencyList или AdjacencyMatrix)

    Returns:
        Список компонент связности, каждая компонента - список вершин

    Сложность: O(V + E) для списка смежности, O(V²) для матрицы
    Особенности: Работает для неориентированных графов
    """
    if graph.directed:
        raise ValueError(
            "Для ориентированных графов используйте strongly_connected_components"
        )

    visited = set()
    components = []

    # Функция BFS для одной компоненты
    def bfs(start_vertex):
        component = []
        queue = deque([start_vertex])
        visited.add(start_vertex)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component

    # Обходим все вершины
    for vertex in graph.vertices:
        if vertex not in visited:
            component = bfs(vertex)
            components.append(sorted(component))

    return components


def connected_components_dfs(graph):
    """
    Поиск компонент связности с помощью DFS.

    Args:
        graph: граф (объект AdjacencyList или AdjacencyMatrix)

    Returns:
        Список компонент связности

    Сложность: O(V + E) для списка смежности, O(V²) для матрицы
    Особенности: Рекурсивная реализация
    """
    if graph.directed:
        raise ValueError(
            "Для ориентированных графов используйте strongly_connected_components"
        )

    visited = set()
    components = []

    def dfs(vertex, component):
        visited.add(vertex)
        component.append(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs(neighbor, component)

    for vertex in graph.vertices:
        if vertex not in visited:
            component = []
            dfs(vertex, component)
            components.append(sorted(component))

    return components


def strongly_connected_components(graph):
    """
    Поиск компонент сильной связности в ориентированном графе (алгоритм Косарайю).

    Args:
        graph: ориентированный граф (AdjacencyList)

    Returns:
        Список компонент сильной связности

    Сложность: O(V + E)
    Особенности: Двухпроходный алгоритм (прямой и обратный обход)
    """
    if not graph.directed:
        raise ValueError("Алгоритм Косарайю работает только для ориентированных графов")

    # Первый проход: обычный DFS для получения порядка завершения
    visited = set()
    finish_order = []

    def dfs_first(vertex):
        visited.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_first(neighbor)

        finish_order.append(vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            dfs_first(vertex)

    # Создаем транспонированный граф
    transposed = defaultdict(list)
    for u in graph.vertices:
        for v in graph.get_neighbors(u):
            transposed[v].append(u)

    # Второй проход: DFS на транспонированном графе
    visited.clear()
    components = []

    def dfs_second(vertex, component):
        visited.add(vertex)
        component.append(vertex)

        for neighbor in transposed[vertex]:
            if neighbor not in visited:
                dfs_second(neighbor, component)

    # Обходим в порядке убывания времени завершения
    for vertex in reversed(finish_order):
        if vertex not in visited:
            component = []
            dfs_second(vertex, component)
            components.append(sorted(component))

    return components


# ====================== Топологическая сортировка ======================


def topological_sort_kahn(graph):
    """
    Топологическая сортировка (алгоритм Кана).

    Args:
        graph: ориентированный ациклический граф (DAG)

    Returns:
        Список вершин в топологическом порядке

    Сложность: O(V + E)
    Особенности: Использует поиск вершин без входящих ребер
    """
    if not graph.directed:
        raise ValueError(
            "Топологическая сортировка возможна только для ориентированных графов"
        )

    # Вычисляем полустепень захода для каждой вершины
    in_degree = {vertex: 0 for vertex in graph.vertices}

    for u in graph.vertices:
        for v in graph.get_neighbors(u):
            in_degree[v] = in_degree.get(v, 0) + 1

    # Очередь вершин с нулевой полустепенью захода
    queue = deque([v for v in in_degree if in_degree[v] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)

        for v in graph.get_neighbors(u):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Проверка на цикл
    if len(topo_order) != len(graph.vertices):
        print("Граф содержит цикл, топологическая сортировка невозможна")

    return topo_order


def topological_sort_dfs(graph):
    """
    Топологическая сортировка с помощью DFS.

    Args:
        graph: ориентированный ациклический граф (DAG)

    Returns:
        Список вершин в топологическом порядке

    Сложность: O(V + E)
    Особенности: Использует обратный порядок завершения DFS
    """
    if not graph.directed:
        raise ValueError(
            "Топологическая сортировка возможна только для ориентированных графов"
        )

    visited = set()
    temp_visited = set()  # Для обнаружения циклов
    topo_order = []

    def dfs(vertex):
        # Обнаружение цикла
        if vertex in temp_visited:
            raise ValueError("Граф содержит цикл, топологическая сортировка невозможна")

        if vertex in visited:
            return

        temp_visited.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            dfs(neighbor)

        temp_visited.remove(vertex)
        visited.add(vertex)
        topo_order.append(vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            dfs(vertex)

    # Разворачиваем порядок (DFS добавляет в конец при завершении)
    return list(reversed(topo_order))


# ====================== Алгоритм Дейкстры ======================


def dijkstra_adjacency_list(graph, start_vertex):
    """
    Алгоритм Дейкстры для нахождения кратчайших путей во взвешенном графе.

    Args:
        graph: взвешенный граф (AdjacencyList)
        start_vertex: начальная вершина

    Returns:
        distances: словарь расстояний от start_vertex до всех вершин
        predecessors: словарь предшественников для восстановления путей

    Сложность: O((V + E) log V) с использованием кучи
    Особенности: Работает только с неотрицательными весами
    """
    if not graph.weighted:
        raise ValueError("Алгоритм Дейкстры требует взвешенный граф")

    if start_vertex not in graph.vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    distances = {vertex: math.inf for vertex in graph.vertices}
    distances[start_vertex] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Приоритетная очередь (мин-куча)
    priority_queue = [(0, start_vertex)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Если нашли более короткий путь через другую вершину, пропускаем
        if current_distance > distances[current_vertex]:
            continue

        # Обход соседей текущей вершины
        for neighbor, weight in graph.get_neighbors_with_weights(current_vertex):
            # Проверка на отрицательные веса
            if weight < 0:
                raise ValueError(
                    "Алгоритм Дейкстры не работает с отрицательными весами"
                )

            distance = current_distance + weight

            # Если нашли более короткий путь до соседа
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


def dijkstra_adjacency_matrix(graph, start_vertex):
    """
    Алгоритм Дейкстры для матрицы смежности.

    Args:
        graph: взвешенный граф (AdjacencyMatrix)
        start_vertex: начальная вершина

    Returns:
        distances: список расстояний от start_vertex до всех вершин
        predecessors: список предшественников

    Сложность: O(V²) - для каждой вершины ищем минимальное расстояние
    Особенности: Без кучи, подходит для плотных графов
    """
    if not graph.weighted:
        raise ValueError("Алгоритм Дейкстры требует взвешенный граф")

    if start_vertex < 0 or start_vertex >= graph.num_vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    distances = [math.inf] * graph.num_vertices
    distances[start_vertex] = 0
    predecessors = [-1] * graph.num_vertices
    visited = [False] * graph.num_vertices

    for _ in range(graph.num_vertices):
        # Находим вершину с минимальным расстоянием среди непосещенных
        min_distance = math.inf
        u = -1

        for v in range(graph.num_vertices):
            if not visited[v] and distances[v] < min_distance:
                min_distance = distances[v]
                u = v

        # Если все достижимые вершины посещены
        if u == -1:
            break

        visited[u] = True

        # Обновляем расстояния до соседей
        for v in range(graph.num_vertices):
            if not visited[v] and graph.has_edge(u, v):
                weight = graph.get_edge_weight(u, v)

                if weight is None:
                    continue

                # Проверка на отрицательные веса
                if weight < 0:
                    raise ValueError(
                        "Алгоритм Дейкстры не работает с отрицательными весами"
                    )

                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

    return distances, predecessors


def dijkstra_get_path(predecessors, target_vertex):
    """
    Восстановление кратчайшего пути по словарю предшественников.

    Args:
        predecessors: словарь/список предшественников из алгоритма Дейкстры
        target_vertex: целевая вершина

    Returns:
        Список вершин кратчайшего пути от начальной до целевой
    """
    if isinstance(predecessors, list):
        # Для списка (матрица смежности)
        if target_vertex < 0 or target_vertex >= len(predecessors):
            return []

        path = []
        current = target_vertex

        while current != -1:
            path.append(current)
            current = predecessors[current]

        path.reverse()
        return path
    else:
        # Для словаря (список смежности)
        if target_vertex not in predecessors:
            return []

        path = []
        current = target_vertex

        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()
        return path


def dijkstra_single_source_all_targets(graph, start_vertex):
    """
    Алгоритм Дейкстры с возвратом всех кратчайших путей.

    Args:
        graph: взвешенный граф
        start_vertex: начальная вершина

    Returns:
        distances: расстояния до всех вершин
        paths: кратчайшие пути до всех вершин
    """
    if hasattr(graph, "matrix"):
        distances, predecessors = dijkstra_adjacency_matrix(graph, start_vertex)
    else:
        distances, predecessors = dijkstra_adjacency_list(graph, start_vertex)

    paths = {}

    if isinstance(distances, list):
        # Для матрицы смежности
        for v in range(len(distances)):
            if distances[v] != math.inf:
                paths[v] = dijkstra_get_path(predecessors, v)
    else:
        # Для списка смежности
        for v in distances:
            if distances[v] != math.inf:
                paths[v] = dijkstra_get_path(predecessors, v)

    return distances, paths


# ====================== Вспомогательные функции ======================


def has_negative_cycle(graph):
    """
    Проверка наличия отрицательного цикла во взвешенном графе.

    Args:
        graph: взвешенный граф

    Returns:
        True если есть отрицательный цикл, иначе False

    Сложность: O(V * E) - алгоритм Беллмана-Форда
    """
    if not graph.weighted:
        return False

    # Инициализация
    if hasattr(graph, "vertices"):
        # Список смежности
        vertices = list(graph.vertices)
        distances = {v: 0 for v in vertices}

        # Алгоритм Беллмана-Форда
        for _ in range(len(vertices) - 1):
            for u in vertices:
                for v, weight in graph.get_neighbors_with_weights(u):
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight

        # Проверка на отрицательные циклы
        for u in vertices:
            for v, weight in graph.get_neighbors_with_weights(u):
                if distances[u] + weight < distances[v]:
                    return True
    else:
        # Матрица смежности
        distances = [0] * graph.num_vertices

        for _ in range(graph.num_vertices - 1):
            for u in range(graph.num_vertices):
                for v in range(graph.num_vertices):
                    if graph.has_edge(u, v):
                        weight = graph.get_edge_weight(u, v)
                        if weight is not None and distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight

        # Проверка на отрицательные циклы
        for u in range(graph.num_vertices):
            for v in range(graph.num_vertices):
                if graph.has_edge(u, v):
                    weight = graph.get_edge_weight(u, v)
                    if weight is not None and distances[u] + weight < distances[v]:
                        return True

    return False
