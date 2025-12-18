from collections import deque

# ====================== Поиск в ширину (BFS) ======================


def bfs_adjacency_matrix(graph, start_vertex):
    """
    Поиск в ширину для графа, представленного матрицей смежности.

    Args:
        graph: объект AdjacencyMatrix
        start_vertex: индекс начальной вершины

    Returns:
        distances: словарь расстояний от start_vertex до всех вершин
        parents: словарь родителей для восстановления путей

    Сложность: O(V²) - необходимо проверять все вершины на наличие ребра
    Особенности: Использует матрицу смежности, подходит для плотных графов
    """
    if start_vertex < 0 or start_vertex >= graph.num_vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    visited = [False] * graph.num_vertices
    distances = [-1] * graph.num_vertices  # -1 означает недостижимо
    parents = [-1] * graph.num_vertices  # -1 означает отсутствие родителя
    queue = deque()

    visited[start_vertex] = True
    distances[start_vertex] = 0
    queue.append(start_vertex)

    # Обход в ширину
    while queue:
        current = queue.popleft()
        current_distance = distances[current]

        # Получаем соседей через матрицу
        for neighbor in range(graph.num_vertices):
            # Проверяем наличие ребра
            if graph.has_edge(current, neighbor) and not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = current_distance + 1
                parents[neighbor] = current
                queue.append(neighbor)

    return distances, parents


def bfs_adjacency_list(graph, start_vertex):
    """
    Поиск в ширину для графа, представленного списком смежности.

    Args:
        graph: объект AdjacencyList
        start_vertex: идентификатор начальной вершины

    Returns:
        distances: словарь расстояний от start_vertex до всех вершин
        parents: словарь родителей для восстановления путей

    Сложность: O(V + E) - каждая вершина и ребро посещаются один раз
    Особенности: Более эффективен для разреженных графов
    """
    if start_vertex not in graph.vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    visited = set()
    distances = {}
    parents = {}
    queue = deque()

    visited.add(start_vertex)
    distances[start_vertex] = 0
    parents[start_vertex] = None
    queue.append(start_vertex)

    # Обход в ширину
    while queue:
        current = queue.popleft()
        current_distance = distances[current]

        # Получаем соседей через список смежности
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_distance + 1
                parents[neighbor] = current
                queue.append(neighbor)

    return distances, parents


def bfs_get_path(parents, target_vertex):
    """
    Восстановление пути от начальной вершины до target_vertex.

    Args:
        parents: словарь родителей, полученный из BFS
        target_vertex: целевая вершина

    Returns:
        Список вершин пути от начальной до целевой
    """
    # Если вершина не найдена в словаре родителей
    if target_vertex not in parents:
        return []

    # Если вершина является начальной (родитель None)
    if parents[target_vertex] is None:
        return [target_vertex]

    path = []
    current = target_vertex

    # Восстанавливаем путь в обратном порядке
    while current is not None:
        path.append(current)
        # Проверяем, есть ли родитель у текущей вершины
        if current in parents:
            current = parents[current]
        else:
            break

    # Разворачиваем путь, чтобы он шел от начальной вершины к целевой
    path.reverse()
    return path


def bfs_complete_traversal(graph, start_vertex):
    """
    Полный обход графа с помощью BFS с обработкой компонент связности.

    Args:
        graph: граф (объект AdjacencyList)
        start_vertex: начальная вершина

    Returns:
        visited: множество всех посещенных вершин
        components: список компонент связности
        distances: словарь расстояний

    Сложность: O(V + E) для списка смежности, O(V²) для матрицы
    """
    if isinstance(graph, list) or hasattr(graph, "matrix"):
        # Для матрицы смежности
        distances, parents = bfs_adjacency_matrix(graph, start_vertex)
        visited = {i for i, d in enumerate(distances) if d != -1}
    else:
        # Для списка смежности
        distances, parents = bfs_adjacency_list(graph, start_vertex)
        visited = set(distances.keys())

    # Находим компоненты связности
    components = []
    all_vertices = set()

    if hasattr(graph, "vertices"):
        # Для списка смежности
        all_vertices = graph.vertices.copy()
    elif hasattr(graph, "num_vertices"):
        # Для матрицы смежности
        all_vertices = set(range(graph.num_vertices))

    # Компонента, содержащая стартовую вершину
    if visited:
        components.append(sorted(visited))

    # Находим остальные компоненты
    remaining = all_vertices - visited

    while remaining:
        # Выбираем произвольную вершину из оставшихся
        next_start = next(iter(remaining))

        if hasattr(graph, "matrix"):
            # Матрица смежности
            dist, _ = bfs_adjacency_matrix(graph, next_start)
            new_component = {i for i, d in enumerate(dist) if d != -1}
        else:
            # Список смежности
            dist, _ = bfs_adjacency_list(graph, next_start)
            new_component = set(dist.keys())

        components.append(sorted(new_component))
        remaining -= new_component

    return visited, components, distances


# ====================== Поиск в глубину (DFS) ======================


def dfs_iterative_adjacency_matrix(graph, start_vertex):
    """
    Итеративный поиск в глубину для матрицы смежности.

    Args:
        graph: объект AdjacencyMatrix
        start_vertex: индекс начальной вершины

    Returns:
        traversal_order: порядок обхода вершин
        parents: словарь родителей

    Сложность: O(V²) - необходимо проверять все вершины на наличие ребра
    Особенности: Использует явный стек, не рискует переполнить стек вызовов
    """
    if start_vertex < 0 or start_vertex >= graph.num_vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    visited = [False] * graph.num_vertices
    parents = [-1] * graph.num_vertices
    traversal_order = []
    stack = [start_vertex]

    while stack:
        current = stack.pop()

        if not visited[current]:
            visited[current] = True
            traversal_order.append(current)

            # Добавляем соседей в обратном порядке для правильного порядка обхода
            neighbors = []
            for neighbor in range(graph.num_vertices):
                if graph.has_edge(current, neighbor) and not visited[neighbor]:
                    neighbors.append(neighbor)
                    if parents[neighbor] == -1:
                        parents[neighbor] = current

            # Добавляем в стек в обратном порядке
            for neighbor in reversed(neighbors):
                stack.append(neighbor)

    return traversal_order, parents


def dfs_iterative_adjacency_list(graph, start_vertex):
    """
    Итеративный поиск в глубину для списка смежности.

    Args:
        graph: объект AdjacencyList
        start_vertex: идентификатор начальной вершины

    Returns:
        traversal_order: порядок обхода вершин
        parents: словарь родителей

    Сложность: O(V + E) - каждая вершина и ребро посещаются один раз
    Особенности: Более эффективен, использует явный стек
    """
    if start_vertex not in graph.vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация
    visited = set()
    parents = {}
    traversal_order = []
    stack = [start_vertex]

    while stack:
        current = stack.pop()

        if current not in visited:
            visited.add(current)
            traversal_order.append(current)

            # Получаем соседей
            neighbors = graph.get_neighbors(current)

            # Добавляем непосещенных соседей в стек
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
                    if neighbor not in parents:
                        parents[neighbor] = current

    return traversal_order, parents


def dfs_recursive_adjacency_list(
    graph, start_vertex, visited=None, parents=None, traversal_order=None
):
    """
    Рекурсивный поиск в глубину для списка смежности.

    Args:
        graph: объект AdjacencyList
        start_vertex: идентификатор начальной вершины
        visited: множество посещенных вершин (для рекурсии)
        parents: словарь родителей (для рекурсии)
        traversal_order: список порядка обхода (для рекурсии)

    Returns:
        visited, parents, traversal_order

    Сложность: O(V + E) - каждая вершина и ребро посещаются один раз
    Особенности: Простая реализация, но может переполнить стек для больших графов
    """
    if start_vertex not in graph.vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация при первом вызове
    if visited is None:
        visited = set()
    if parents is None:
        parents = {}
    if traversal_order is None:
        traversal_order = []

    # Помечаем вершину как посещенную
    visited.add(start_vertex)
    traversal_order.append(start_vertex)

    # Рекурсивно обходим всех соседей
    for neighbor in graph.get_neighbors(start_vertex):
        if neighbor not in visited:
            parents[neighbor] = start_vertex
            dfs_recursive_adjacency_list(
                graph, neighbor, visited, parents, traversal_order
            )

    return visited, parents, traversal_order


def dfs_recursive_matrix(
    graph, start_vertex, visited=None, parents=None, traversal_order=None
):
    """
    Рекурсивный поиск в глубину для матрицы смежности.

    Args:
        graph: объект AdjacencyMatrix
        start_vertex: индекс начальной вершины
        visited: список посещенных вершин (для рекурсии)
        parents: список родителей (для рекурсии)
        traversal_order: список порядка обхода (для рекурсии)

    Returns:
        visited, parents, traversal_order

    Сложность: O(V²) - необходимо проверять все вершины на наличие ребра
    Особенности: Менее эффективен из-за проверки всех вершин
    """
    if start_vertex < 0 or start_vertex >= graph.num_vertices:
        raise ValueError(f"Вершина {start_vertex} не существует в графе")

    # Инициализация при первом вызове
    if visited is None:
        visited = [False] * graph.num_vertices
    if parents is None:
        parents = [-1] * graph.num_vertices
    if traversal_order is None:
        traversal_order = []

    # Помечаем вершину как посещенную
    visited[start_vertex] = True
    traversal_order.append(start_vertex)

    # Рекурсивно обходим всех соседей
    for neighbor in range(graph.num_vertices):
        if graph.has_edge(start_vertex, neighbor) and not visited[neighbor]:
            parents[neighbor] = start_vertex
            dfs_recursive_matrix(graph, neighbor, visited, parents, traversal_order)

    return visited, parents, traversal_order


def dfs_detect_cycle_adjacency_list(graph, directed=True):
    """
    Обнаружение циклов в графе с помощью DFS.

    Args:
        graph: объект AdjacencyList
        directed: ориентированный ли граф

    Returns:
        True если граф содержит цикл, иначе False

    Сложность: O(V + E)
    Особенности: Использует три цвета для вершин
    """
    # Три цвета: 0 - не посещена, 1 - в процессе обработки, 2 - обработана
    color = {vertex: 0 for vertex in graph.vertices}

    def dfs_visit(vertex):
        color[vertex] = 1  # Начинаем обработку вершины

        for neighbor in graph.get_neighbors(vertex):
            if color[neighbor] == 0:
                # Вершина не посещена
                if dfs_visit(neighbor):
                    return True
            elif color[neighbor] == 1:
                # Обнаружен цикл
                if (
                    directed or parent != neighbor
                ):  # Для неориентированных игнорируем обратное ребро
                    return True

        color[vertex] = 2  # Завершаем обработку вершины
        return False

    # Для ориентированных графов достаточно одного вызова из каждой вершины
    if directed:
        for vertex in graph.vertices:
            if color[vertex] == 0:
                if dfs_visit(vertex):
                    return True
        return False
    else:
        # Для неориентированных графов используем модифицированный DFS
        visited = set()

        def dfs_undirected(vertex, parent):
            visited.add(vertex)

            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    if dfs_undirected(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    # Обнаружен цикл (не через родителя)
                    return True

            return False

        for vertex in graph.vertices:
            if vertex not in visited:
                if dfs_undirected(vertex, -1):
                    return True

        return False


def dfs_topological_sort(graph):
    """
    Топологическая сортировка ориентированного ациклического графа (DAG).

    Args:
        graph: ориентированный граф (объект AdjacencyList)

    Returns:
        Список вершин в топологическом порядке

    Сложность: O(V + E)
    Особенности: Работает только для DAG, использует обратный порядок завершения DFS
    """
    if not graph.directed:
        raise ValueError(
            "Топологическая сортировка возможна только для ориентированных графов"
        )

    visited = set()
    result = []

    def dfs_visit(vertex):
        visited.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_visit(neighbor)

        # Добавляем вершину в начало списка
        result.insert(0, vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            dfs_visit(vertex)

    return result
