# ====================== Класс для матрицы смежности ======================


class AdjacencyMatrix:
    """
    Представление графа с помощью матрицы смежности.

    Потребление памяти: O(V²), где V - количество вершин.
    """

    def __init__(self, num_vertices, directed=False, weighted=False):
        """
        Инициализация матрицы смежности.

        Args:
            num_vertices: количество вершин (вершины нумеруются от 0 до num_vertices-1)
            directed: ориентированный ли граф (по умолчанию False - неориентированный)
            weighted: взвешенный ли граф (по умолчанию False - невзвешенный)
        """
        self.num_vertices = num_vertices
        self.directed = directed
        self.weighted = weighted

        # Инициализация матрицы
        # Для невзвешенных графов: 0 - нет ребра, 1 - есть ребро
        # Для взвешенных графов: 0/None/INF - нет ребра, вес - есть ребро
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]

        if weighted:
            import math

            for i in range(num_vertices):
                for j in range(num_vertices):
                    if i != j:
                        self.matrix[i][j] = math.inf

    def add_edge(self, u, v, weight=1):
        """
        Добавление ребра между вершинами u и v.

        Args:
            u: индекс начальной вершины (0-based)
            v: индекс конечной вершины (0-based)
            weight: вес ребра (по умолчанию 1 для невзвешенных графов)

        Сложность: O(1)
        """
        if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
            raise ValueError(
                f"Индексы вершин должны быть в диапазоне [0, {self.num_vertices-1}]"
            )

        self.matrix[u][v] = weight

        if not self.directed:
            self.matrix[v][u] = weight

    def remove_edge(self, u, v):
        """
        Удаление ребра между вершинами u и v.

        Args:
            u: индекс начальной вершины
            v: индекс конечной вершины

        Сложность: O(1)
        """
        if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
            raise ValueError(
                f"Индексы вершин должны быть в диапазоне [0, {self.num_vertices-1}]"
            )

        if self.weighted:
            import math

            self.matrix[u][v] = math.inf
            if not self.directed:
                self.matrix[v][u] = math.inf
        else:
            self.matrix[u][v] = 0
            if not self.directed:
                self.matrix[v][u] = 0

    def has_edge(self, u, v):
        """
        Проверка наличия ребра между вершинами u и v.

        Args:
            u: индекс начальной вершины
            v: индекс конечной вершины

        Returns:
            True если ребро существует, иначе False

        Сложность: O(1)
        """
        if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
            return False

        if self.weighted:
            import math

            return self.matrix[u][v] != math.inf
        else:
            return self.matrix[u][v] != 0

    def get_edge_weight(self, u, v):
        """
        Получение веса ребра между вершинами u и v.

        Args:
            u: индекс начальной вершины
            v: индекс конечной вершины

        Returns:
            Вес ребра или None если ребро не существует

        Сложность: O(1)
        """
        if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
            return None

        if self.weighted:
            import math

            return self.matrix[u][v] if self.matrix[u][v] != math.inf else None
        else:
            return self.matrix[u][v] if self.matrix[u][v] != 0 else None

    def get_neighbors(self, vertex):
        """
        Получение списка соседей вершины.

        Args:
            vertex: индекс вершины

        Returns:
            Список индексов соседних вершин

        Сложность: O(V) - необходимо проверить все вершины
        """
        if not 0 <= vertex < self.num_vertices:
            return []

        neighbors = []

        if self.weighted:
            import math

            for i in range(self.num_vertices):
                if self.matrix[vertex][i] != math.inf:
                    neighbors.append(i)
        else:
            for i in range(self.num_vertices):
                if self.matrix[vertex][i] != 0:
                    neighbors.append(i)

        return neighbors

    def get_degree(self, vertex):
        """
        Получение степени вершины.

        Args:
            vertex: индекс вершины

        Returns:
            Степень вершины (количество инцидентных ребер)

        Сложность: O(V)
        """
        return len(self.get_neighbors(vertex))

    def get_all_edges(self):
        """
        Получение всех ребер графа.

        Returns:
            Список кортежей (u, v, weight)

        Сложность: O(V²)
        """
        edges = []

        if self.directed:
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if self.has_edge(i, j):
                        weight = self.get_edge_weight(i, j)
                        edges.append((i, j, weight))
        else:
            # Для неориентированных графов каждое ребро учитываем один раз
            for i in range(self.num_vertices):
                for j in range(i, self.num_vertices):
                    if self.has_edge(i, j):
                        weight = self.get_edge_weight(i, j)
                        edges.append((i, j, weight))

        return edges

    def add_vertex(self):
        """
        Добавление новой вершины.

        Returns:
            Индекс новой вершины

        Сложность: O(V) - создание новой матрицы
        """
        # Увеличиваем размер матрицы
        new_size = self.num_vertices + 1

        # Создаем новую матрицу
        new_matrix = [[0] * new_size for _ in range(new_size)]

        if self.weighted:
            import math

            # Копируем старую матрицу
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    new_matrix[i][j] = self.matrix[i][j]

            # Инициализируем новые строки/столбцы
            for i in range(new_size):
                for j in range(new_size):
                    if i >= self.num_vertices or j >= self.num_vertices:
                        if i == j:
                            new_matrix[i][j] = 0
                        else:
                            new_matrix[i][j] = math.inf
        else:
            # Копируем старую матрицу
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    new_matrix[i][j] = self.matrix[i][j]

        # Обновляем матрицу и количество вершин
        self.matrix = new_matrix
        self.num_vertices = new_size

        return new_size - 1

    def __str__(self):
        """Строковое представление матрицы смежности."""
        result = []
        result.append(f"Матрица смежности ({self.num_vertices} вершин):")
        result.append(f"Ориентированный: {self.directed}, Взвешенный: {self.weighted}")

        # Заголовок
        header = "   " + " ".join(f"{i:3}" for i in range(self.num_vertices))
        result.append(header)
        result.append("   " + "-" * (self.num_vertices * 3 + 1))

        # Строки матрицы
        for i in range(self.num_vertices):
            row = f"{i:2}|"
            for j in range(self.num_vertices):
                if self.weighted:
                    import math

                    if self.matrix[i][j] == math.inf:
                        row += " ∞ "
                    else:
                        row += f"{self.matrix[i][j]:3}"
                else:
                    row += f"{self.matrix[i][j]:3}"
            result.append(row)

        return "\n".join(result)


# ====================== Класс для списка смежности ======================


class AdjacencyList:
    """
    Представление графа с помощью списка смежности.

    Потребление памяти: O(V + E), где V - количество вершин, E - количество ребер.
    """

    def __init__(self, directed=False, weighted=False):
        """
        Инициализация списка смежности.

        Args:
            directed: ориентированный ли граф (по умолчанию False - неориентированный)
            weighted: взвешенный ли граф (по умолчанию False - невзвешенный)
        """
        self.directed = directed
        self.weighted = weighted
        self.adj_list = {}  # словарь: вершина -> список соседей
        self.vertices = set()  # множество всех вершин
        self.num_edges = 0

    def add_vertex(self, vertex):
        """
        Добавление вершины в граф.

        Args:
            vertex: идентификатор вершины (может быть строкой или числом)

        Сложность: O(1) в среднем случае
        """
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []
            self.vertices.add(vertex)

    def add_edge(self, u, v, weight=1):
        """
        Добавление ребра между вершинами u и v.

        Args:
            u: начальная вершина
            v: конечная вершина
            weight: вес ребра (по умолчанию 1 для невзвешенных графов)

        Сложность: O(1) в среднем случае
        """
        # Добавляем вершины, если их нет
        self.add_vertex(u)
        self.add_vertex(v)

        # Добавляем ребро u -> v
        if self.weighted:
            self.adj_list[u].append((v, weight))
        else:
            self.adj_list[u].append(v)

        # Если граф неориентированный, добавляем обратное ребро
        if not self.directed:
            if self.weighted:
                self.adj_list[v].append((u, weight))
            else:
                self.adj_list[v].append(u)

        self.num_edges += 1

    def remove_edge(self, u, v):
        """
        Удаление ребра между вершинами u и v.

        Args:
            u: начальная вершина
            v: конечная вершина

        Сложность: O(deg(u)) в среднем случае, где deg(u) - степень вершины u
        """
        if u not in self.adj_list or v not in self.adj_list:
            return

        # Удаляем ребро u -> v
        if self.weighted:
            # Ищем и удаляем пару (v, weight)
            for i, (neighbor, weight) in enumerate(self.adj_list[u]):
                if neighbor == v:
                    del self.adj_list[u][i]
                    break
        else:
            if v in self.adj_list[u]:
                self.adj_list[u].remove(v)

        # Если граф неориентированный, удаляем обратное ребро
        if not self.directed:
            if self.weighted:
                for i, (neighbor, weight) in enumerate(self.adj_list[v]):
                    if neighbor == u:
                        del self.adj_list[v][i]
                        break
            else:
                if u in self.adj_list[v]:
                    self.adj_list[v].remove(u)

        self.num_edges -= 1

    def has_edge(self, u, v):
        """
        Проверка наличия ребра между вершинами u и v.

        Args:
            u: начальная вершина
            v: конечная вершина

        Returns:
            True если ребро существует, иначе False

        Сложность: O(deg(u)) в среднем случае
        """
        if u not in self.adj_list:
            return False

        if self.weighted:
            for neighbor, weight in self.adj_list[u]:
                if neighbor == v:
                    return True
            return False
        else:
            return v in self.adj_list[u]

    def get_edge_weight(self, u, v):
        """
        Получение веса ребра между вершинами u и v.

        Args:
            u: начальная вершина
            v: конечная вершина

        Returns:
            Вес ребра или None если ребро не существует

        Сложность: O(deg(u)) в среднем случае
        """
        if u not in self.adj_list:
            return None

        if self.weighted:
            for neighbor, weight in self.adj_list[u]:
                if neighbor == v:
                    return weight
            return None
        else:
            return 1 if v in self.adj_list[u] else None

    def get_neighbors(self, vertex):
        """
        Получение списка соседей вершины.

        Args:
            vertex: идентификатор вершины

        Returns:
            Список соседних вершин

        Сложность: O(1) в среднем случае (возврат ссылки на список)
        """
        if vertex not in self.adj_list:
            return []

        if self.weighted:
            # Возвращаем только вершины без весов
            return [neighbor for neighbor, weight in self.adj_list[vertex]]
        else:
            return self.adj_list[vertex].copy()

    def get_neighbors_with_weights(self, vertex):
        """
        Получение списка соседей вершины с весами.

        Args:
            vertex: идентификатор вершины

        Returns:
            Список кортежей (сосед, вес) или список соседей если граф невзвешенный

        Сложность: O(1) в среднем случае
        """
        if vertex not in self.adj_list:
            return []

        if self.weighted:
            return self.adj_list[vertex].copy()
        else:
            return [(neighbor, 1) for neighbor in self.adj_list[vertex]]

    def get_degree(self, vertex):
        """
        Получение степени вершины.

        Args:
            vertex: идентификатор вершины

        Returns:
            Степень вершины (количество инцидентных ребер)

        Сложность: O(1) в среднем случае
        """
        if vertex not in self.adj_list:
            return 0
        return len(self.adj_list[vertex])

    def get_all_edges(self):
        """
        Получение всех ребер графа.

        Returns:
            Список кортежей (u, v, weight)

        Сложность: O(V + E)
        """
        edges = []
        visited = set()

        for u in self.adj_list:
            if self.weighted:
                for v, weight in self.adj_list[u]:
                    if self.directed:
                        edges.append((u, v, weight))
                    else:
                        # Для неориентированных графов каждое ребро учитываем один раз
                        edge_key = (min(u, v), max(u, v))
                        if edge_key not in visited:
                            edges.append((u, v, weight))
                            visited.add(edge_key)
            else:
                for v in self.adj_list[u]:
                    if self.directed:
                        edges.append((u, v, 1))
                    else:
                        edge_key = (min(u, v), max(u, v))
                        if edge_key not in visited:
                            edges.append((u, v, 1))
                            visited.add(edge_key)

        return edges

    def remove_vertex(self, vertex):
        """
        Удаление вершины из графа.

        Args:
            vertex: идентификатор вершины

        Сложность: O(E) в худшем случае
        """
        if vertex not in self.adj_list:
            return

        # Удаляем все ребра, инцидентные вершине
        if self.directed:
            # Удаляем все исходящие ребра
            outgoing_edges = len(self.adj_list[vertex])
            self.num_edges -= outgoing_edges

            # Удаляем все входящие ребра
            for u in self.adj_list:
                if u != vertex:
                    if self.weighted:
                        self.adj_list[u] = [
                            (v, w) for v, w in self.adj_list[u] if v != vertex
                        ]
                    else:
                        if vertex in self.adj_list[u]:
                            self.adj_list[u].remove(vertex)
                            self.num_edges -= 1
        else:
            # Для неориентированного графа
            degree = len(self.adj_list[vertex])
            self.num_edges -= degree

            # Удаляем вершину из списков соседей
            for neighbor in self.get_neighbors(vertex):
                if self.weighted:
                    self.adj_list[neighbor] = [
                        (v, w) for v, w in self.adj_list[neighbor] if v != vertex
                    ]
                else:
                    self.adj_list[neighbor].remove(vertex)

        # Удаляем вершину
        del self.adj_list[vertex]
        self.vertices.remove(vertex)

    def get_num_vertices(self):
        """
        Получение количества вершин.

        Returns:
            Количество вершин в графе

        Сложность: O(1)
        """
        return len(self.vertices)

    def get_num_edges(self):
        """
        Получение количества ребер.

        Returns:
            Количество ребер в графе

        Сложность: O(1)
        """
        return self.num_edges

    def __str__(self):
        """Строковое представление списка смежности."""
        result = []
        result.append(
            f"Список смежности ({len(self.vertices)} вершин, {self.num_edges} ребер):"
        )
        result.append(f"Ориентированный: {self.directed}, Взвешенный: {self.weighted}")

        for vertex in sorted(self.vertices):
            if self.weighted:
                neighbors_str = ", ".join(f"{v}({w})" for v, w in self.adj_list[vertex])
            else:
                neighbors_str = ", ".join(str(v) for v in self.adj_list[vertex])
            result.append(f"{vertex}: [{neighbors_str}]")

        return "\n".join(result)
