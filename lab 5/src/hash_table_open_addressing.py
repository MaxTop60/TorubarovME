from hash_functions import HASH_FUNCTIONS


class HashTableOpenAddressing:
    """
    Хеш-таблица с открытой адресацией
    """

    def __init__(
        self,
        size=10,
        hash_function="polynomial",
        probing_method="linear",
        load_factor_threshold=0.7,
    ):
        self.size = size  # O(1)
        self.hash_function = HASH_FUNCTIONS[hash_function]  # O(1)
        self.probing_method = probing_method  # O(1)
        self.load_factor_threshold = load_factor_threshold  # O(1)
        self.table = [None] * size  # O(size)
        self.count = 0  # O(1)
        self.DELETED = object()  # O(1)
        # Общая сложность инициализации: O(size)

    def _hash(self, key, attempt=0):
        """
        Вычисляет хеш с учетом номера попытки
        Сложность: O(len(key))
        """
        if self.probing_method == "linear":  # O(1)
            return (
                self.hash_function(key, self.size) + attempt
            ) % self.size  # O(len(key))
        elif self.probing_method == "double":  # O(1)
            h1 = self.hash_function(key, self.size)  # O(len(key))
            h2 = 1 + (self.hash_function(key, self.size - 1))  # O(len(key))
            return (h1 + attempt * h2) % self.size  # O(1)
        # Общая сложность: O(len(key))

    def _resize(self, new_size):
        """
        Увеличивает размер таблицы
        Сложность: O(n) где n - количество элементов
        """
        old_table = self.table  # O(1)
        self.size = new_size  # O(1)
        self.table = [None] * new_size  # O(new_size)
        self.count = 0  # O(1)

        for item in old_table:  # O(old_size) итераций
            if item is not None and item is not self.DELETED:  # O(1)
                key, value = item  # O(1)
                # Используем внутренний метод вставки без ресайза
                self._insert_without_resize(key, value)  # O(1/(1-α))
        # Общая сложность: O(n)

    def _insert_without_resize(self, key, value):
        """
        Внутренний метод вставки без проверки ресайза
        Сложность: O(1/(1-α)) в среднем
        """
        attempt = 0  # O(1)
        while attempt < self.size:  # O(1/(1-α)) итераций в среднем
            index = self._hash(key, attempt)  # O(len(key))

            if self.table[index] is None or self.table[index] is self.DELETED:
                self.table[index] = (key, value)  # O(1)
                self.count += 1  # O(1)
                return True  # O(1)
            elif self.table[index][0] == key:  # O(1)
                self.table[index] = (key, value)  # O(1)
                return True  # O(1)

            attempt += 1  # O(1)

        return False  # O(1) - не удалось вставить

    def insert(self, key, value):
        """
        Вставка элемента в хеш-таблицу
        Средняя сложность: O(1/(1-α)) где α - коэффициент заполнения
        Худшая сложность: O(n)
        """
        # Проверяем необходимость ресайза перед вставкой
        if self.load_factor >= self.load_factor_threshold:  # O(1)
            self._resize(self.size * 2)  # O(n)

        attempt = 0  # O(1)
        while attempt < self.size:  # O(1/(1-α)) итераций в среднем
            index = self._hash(key, attempt)  # O(len(key))

            if self.table[index] is None or self.table[index] is self.DELETED:
                self.table[index] = (key, value)  # O(1)
                self.count += 1  # O(1)
                return  # O(1)
            elif self.table[index][0] == key:  # O(1)
                self.table[index] = (key, value)  # O(1)
                return  # O(1)

            attempt += 1  # O(1)

        # Если таблица полная, увеличиваем размер и пробуем снова
        self._resize(self.size * 2)  # O(n)
        self.insert(key, value)  # O(1) рекурсивный вызов

    def search(self, key):
        """
        Поиск элемента в хеш-таблице
        Средняя сложность: O(1/(1-α))
        Худшая сложность: O(n)
        """
        attempt = 0  # O(1)
        while attempt < self.size:  # O(1/(1-α)) итераций в среднем
            index = self._hash(key, attempt)  # O(len(key))

            if self.table[index] is None:  # O(1)
                return None  # O(1)
            elif (
                self.table[index] is not self.DELETED and self.table[
                    index][0] == key
            ):  # O(1)
                return self.table[index][1]  # O(1)

            attempt += 1  # O(1)

        return None  # O(1)
        # Средняя сложность: O(1/(1-α))
        # Худшая сложность: O(n)

    def delete(self, key):
        """
        Удаление элемента из хеш-таблицы
        Средняя сложность: O(1/(1-α))
        Худшая сложность: O(n)
        """
        attempt = 0  # O(1)
        while attempt < self.size:  # O(1/(1-α)) итераций в среднем
            index = self._hash(key, attempt)  # O(len(key))

            if self.table[index] is None:  # O(1)
                return False  # O(1)
            elif (
                self.table[index] is not self.DELETED and self.table[
                    index][0] == key
            ):  # O(1)
                self.table[index] = self.DELETED  # O(1)
                self.count -= 1  # O(1)
                return True  # O(1)

            attempt += 1  # O(1)

        return False  # O(1)
        # Средняя сложность: O(1/(1-α))
        # Худшая сложность: O(n)

    @property
    def load_factor(self):
        """Коэффициент заполнения - O(1)"""
        return self.count / self.size  # O(1)

    def get_collision_stats(self):
        """
        Статистика коллизий и пробирований
        Сложность: O(size)
        """
        total_probes = 0  # O(1)
        max_probes = 0  # O(1)
        occupied_cells = 0  # O(1)

        for i in range(self.size):  # O(size) итераций
            if self.table[i] is not None and self.table[i] is not self.DELETED:
                occupied_cells += 1  # O(1)
                key, value = self.table[i]  # O(1)
                # Измеряем количество пробирований для поиска этого элемента
                probes = self._measure_probes(key)  # O(1/(1-α))
                total_probes += probes  # O(1)
                max_probes = max(max_probes, probes)  # O(1)

        avg_probes = total_probes / occupied_cells if occupied_cells > 0 else 0

        return {  # O(1)
            "average_probes": avg_probes,
            "max_probes": max_probes,
            "load_factor": self.load_factor,
            "occupied_cells": occupied_cells,
        }

    def _measure_probes(self, key):
        """
        Измеряет количество пробирований для поиска ключа
        Сложность: O(1/(1-α)) в среднем
        """
        attempt = 0  # O(1)
        while attempt < self.size:  # O(1/(1-α)) итераций
            index = self._hash(key, attempt)  # O(len(key))

            if self.table[index] is None:  # O(1)
                break  # O(1)
            elif (
                self.table[index] is not self.DELETED and self.table[
                    index][0] == key
            ):  # O(1)
                return attempt + 1  # O(1)

            attempt += 1  # O(1)

        return attempt + 1  # O(1)
