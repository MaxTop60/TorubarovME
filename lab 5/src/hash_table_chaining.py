from hash_functions import HASH_FUNCTIONS


class HashTableChaining:
    """
    Хеш-таблица с методом цепочек
    """

    def __init__(
            self, size=10,
            hash_function='polynomial', load_factor_threshold=0.7):
        self.size = size  # O(1)
        self.hash_function = HASH_FUNCTIONS[hash_function]  # O(1)
        self.load_factor_threshold = load_factor_threshold  # O(1)
        self.table = [[] for _ in range(size)]  # O(size)
        self.count = 0  # O(1)
        # Общая сложность инициализации: O(size)

    def _hash(self, key):
        return self.hash_function(key, self.size)  # O(len(key))

    def _resize(self, new_size):
        """
        Увеличивает размер таблицы и перехеширует все элементы
        Сложность: O(n) где n - количество элементов
        """
        old_table = self.table  # O(1)
        self.size = new_size  # O(1)
        self.table = [[] for _ in range(new_size)]  # O(new_size)
        self.count = 0  # O(1)

        for bucket in old_table:  # O(old_size) итераций
            for key, value in bucket:  # O(длина цепочки) итераций
                self.insert(key, value)  # O(1) в среднем
        # Общая сложность: O(n)

    def insert(self, key, value):
        """
        Вставка элемента в хеш-таблицу
        Средняя сложность: O(1)
        Худшая сложность: O(n)
        """
        if self.load_factor > self.load_factor_threshold:  # O(1)
            self._resize(self.size * 2)  # O(n) в худшем случае

        index = self._hash(key)  # O(len(key))
        bucket = self.table[index]  # O(1)

        # Проверяем, нет ли уже такого ключа - O(длина цепочки)
        for i, (k, v) in enumerate(bucket):
            # O(α) где α - коэффициент заполнения
            if k == key:  # O(1)
                bucket[i] = (key, value)  # O(1)
                return  # O(1)

        bucket.append((key, value))  # O(1)
        self.count += 1  # O(1)
        # Средняя сложность: O(1)
        # Худшая сложность: O(n)

    def search(self, key):
        """
        Поиск элемента в хеш-таблице
        Средняя сложность: O(1)
        Худшая сложность: O(n)
        """
        index = self._hash(key)  # O(len(key))
        bucket = self.table[index]  # O(1)

        for k, v in bucket:  # O(α) итераций
            if k == key:  # O(1)
                return v  # O(1)
        return None  # O(1)
        # Средняя сложность: O(1)
        # Худшая сложность: O(n)

    def delete(self, key):
        """
        Удаление элемента из хеш-таблицы
        Средняя сложность: O(1)
        Худшая сложность: O(n)
        """
        index = self._hash(key)  # O(len(key))
        bucket = self.table[index]  # O(1)

        for i, (k, v) in enumerate(bucket):  # O(α) итераций
            if k == key:  # O(1)
                del bucket[i]  # O(длина цепочки)
                self.count -= 1  # O(1)
                return True  # O(1)
        return False  # O(1)
        # Средняя сложность: O(1)
        # Худшая сложность: O(n)

    @property
    def load_factor(self):
        """Коэффициент заполнения - O(1)"""
        return self.count / self.size  # O(1)

    def get_collision_stats(self):
        """
        Статистика коллизий
        Сложность: O(size)
        """
        collisions = 0  # O(1)
        max_chain_length = 0  # O(1)
        empty_buckets = 0  # O(1)

        for bucket in self.table:  # O(size) итераций
            if len(bucket) == 0:  # O(1)
                empty_buckets += 1  # O(1)
            elif len(bucket) > 1:  # O(1)
                collisions += len(bucket) - 1  # O(1)
            max_chain_length = max(max_chain_length, len(bucket))  # O(1)

        return {  # O(1)
            'total_collisions': collisions,
            'max_chain_length': max_chain_length,
            'empty_buckets': empty_buckets,
            'load_factor': self.load_factor
        }
