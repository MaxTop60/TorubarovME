def simple_hash(key, table_size):
    """
    Простая хеш-функция - сумма кодов символов
    """
    hash_value = 0  # O(1)
    for char in str(key):  # O(n) где n - длина ключа
        hash_value += ord(char)  # O(1)
    return hash_value % table_size  # O(1)
    # Общая сложность: O(n)
    # Качество: низкое (плохое распределение)


def polynomial_hash(key, table_size, base=31):
    """
    Полиномиальная хеш-функция
    """
    hash_value = 0  # O(1)
    for char in str(key):  # O(n)
        hash_value = (hash_value * base + ord(char)) % table_size  # O(1)
    return hash_value  # O(1)
    # Общая сложность: O(n)
    # Качество: высокое


def djb2_hash(key, table_size):
    """
    Хеш-функция DJB2
    """
    hash_value = 5381  # O(1)
    for char in str(key):  # O(n)
        hash_value = ((hash_value << 5) + hash_value) + ord(char)  # O(1)
    return hash_value % table_size  # O(1)
    # Общая сложность: O(n)
    # Качество: очень высокое


HASH_FUNCTIONS = {  # O(1)
    'simple': simple_hash,
    'polynomial': polynomial_hash,
    'djb2': djb2_hash
}
