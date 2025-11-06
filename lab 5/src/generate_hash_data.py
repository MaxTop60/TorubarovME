import random
import string


def generate_random_string(length=10):
    """
    Генерация случайной строки
    Сложность: O(length)
    """
    return ''.join(random.choices(
        string.ascii_letters + string.digits, k=length))  # O(length)


def generate_test_data(num_items=1000, key_length=10):
    """
    Генерация тестовых данных
    Сложность: O(num_items * key_length)
    """
    test_data = []  # O(1)
    for i in range(num_items):  # O(num_items) итераций
        key = generate_random_string(key_length)  # O(key_length)
        value = f"value_{i}"  # O(1)
        test_data.append((key, value))  # O(1)
    return test_data  # O(1)
    # Общая сложность: O(num_items * key_length)


def generate_collision_data(base_key, num_variants=100):
    """
    Генерация данных для тестирования коллизий
    Сложность: O(num_variants * len(base_key))
    """
    collision_data = []  # O(1)
    for i in range(num_variants):  # O(num_variants) итераций
        # Создаем варианты с небольшими изменениями
        variant = base_key + str(i)  # O(len(base_key))
        collision_data.append((variant, f"value_{i}"))  # O(1)
    return collision_data  # O(1)
