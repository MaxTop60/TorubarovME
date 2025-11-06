class Node:
    """Класс узла для связного списка.

    Attributes:
        value: Значение, хранящееся в узле.
        next: Ссылка на следующий узел в списке.
    """

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    """Реализация односвязного списка.

    Attributes:
        head: Первый элемент списка.
        tail: Последний элемент списка.
    """

    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self) -> bool:
        """Проверяет, пуст ли список.

        Returns:
            bool: True, если список пуст, иначе False.
        """
        return self.head is None

    def insert_at_start(self, value):
        """Добавляет новый узел в начало списка.

        Args:
            value: Значение для нового узла.
        """
        new_node = Node(value)

        if self.is_empty():
            self.head = new_node  # O(1)
            self.tail = new_node  # O(1)
        else:
            new_node.next = self.head  # O(1)
            self.head = new_node  # O(1)
        # Общая сложность: O(1)

    def insert_at_end(self, value):
        """Добавляет новый узел в конец списка.

        Args:
            value: Значение для нового узла.
        """
        new_node = Node(value)

        if self.is_empty():
            self.head = new_node  # O(1)
            self.tail = new_node  # O(1)
        else:
            self.tail.next = new_node  # O(n)
            self.tail = new_node  # O(1)
        # Общая сложность: O(n)

    def delete_from_start(self):
        """Удаляет первый элемент списка.

        Returns:
            any: Значение удалённого элемента или None, если список пуст.
        """
        if self.is_empty():
            return None  # O(1)
        else:
            deleted_value = self.head.value  # O(1)
            self.head = self.head.next  # O(1)
            return deleted_value
        # Общая сложность: O(1)

    def traversal(self):
        """Проходит по всем элементам списка и выводит их значения."""
        if self.is_empty():
            print("Список пуст")
            return

        current_node = self.head  # O(1)
        while current_node:  # O(n)
            print(current_node.value)  # O(1)
            current_node = current_node.next  # O(1)
        # Общая сложность: O(n)
