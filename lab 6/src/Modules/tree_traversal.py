def in_order_recursive(node):
    """
    Рекурсивный in-order обход (левый-корень-правый)
    Выводит элементы в отсортированном порядке
    """
    if node:
        in_order_recursive(node.left)  # Рекурсивно обойти левое поддерево
        print(node.value, end=" ")  # Посетить корень
        in_order_recursive(node.right)  # Рекурсивно обойти правое поддерево


def pre_order_recursive(node):
    """
    Рекурсивный pre-order обход (корень-левый-правый)
    Полезен для копирования структуры дерева
    """
    if node:
        print(node.value, end=" ")  # Посетить корень
        pre_order_recursive(node.left)  # Рекурсивно обойти левое поддерево
        pre_order_recursive(node.right)  # Рекурсивно обойти правое поддерево


def post_order_recursive(node):
    """
    Рекурсивный post-order обход (левый-правый-корень)
    Полезен для удаления дерева
    """
    if node:
        post_order_recursive(node.left)  # Рекурсивно обойти левое поддерево
        post_order_recursive(node.right)  # Рекурсивно обойти правое поддерево
        print(node.value, end=" ")  # Посетить корень


def in_order_iterative(root):
    """
    Итеративный in-order обход с использованием стека
    Выводит элементы в отсортированном порядке
    """
    stack = []
    current = root

    while True:
        # Дойти до самого левого узла текущего поддерева
        if current is not None:
            stack.append(current)  # Добавить узел в стек
            current = current.left  # Перейти к левому потомку

        # Возвращение из пустого поддерева
        elif stack:
            current = stack.pop()  # Взять узел из стека
            print(current.value, end=" ")  # Обработать узел
            current = current.right  # Перейти к правому поддереву

        else:
            break  # Обход завершен
