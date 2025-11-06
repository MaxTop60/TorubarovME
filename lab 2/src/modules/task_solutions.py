from collections import deque


# Задача 1
def is_balanced_brackets(s):
    stack = []
    brackets_map = {")": "(", "}": "{", "]": "["}

    for char in s:  # Обход строки
        if char in brackets_map.values():
            stack.append(char)  # Добавление открывающей скобки в стек
        elif char in brackets_map:
            # Проверка соответствия закрывающей скобки
            if not stack or stack[-1] != brackets_map[char]:
                return False
            stack.pop()  # Удаление соответствующей открывающей скобки

    return len(stack) == 0


# Задача 2
def printer(arr):
    queue = deque()

    for el in arr:
        queue.append(el)

    # Обход очереди
    for i in range(len(queue)):
        print(f"Обработка файла: {queue.popleft()}")
        print(f"Очередь: {queue}")


# Задача 3
def is_palindrome(s):
    dq = deque(s)

    while len(dq) > 1:
        # Сравнение крайних элементов и удаление
        if dq.popleft() != dq.pop():
            return False
    return True
