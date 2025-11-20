from modules.heap import MinHeap


def heapsort(array):
    """
    Сортировка кучей с использованием класса MinHeap
    Возвращает отсортированный массив в возрастающем порядке
    """
    heap = MinHeap()
    heap.build_heap(array)

    sorted_array = []
    while heap.tree:
        min_val = heap.extract()
        sorted_array.append(min_val)

    return sorted_array


def heapsort_inplace(array):
    """
    In-place сортировка кучей с использованием логики MinHeap
    Сортирует исходный массив в убывающем порядке
    Не использует дополнительную память
    """
    n = len(array)

    # Построение Min-Heap из массива (in-place)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_min_inplace(array, i, n)

    # Последовательное извлечение минимальных элементов
    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        _sift_down_min_inplace(array, 0, i)

    return array


def _sift_down_min_inplace(arr, index, size):
    """
    Погружение элемента для Min-Heap (in-place версия)
    Аналогично методу _sift_down из класса MinHeap
    """
    while True:
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        # Ищем наименьшего потомка
        if left < size and arr[left] < arr[smallest]:
            smallest = left
        if right < size and arr[right] < arr[smallest]:
            smallest = right

        # Если нашли меньшего потомка - меняем местами
        if smallest != index:
            arr[index], arr[smallest] = arr[smallest], arr[index]
            index = smallest
        else:
            break
