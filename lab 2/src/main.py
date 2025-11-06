import modules.perfomance_analysis as pa
import modules.task_solutions as ts

if __name__ == "__main__":
    # Анализ производительности
    sizes = [100, 1000, 10000, 50000]
    pa.comparision(sizes)

    # Скобки
    print(ts.is_balanced_brackets("{[()]}"))
    print("")

    # Принтер
    orders = {"Документ 1", "Документ 2", "Документ 3"}
    ts.printer(orders)
    print()

    # Палиндром
    print(ts.is_palindrome("12332"))
