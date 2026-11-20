"""
apriori.py - поиск частых наборов объектов алгоритмом Apriori.

Файл содержит:
    load_transactions(path, encoding)          - чтение корзин из CSV
    apriori(transactions, min_support)         - сам алгоритм
    sort_itemsets(result, order)               - упорядочивание результата
    run_experiments(transactions, thresholds)  - серия запусков с замером времени
    main()                                     - запуск из командной строки

Запуск:  python apriori.py data/baskets.csv 0.05 support
"""
import csv
import sys
import time
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple

Itemset = Tuple[str, ...]          # набор товаров: отсортированный кортеж
Result = Dict[Itemset, float]      # набор -> поддержка (доля корзин, 0..1)


def load_transactions(path: str, encoding: str = "cp1251") -> List[FrozenSet[str]]:
    """Читает файл корзин.

    Вход:  path     - путь к CSV (одна строка = одна корзина, товары через запятую,
                      заголовка нет);
           encoding - кодировка файла (baskets.csv - cp1251).
    Выход: список корзин; каждая корзина - frozenset (дубликаты внутри корзины
           убираются, пустые значения отбрасываются).
    """
    transactions = []
    with open(path, encoding=encoding, newline="") as f:
        for row in csv.reader(f):
            basket = frozenset(x.strip() for x in row if x.strip())
            if basket:
                transactions.append(basket)
    return transactions


def apriori(transactions: Sequence[FrozenSet[str]], min_support: float) -> Result:
    """Находит все частые наборы алгоритмом Apriori.

    Вход:  transactions - список корзин (множества товаров);
           min_support  - порог поддержки как доля от 0 до 1 (5% -> 0.05).
    Выход: словарь {набор (отсортированный кортеж): поддержка (доля)}
           для всех наборов с поддержкой >= min_support.

    Идея: поддержка не растёт при добавлении товара (антимонотонность), поэтому
    набор длины k может быть частым, только если все его подмножества длины k-1
    частые. Уровни строятся по возрастанию k:
      1) частые 1-наборы;  2) join - склейка пар (k-1)-наборов с общим префиксом;
      3) prune - отбрасывание кандидатов с редким подмножеством;
      4) подсчёт поддержки оставшихся кандидатов; повтор, пока есть частые.
    Реализация подсчёта: для каждого товара хранится битовая маска корзин,
    где он есть; поддержка кандидата = число единиц в побитовом И масок его
    товаров (вертикальное представление, как в Eclat/Apriori-TID).
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support должен быть в диапазоне (0, 1]")
    n = len(transactions)
    if n == 0:
        return {}
    min_count = -(-min_support * n // 1)          # ceil: минимальное число корзин
    min_count = int(min_count)

    # битовые маски корзин для каждого товара
    masks: Dict[str, int] = {}
    for tid, basket in enumerate(transactions):
        bit = 1 << tid
        for item in basket:
            masks[item] = masks.get(item, 0) | bit

    result: Result = {}
    level: Dict[Itemset, int] = {}                # текущие частые наборы -> маска
    for item, m in masks.items():
        c = m.bit_count()
        if c >= min_count:
            level[(item,)] = m
            result[(item,)] = c / n

    k = 2
    while level:
        prev = sorted(level)                      # лексикографический порядок
        prev_set = set(prev)
        nxt: Dict[Itemset, int] = {}
        for i, a in enumerate(prev):
            for b in prev[i + 1:]:
                if a[:-1] != b[:-1]:              # префиксы разошлись - дальше не склеить
                    break
                cand = a + (b[-1],)               # join
                if all(sub in prev_set for sub in combinations(cand, k - 1)):  # prune
                    m = level[a] & masks[b[-1]]
                    c = m.bit_count()
                    if c >= min_count:
                        nxt[cand] = m
                        result[cand] = c / n
        level = nxt
        k += 1
    return result


def sort_itemsets(result: Result, order: str = "support") -> List[Tuple[Itemset, float]]:
    """Упорядочивает результат.

    Вход:  result - словарь из apriori();
           order  - "support" (по убыванию поддержки, при равенстве по длине
                    и лексикографически) или "lex" (лексикографически).
    Выход: список пар (набор, поддержка).
    """
    if order == "support":
        return sorted(result.items(), key=lambda p: (-p[1], len(p[0]), p[0]))
    if order == "lex":
        return sorted(result.items(), key=lambda p: p[0])
    raise ValueError("order должен быть 'support' или 'lex'")


def count_by_length(result: Result) -> Dict[int, int]:
    """Вход: результат apriori(). Выход: {длина набора: количество наборов}."""
    counts: Dict[int, int] = {}
    for itemset in result:
        counts[len(itemset)] = counts.get(len(itemset), 0) + 1
    return dict(sorted(counts.items()))


def run_experiments(transactions, thresholds, repeats: int = 5):
    """Запускает apriori для каждого порога и измеряет время.

    Вход:  transactions - корзины; thresholds - пороги в долях (0.01, ...);
           repeats - число повторов (берётся медиана времени).
    Выход: список словарей {threshold, seconds, total, by_length, result}.
    """
    out = []
    for t in thresholds:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = apriori(transactions, t)
            times.append(time.perf_counter() - t0)
        times.sort()
        out.append({"threshold": t, "seconds": times[len(times) // 2],
                    "total": len(res), "by_length": count_by_length(res), "result": res})
    return out


def main(argv=None) -> None:
    """Командная строка: apriori.py <файл> <порог 0..1> [support|lex] [кодировка]."""
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print(main.__doc__)
        return
    data = load_transactions(argv[0], argv[3] if len(argv) > 3 else "cp1251")
    res = apriori(data, float(argv[1]))
    for itemset, sup in sort_itemsets(res, argv[2] if len(argv) > 2 else "support"):
        print(f"{sup * 100:6.2f}%  {{{', '.join(itemset)}}}")
    print(f"Всего частых наборов: {len(res)}")


if __name__ == "__main__":
    main()
