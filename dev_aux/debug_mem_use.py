import sys
from typing import NamedTuple


class Allocation(NamedTuple):
    memory_size: int
    op_name: str


def main():
    f = open(sys.argv[1], "r")

    sum_memory = 0
    replica1_memory = 0
    usages = []
    for line in f:
        if "tensorflow/core/common_runtime/bfc_allocator.cc:1046" not in line:
            continue
        tokens = line.split()
        memory_size = int(tokens[9])
        sum_memory += memory_size
        op_name = tokens[12]
        if op_name.startswith("replica_1"):
            replica1_memory += memory_size
        action_count = tokens[14]
        step = tokens[16]
        next_ = tokens[18]

        alloc = Allocation(memory_size, op_name)
        usages.append(alloc)

    usages.sort(key=lambda x: x.memory_size, reverse=True)
    for alloc in usages[:100]:
        print(alloc)

    MB = 1024 * 1024
    print("total memory {} MB".format(sum_memory / MB))
    print("replica_1 memory {} MB".format(replica1_memory / MB))


if __name__ == "__main__":
    main()