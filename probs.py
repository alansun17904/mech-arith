import random


def generate_carry_kth_digit_problems(n, k, count):
    problems = []
    for _ in range(count):
        while True:
            a = random.randint(10 ** (n - 1), 10**n - 1)
            b = random.randint(10 ** (n - 1), 10**n - 1)
            a_str, b_str = str(a), str(b)
            if k == 0:
                no_carry_others = all(
                    (int(a_str[i]) + int(b_str[i]) < 10) for i in range(n)
                )
                if no_carry_others:
                    problems.append(f"{a} + {b} = ")
                    break
                else:
                    continue

            # Ensure carry in the k-th digit place
            carry_kth = (int(a_str[-k]) + int(b_str[-k])) >= 10

            # Ensure no carry in other places if k != 1
            no_carry_others = all(
                (int(a_str[i]) + int(b_str[i]) + 1 < 10) for i in range(n) if i != n - k
            )

            if carry_kth and no_carry_others:
                problems.append(f"{a} + {b} = ")
                break
    return problems


if __name__ == "__main__":
    for x in range(1, 7):
        for y in range(0, x + 1):
            problems = generate_carry_kth_digit_problems(x, y, 1000)
            f = open(f"{x}digit-{y}carry-problems+new.txt", "w")
            f.write("\n".join(problems))
