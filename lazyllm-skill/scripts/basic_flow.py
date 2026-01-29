from lazyllm import pipeline, parallel, bind

# 示例 1: 基础 Pipeline
print("=" * 50)
print("示例 1: 基础 Pipeline")
print("=" * 50)

def f1(input):
    return input + 1

def f2(input):
    return input * 2

def f3(input):
    return input ** 2

# 创建 pipeline
p = pipeline(f1, f2, f3)
result = p(1)
print("输入: 1")
print(f"结果: {result}  ( ((1+1)*2)^2 )")
print()

# 示例 2: 使用 with 语句
print("=" * 50)
print("示例 2: 使用 with 语句")
print("=" * 50)

with pipeline() as p:
    p.step1 = f1
    p.step2 = f2
    p.step3 = f3

result = p(2)
print("输入: 2")
print(f"结果: {result}")
print()

# 示例 3: Parallel 并行执行
print("=" * 50)
print("示例 3: Parallel 并行执行")
print("=" * 50)

def task1(input):
    return input * 2

def task2(input):
    return input + 3

with parallel() as p:
    p.task_a = task1
    p.task_b = task2

result = p(1)
print("输入: 1")
print(f"结果: {result}  ( (1*2, 1+3) )")
print()

# 示例 4: Parallel sum 后处理
print("=" * 50)
print("示例 4: Parallel sum 后处理")
print("=" * 50)

with parallel().sum as p:
    p.task_a = task1
    p.task_b = task2

result = p(1)
print("输入: 1")
print(f"结果: {result}  ( (1*2) + (1+3) )")
print()

# 示例 5: Parallel asdict 后处理
print("=" * 50)
print("示例 5: Parallel asdict 后处理")
print("=" * 50)

with parallel().asdict as p:
    p.task_a = task1
    p.task_b = task2

result = p(1)
print("输入: 1")
print(f"结果: {result}")
print()

# 示例 6: 参数绑定
print("=" * 50)
print("示例 6: 参数绑定")
print("=" * 50)

def add(input, extra):
    return input + extra

def multiply(input):
    return input * 3

with pipeline() as p:
    p.step1 = add | bind(extra=p.input)
    p.step2 = multiply

result = p(5)
print("输入: 5")
print(f"结果: {result}  ( (5+5)*3 )")
print()

# 示例 7: 复杂 Pipeline + Parallel
print("=" * 50)
print("示例 7: 复杂 Pipeline + Parallel")
print("=" * 50)

def preprocess(input):
    return input.strip()

def process1(input):
    return input.upper()

def process2(input):
    return input.lower()

def combine(results):
    return f"UPPER: {results[0]}, LOWER: {results[1]}"

with pipeline() as ppl:
    ppl.preprocess = preprocess

    with parallel() as ppl.prl:
        ppl.prl.upper = process1
        ppl.prl.lower = process2

    ppl.combine = combine

result = ppl("  Hello World  ")
print("输入: '  Hello World  '")
print(f"结果: {result}")
print()
