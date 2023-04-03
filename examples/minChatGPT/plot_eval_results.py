import matplotlib.pyplot as plt
import json


files = [1300]


x = [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1100, 1200, 1300]
y = []

for index in x:
    path = f"/home/edward/work/trl/runs/run_128_8_8_False_4_0/eval_results_step_{index}.json"
    with open(path) as fp:
        data = json.load(fp)

    y.append(sum(data) / len(data))

plt.plot(x, y, marker="o")
plt.title("llama-se RL finetune results with gpt-xl RM")
plt.grid()
plt.xlabel("step")
plt.ylabel("eval reward")
plt.show()
