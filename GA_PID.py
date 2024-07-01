from PID_Simulator.PID_Simulator import PID
from PID_Simulator.PID_Simulator import HeatingSystem
from PID_Simulator.PID_Simulator import simulate
import numpy as np
import matplotlib.pyplot as plt

def fitness_function(params, setpoint, heating_system, sample_time, simulation_time):
    Kp, Ki, Kd = params
    pid = PID(Kp, Ki, Kd, setpoint, sample_time)
    heating_system.current_temp = heating_system.ambient_temp

    error_sum = 0
    for t in range(0, simulation_time, sample_time):
        power_input = pid.update(heating_system.current_temp)
        current_temp = heating_system.update(power_input, sample_time)
        error = setpoint - current_temp

        # 限制误差值的范围，防止溢出
        error = np.clip(error, -1e6, 1e6)
        error_sum += error**2

    # 增加对大参数的惩罚，防止参数过大
    penalty = 0.1 * (Kp**2 + Ki**2 + Kd**2)

    return -(error_sum + penalty)  # 返回负的误差平方和加惩罚项



# 遗传算法参数
population_size = 100
generations = 500
mutation_rate = 0.1
crossover_rate = 0.7

# 参数范围
param_bounds = [(0.01, 5), (0, 0.5), (0, 0.5)]  # (Kp, Ki, Kd)的范围

# 初始化种群
population = np.random.rand(population_size, 3)  # 每个个体包含3个参数（Kp、Ki、Kd）
for i in range(3):
    population[:, i] = population[:, i] * (param_bounds[i][1] - param_bounds[i][0]) + param_bounds[i][0]

best_fitness_history = []

# 进行遗传算法
for generation in range(generations):
    fitness = []
    for individual in population:
        try:
            fitness_value = fitness_function(individual, setpoint=100, heating_system=HeatingSystem(20, 1000, 500),
                                             sample_time=1, simulation_time=500)
            if np.isnan(fitness_value):
                fitness_value = -np.inf
        except:
            fitness_value = -np.inf
        fitness.append(fitness_value)
    fitness = np.array(fitness)

    # 避免NaN和无穷值
    if np.any(np.isnan(fitness)) or np.any(np.isinf(fitness)):
        fitness[np.isnan(fitness) | np.isinf(fitness)] = -np.inf

    best_fitness = np.max(fitness)
    best_fitness_history.append(best_fitness)

    # 选择（轮盘赌选择）
    fitness_sum = np.sum(fitness)
    if fitness_sum == 0 or np.isinf(fitness_sum):
        probabilities = np.ones(population_size) / population_size
    else:
        probabilities = fitness / fitness_sum

    if np.any(np.isnan(probabilities)):
        probabilities = np.ones(population_size) / population_size

    selected_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
    selected_population = population[selected_indices]

    # 交叉
    next_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_population[i], selected_population[i + 1]
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, 3)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            child1, child2 = parent1, parent2
        next_population.extend([child1, child2])

    # 变异
    next_population = np.array(next_population)
    for individual in next_population:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, 3)
            individual[mutation_point] = np.random.rand() * (
                        param_bounds[mutation_point][1] - param_bounds[mutation_point][0]) + \
                                         param_bounds[mutation_point][0]
    population = next_population

# 找到最优参数
plt.plot(best_fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness Convergence Curve')
plt.show()


best_individual = population[np.argmax(fitness)]
best_Kp, best_Ki, best_Kd = best_individual
print(f"Best Kp: {best_Kp}, Best Ki: {best_Ki}, Best Kd: {best_Kd}")

pid = simulate(best_Kp, best_Ki, best_Kd)