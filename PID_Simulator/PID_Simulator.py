import matplotlib.pyplot as plt
import numpy as np


class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, sample_time):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._last_error = 0.0
        self._integral = 0.0

    def update(self, current_value):
        error = self.setpoint - current_value
        self._integral += error * self.sample_time
        derivative = (error - self._last_error) / self.sample_time

        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        self._last_error = error

        return output


class HeatingSystem:
    def __init__(self, ambient_temp, heating_power, thermal_mass):
        self.ambient_temp = ambient_temp
        self.heating_power = heating_power
        self.thermal_mass = thermal_mass
        self.current_temp = ambient_temp

    def update(self, power_input, time_step):
        power_input = np.clip(power_input, -1e6, 1e6)
        temp_change = power_input * self.heating_power / self.thermal_mass * time_step
        self.current_temp += temp_change
        return self.current_temp


def simulate(Kp,Ki,Kd):
    # 仿真参数
    setpoint = 100  # 目标温度
    initial_temp = 20  # 初始温度
    ambient_temp = 20  # 环境温度
    heating_power = 100  # 加热功率
    thermal_mass = 500  # 热容
    sample_time = 1  # 采样时间
    simulation_time = 100  # 仿真时间

    # PID参数
    # Kp = 0.47
    # Ki = 0.86
    # Kd = 0.87

    # 创建PID控制器和加热系统
    pid = PID(Kp, Ki, Kd, setpoint, sample_time)
    heating_system = HeatingSystem(ambient_temp, heating_power, thermal_mass)

    # 存储仿真结果
    times = []
    temperatures = []

    # 进行仿真
    for t in range(0, simulation_time, sample_time):
        power_input = pid.update(heating_system.current_temp)
        current_temp = heating_system.update(power_input, sample_time)

        times.append(t)
        temperatures.append(current_temp)

    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(times, temperatures, label="实际温度")
    plt.axhline(y=setpoint, color='r', linestyle='--', label="设定温度")
    plt.xlabel('时间 (s)')
    plt.ylabel('温度 (C)')
    plt.legend()
    plt.savefig("PID_Simulator.png")
    plt.show()

    settling_time = None
    overshoot = None
    rise_time = None
    steady_state_error = abs(100 - temperatures[-1])

    # 计算上升时间
    for t, temp in zip(times, temperatures):
        if temp >= 100:
            rise_time = t
            break

    # 计算过冲
    overshoot = max(temperatures) - 100

    # 计算稳定时间（误差小于2%设定值）
    for t, temp in reversed(list(zip(times, temperatures))):
        if abs(temp - 100) > 2:
            settling_time = t
            break

    settling_time = times[-1] - settling_time

    print(f"Rise Time: {rise_time} s")
    print(f"Overshoot: {overshoot} C")
    print(f"Settling Time: {settling_time} s")
    print(f"Steady State Error: {steady_state_error} C")

if __name__ == '__main__':
    simulate(0.47,0.86,0.87)
