import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class QuadcopterDynamics:
    def __init__(self, dt, t_start, t_end,m=4.34,g=9.8,kx=16,kv=5.6,kR=8.81,komega=2.54,J=np.array([[0.0820, 0, 0],   [0, 0.0845, 0],   [0, 0, 0.1377]])):
        # 初始化参数
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end
        self.tspan = np.arange(t_start, t_end + dt, dt)
        self.x0 = np.array([0., 0., 0.])
        self.v0 = np.array([0., 0., 0.])
        self.R0 = np.eye(3)
        self.omega0 = np.array([0., 0., 0.])
        self.initial_conditions = np.concatenate([self.x0, self.v0, self.R0.ravel(), self.omega0])
        self.omegad_diff = np.zeros((len(self.tspan), 3))
        self.domegad_diff = np.zeros((len(self.tspan), 3))
        self.bd = np.array([np.cos(np.pi * self.tspan), np.sin(np.pi * self.tspan), np.zeros(len(self.tspan))]).T
        self.dxds = np.zeros((len(self.tspan), 3))
        self.ddxds = np.zeros((len(self.tspan), 3))
        self.sol = np.zeros((len(self.tspan), len(self.initial_conditions)))
        self.omegads = np.zeros((len(self.tspan), 3))
        self.J = J
        self.m = m
        self.g = g
        self.kx = kx*m
        self.kv = kv*m
        self.kR = kR
        self.komega = komega
        self.exs = np.zeros((len(self.tspan), 3))
        self.evs = np.zeros((len(self.tspan), 3))



    def dynamics(self, t, i):

        # 状态变量解包
        x = self.x0
        v = self.v0
        R = self.R0
        omega = self.omega0
        pi = np.pi

        # 设置系统参数
        J = self.J
        m = self.m
        g = self.g
        kx = self.kx
        kv = self.kv
        kR = self.kR
        komega = self.komega
        e3 = np.array([0., 0., 1.])


        # 参考轨迹
        xd = np.array([0.4 * t, 0.4 * np.sin(np.pi * t), 0.6 * np.cos(np.pi * t)])
        dxd = np.array([0.4, 0.4 * pi * np.cos(pi * t), -0.6 * pi * np.sin(pi * t)])
        ddxd = np.array([0, -0.4 * (pi ** 2) * np.sin(pi * t), -0.6 * (pi ** 2) * np.cos(pi * t)])

        # 控制律设计
        # 计算速度差分
        # 计算期望速度和加速度


        # 误差指定
        ex = x - xd
        ev = v - dxd

        # 计算角度差分
        Rd = self.b2R(self.bd[i], ddxd, ex, ev, e3)
        if i == 0:
            Rd_dot = np.zeros((3, 3))
            omegad = np.array([0,0,1])
            domegad = np.zeros(3)
        else:
            Rd_last = self.b2R(self.bd[i - 1],self.ddxds[i - 1],self.exs[i - 1],self.evs[i - 1],e3)
            Rd_dot = self.diff(Rd, Rd_last, self.dt)
            omegad = self.vee(Rd.T @ Rd_dot)
            domegad = self.diff(omegad, self.omegads[i - 1], self.dt)


        # 仿真存值
        self.omegads[i] = omegad
        self.exs[i] = ex
        self.evs[i] = ev
        self.ddxds[i] = ddxd

        # 计算姿态误差
        eR = 0.5 * self.vee(Rd.T @ R - R.T @ Rd)
        eomega = omega - R.T @ Rd @ omegad

        # 计算控制力
        # f = np.dot((-kx * ex - kv * ev - m * g*e3 + m * ddxd),R.T[2])
        f = -(-kx * ex - kv * ev - m * g * e3 + m * ddxd)@R.T[2]

        # 计算控制力矩
        M = -kR * eR - komega * eomega + np.cross(omega, J @ omega) - J @ (self.skew(omega) @ R.T @ Rd @ omegad - R.T @ Rd @ domegad)

        # 状态方程
        dx = v
        # temp =R.T[2]
        # temp2 =R*e3
        ddx = g*e3 - f * R.T[2] / m
        dR = R @ self.skew(omega)
        domega = np.linalg.inv(J) @ (M - np.cross(omega, J @ omega))



        # 合并状态导数
        return dx, ddx, dR, domega

    def simulate(self):
        for i, t in enumerate(self.tspan):
            dx, ddx, dR, domega = self.dynamics(t, i)
            self.x0 += self.dt * dx
            self.v0 += self.dt * ddx
            self.R0 += self.dt * dR
            self.omega0 += self.dt * domega

            # 标准化旋转矩阵
            U, S, V = np.linalg.svd(self.R0)
            self.R0 = U @ V

            self.initial_conditions = np.concatenate([self.x0, self.v0, self.R0.ravel(), self.omega0])
            self.sol[i, :] = self.initial_conditions

    # 画图函数
    def plot(self):
        # 绘制位移x, y, z
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.tspan, self.sol[:, 0], label='Position x')
        plt.plot(self.tspan, 0.4 * self.tspan, label='Position xd')
        plt.xlabel('Time [s]')
        plt.ylabel('Position x [m]')
        plt.title('Position x vs Time')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.tspan, self.sol[:, 1], label='Position y')
        plt.plot(self.tspan, 0.4 * np.sin(np.pi*self.tspan), label='Position yd')
        plt.xlabel('Time [s]')
        plt.ylabel('Position y [m]')
        plt.title('Position y vs Time')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.tspan, self.sol[:, 2], label='Position z')
        plt.plot(self.tspan, 0.6 * np.cos(np.pi * self.tspan), label='Position zd')
        plt.xlabel('Time [s]')
        plt.ylabel('Position z [m]')
        plt.title('Position z vs Time')
        plt.legend()

        # 绘制角速度ωx, ωy, ωz
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.tspan, self.sol[:, 15], label='Angular Velocity ωx')
        plt.plot(self.tspan, self.omegads[:, 0], label='Angular Velocity ωxd')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity ωx [rad/s]')
        plt.title('Angular Velocity ωx vs Time')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.tspan, self.sol[:, 16], label='Angular Velocity ωy')
        plt.plot(self.tspan, self.omegads[:, 1], label='Angular Velocity ωyd')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity ωy [rad/s]')
        plt.title('Angular Velocity ωy vs Time')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.tspan, self.sol[:, 17], label='Angular Velocity ωz')
        plt.plot(self.tspan, self.omegads[:, 2], label='Angular Velocity ωzd')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity ωz [rad/s]')
        plt.title('Angular Velocity ωz vs Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 将反对称矩阵转换为向量
    def vee(self,matrix):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("输入必须是一个方阵。")
        # if not np.allclose(matrix, -matrix.T):
        #     raise ValueError("输入矩阵不是反对称矩阵。")
        vector = matrix[np.tril_indices(matrix.shape[0], -1)]
        return np.array([vector[2], -vector[1], vector[0]])


    # 将向量转换为反对称矩阵
    def skew(self,v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])


    def diff(self,x_now,x_last,dt):
        return (x_now-x_last)/dt

    # def calculate_xd(self,t):
    #     return np.array([0.4 * t, 0.4 * np.sin(np.pi * t), 0.6 * np.cos(np.pi * t)])

    def b2R(self,b1,ddxd,ex,ev,e3):
        kx = self.kx
        kv = self.kv
        m = self.m
        b3 = (-kx * ex - kv * ev - m * self.g*e3 + m * ddxd)/np.linalg.norm((-kx * ex - kv * ev - m * self.g*e3 + m * ddxd))
        b2 = np.cross(b3,b1)/np.linalg.norm(np.cross(b3,b1))
        R = np.zeros((3,3))
        R[:,0] = np.cross(b2,b3)
        R[:,1] = b2
        R[:,2] = b3


        return R


if __name__ == "__main__":
    quadcopter = QuadcopterDynamics(0.01, 0, 10)
    quadcopter.simulate()
    quadcopter.plot()
