import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义旋转函数
def rotate(theta, r):
    Um = np.array([[np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0], 
                   [0, 0, 1]])
    inv_Um = np.linalg.inv(Um)
    U = np.array([[np.cos(np.pi), np.sin(np.pi), 0], 
                  [-np.sin(np.pi), np.cos(np.pi), 0], 
                  [0, 0, 1]])
    inv_U = np.linalg.inv(U)
    n = int(2 * np.pi / theta)
    
    if r.size == 3:
        R = np.zeros((3, n))
        RB = np.zeros((3, n))
    else:
        R = np.zeros((3, 3, n))
        RB = np.zeros((3, 3, n))
    
    for i in range(n):
        if r.size == 3:
            r = np.dot(inv_Um, r)
            rb = -r
            R[:, i] = r.ravel()
            RB[:, i] = rb.ravel()
        else:
            r = np.dot(np.dot(inv_Um, r), Um)
            rb = np.dot(np.dot(inv_U, r), U)
            R[:, :, i] = r
            RB[:, :, i] = rb
    return R, RB

# 定义刚度矩阵旋转函数
def K(theta, k):
    U = np.array([[np.cos(theta), np.sin(theta), 0], 
                  [-np.sin(theta), np.cos(theta), 0], 
                  [0, 0, 1]])
    inv_U = np.linalg.inv(U)
    k = np.dot(np.dot(inv_U, k), U)
    return k

# 常量定义
a = 1.42e-10  # 晶格参数（米）
m = 1.99e-26  # 质量（千克）
pi = np.pi
h_bar = 1.0546e-34  # 普朗克常数（Js）
kB = 1.38065e-23  # 玻尔兹曼常数（J/K）
n = 100  # k空间的分割数
f = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])*10

# 原子位置
A_I, B_I = rotate(2 / 3 * np.pi, np.array([[1, 0, 0]]).T * a)
A_II, B_II = rotate(1 / 3 * np.pi, np.array([[3 / 2, np.sqrt(3) / 2, 0]]).T * a)
A_III, B_III = rotate(2 / 3 * np.pi, np.array([[1, np.sqrt(3), 0]]).T * a)
A_IV1, B_IV1 = rotate(2 / 3 * np.pi, np.array([[2.5, np.sqrt(3) / 2, 0]]).T * a)
A_IV2, B_IV2 = rotate(2 / 3 * np.pi, np.array([[2.5, -np.sqrt(3) / 2, 0]]).T * a)
# A_V, B_V = rotate(1 / 3 * np.pi, np.array([[3, 0, 0]]).T * a)

A_IV = np.concatenate((A_IV1, A_IV2), axis=1)
B_IV = np.concatenate((B_IV1, B_IV2), axis=1)


[KAB1, KBA1] = rotate(2 / 3 * pi, f[:3, :3])
[KAA, KBB] = rotate(1 / 3 * pi, K(1 / 6 * pi, f[3:6, 3:6]))
[KAB3, KBA3] = rotate(2 / 3 * pi, K(1 / 3 * pi, f[6:9, 6:9]))
[KAB4f, KBA4f] = rotate(2 / 3 * pi, K(0.333473, f[9:12, 9:12]))
[KAB4s, KBA4s] = rotate(2 / 3 * pi, K(0.333473, f[9:12, 9:12]))
# [KAA5, KBB5] = rotate(1 / 3 * pi, f[12:15, 12:15])
KAB4 = np.concatenate((KAB4f, KAB4s), axis=2)
KBA4 = np.concatenate((KBA4f, KBA4s), axis=2)

# 生成k点路径
dlist = 1000
k_max = np.pi/3/a
klist = np.linspace(0, k_max, dlist)

N = 8
len_omega = 12 * N
omega = np.zeros((len_omega, dlist))

for i in range(len(klist)):
    kx = klist[i]
    D = np.zeros((len_omega, len_omega), dtype=complex) 
    D00 = np.zeros((3, 3))
    Dxx = np.sum(KBA1, axis=2) + np.sum(KBB, axis=2) + np.sum(KBA3, axis=2) + np.sum(KBA4, axis=2)

    # Atom 9
    D95 = np.dot(KBB[:, :, 3], np.exp(1j * kx * 0))
    D96 = np.dot(KBA4[:, :, 1], np.exp(1j * kx * (-1/2*a)))
    D98 = np.dot(KBA3[:, :, 1], np.exp(1j * kx * a)) + np.dot(KBA4[:, :, 4], np.exp(1j * kx * (-2*a)))
    D910 = np.dot(KBA1[:, :, 1], np.exp(1j * kx * (-1/2*a))) + np.dot(KBA4[:, :, 5], np.exp(1j * kx * (5/2*a)))
    D911 = np.dot(KBB[:, :, 2], np.exp(1j * kx * (-3/2*a))) + np.dot(KBB[:, :, 4], np.exp(1j * kx * (3/2*a)))
    D912 = np.dot(KBA1[:, :, 2], np.exp(1j * kx * a)) + np.dot(KBA3[:, :, 0], np.exp(1j * kx * (-2*a)))
    D913 = np.dot(KBB[:, :, 0], np.exp(1j * kx * 0))
    D914 = np.dot(KBA1[:, :, 0], np.exp(1j * kx * (-1/2*a))) + np.dot(KBA4[:, :, 2], np.exp(1j * kx * (5/2*a)))
    D915 = np.dot(KBB[:, :, 1], np.exp(1j * kx * (-3/2*a))) + np.dot(KBB[:, :, 5], np.exp(1j * kx * (3/2*a)))
    D916 = np.dot(KBA3[:, :, 2], np.exp(1j * kx * a)) + np.dot(KBA4[:, :, 0], np.exp(1j * kx * (-2*a)))
    D918 = np.dot(KBA4[:, :, 3], np.exp(1j * kx * (-1/2*a)))
    DA = np.hstack([-D00, -D00, -D00, -D00, -D95, -D96, -D00, -D98, Dxx, -D910, -D911, -D912, -D913, -D914, -D915, -D916, -D00, -D918, -D00, -D00])

    # Atom 10
    D101 = np.dot(KAB4[:, :, 3], np.exp(1j * kx * (1/2*a)))
    D105 = np.dot(KAB1[:, :, 0], np.exp(1j * kx * (1/2*a))) + np.dot(KAB4[:, :, 2], np.exp(1j * kx * (-5/2*a)))
    D106 = np.dot(KAA[:, :, 0], np.exp(1j * kx * 0))
    D107 = np.dot(KAB3[:, :, 2], np.exp(1j * kx * (-a))) + np.dot(KAB4[:, :, 0], np.exp(1j * kx * (2*a)))
    D108 = np.dot(KAA[:, :, 1], np.exp(1j * kx * (3/2*a))) + np.dot(KAA[:, :, 5], np.exp(1j * kx * (-3/2*a)))
    D109 = np.dot(KAB1[:, :, 1], np.exp(1j * kx * (1/2*a))) + np.dot(KAB4[:, :, 5], np.exp(1j * kx * (-5/2*a)))
    D1011 = np.dot(KAB1[:, :, 2], np.exp(1j * kx * (-a))) + np.dot(KAB3[:, :, 0], np.exp(1j * kx * (2*a)))
    D1012 = np.dot(KAA[:, :, 2], np.exp(1j * kx * (3/2*a))) + np.dot(KAA[:, :, 4], np.exp(1j * kx * (-3/2*a)))
    D1013 = np.dot(KAB4[:, :, 1], np.exp(1j * kx * (1/2*a)))
    D1014 = np.dot(KAA[:, :, 3], np.exp(1j * kx * 0))
    D1015 = np.dot(KAB3[:, :, 1], np.exp(1j * kx * (-a))) + np.dot(KAB4[:, :, 4], np.exp(1j * kx * (2*a)))
    DB = np.hstack([-D101, -D00, -D00, -D00, -D105, -D106, -D107, -D108, -D109, Dxx, -D1011, -D1012, -D1013, -D1014, -D1015, -D00, -D00, -D00, -D00, -D00])

    # Atom 11
    D114 = np.dot(KBA4[:, :, 1], np.exp(1j * kx * (-1/2*a)))
    D115 = np.dot(KBB[:, :, 4], np.exp(1j * kx * (3/2*a))) + np.dot(KBB[:, :, 2], np.exp(1j * kx * (-3/2*a)))
    D116 = np.dot(KBA3[:, :, 1], np.exp(1j * kx * a)) + np.dot(KBA4[:, :, 4], np.exp(1j * kx * (-2*a)))
    D117 = np.dot(KBB[:, :, 3], np.exp(1j * kx * 0))
    D118 = np.dot(KBA1[:, :, 1], np.exp(1j * kx * (-1/2*a))) + np.dot(KBA4[:, :, 5], np.exp(1j * kx * (5/2*a)))
    D119 = np.dot(KBB[:, :, 5], np.exp(1j * kx * (3/2*a))) + np.dot(KBB[:, :, 1], np.exp(1j * kx * (-3/2*a)))
    D1110 = np.dot(KBA1[:, :, 2], np.exp(1j * kx * a)) + np.dot(KBA3[:, :, 0], np.exp(1j * kx * (-2*a)))
    D1112 = np.dot(KBA1[:, :, 0], np.exp(1j * kx * (-1/2*a))) + np.dot(KBA4[:, :, 2], np.exp(1j * kx * (5/2*a)))
    D1114 = np.dot(KBA3[:, :, 2], np.exp(1j * kx * a)) + np.dot(KBA4[:, :, 0], np.exp(1j * kx * (-2*a)))
    D1115 = np.dot(KBB[:, :, 0], np.exp(1j * kx * 0))
    D1116 = np.dot(KBA4[:, :, 3], np.exp(1j * kx * (-1/2*a)))
    DC = np.hstack([-D00, -D00, -D00, -D114, -D115, -D116, -D117, -D118, -D119, -D1110, Dxx, -D1112, -D00, -D1114, -D1115, -D1116, -D00, -D00, -D00, -D00])

    # Atom 12
    D125 = np.dot(KAB3[:, :, 2], np.exp(1j * kx * (-a))) + np.dot(KAB4[:, :, 0], np.exp(1j * kx * (2*a)))
    D127 = np.dot(KAB4[:, :, 3], np.exp(1j * kx * (1/2*a)))
    D128 = np.dot(KAA[:, :, 0], np.exp(1j * kx * 0))
    D129 = np.dot(KAB1[:, :, 2], np.exp(1j * kx * (-a))) + np.dot(KAB3[:, :, 0], np.exp(1j * kx * (2*a)))
    D1210 = np.dot(KAA[:, :, 1], np.exp(1j * kx * (3/2*a))) + np.dot(KAA[:, :, 5], np.exp(1j * kx * (-3/2*a)))
    D1211 = np.dot(KAB1[:, :, 0], np.exp(1j * kx * (1/2*a))) + np.dot(KAB4[:, :, 2], np.exp(1j * kx * (-5/2*a)))
    D1213 = np.dot(KAB3[:, :, 1], np.exp(1j * kx * (-a))) + np.dot(KAB4[:, :, 4], np.exp(1j * kx * (2*a)))
    D1214 = np.dot(KAA[:, :, 2], np.exp(1j * kx * (3/2*a))) + np.dot(KAA[:, :, 4], np.exp(1j * kx * (-3/2*a)))
    D1215 = np.dot(KAB1[:, :, 1], np.exp(1j * kx * (1/2*a))) + np.dot(KAB4[:, :, 5], np.exp(1j * kx * (-5/2*a)))
    D1216 = np.dot(KAA[:, :, 3], np.exp(1j * kx * 0))
    D1219 = np.dot(KAB4[:, :, 1], np.exp(1j * kx * (1/2*a)))
    DD = np.hstack([-D00, -D00, -D00, -D00, -D125, -D00, -D127, -D128, -D129, -D1210, -D1211, Dxx, -D1213, -D1214, -D1215, -D1216, -D00, -D00, -D1219, -D00])

    D[0:3, np.r_[0:36, 12*N-24:12*N]] = np.hstack([DA[:, 24:], DA[:, :24]])
    D[3:6, np.r_[0:36, 12*N-24:12*N]] = np.hstack([DB[:, 24:], DB[:, :24]])
    D[6:9, np.r_[0:36, 12*N-24:12*N]] = np.hstack([DC[:, 24:], DC[:, :24]])
    D[9:12, np.r_[0:36, 12*N-24:12*N]] = np.hstack([DD[:, 24:], DD[:, :24]])

    D[12:15, np.r_[0:48, 12*N-12:12*N]] = np.hstack([DA[:, 12:], DA[:, :12]])
    D[15:18, np.r_[0:48, 12*N-12:12*N]] = np.hstack([DB[:, 12:], DB[:, :12]])
    D[18:21, np.r_[0:48, 12*N-12:12*N]] = np.hstack([DC[:, 12:], DC[:, :12]])
    D[21:24, np.r_[0:48, 12*N-12:12*N]] = np.hstack([DD[:, 12:], DD[:, :12]])

    if N >= 5:
        for j in range(N-4):
            D[12*j+24:12*j+27, 12*j:12*j+60] = DA
            D[12*j+27:12*j+30, 12*j:12*j+60] = DB
            D[12*j+30:12*j+33, 12*j:12*j+60] = DC
            D[12*j+33:12*j+36, 12*j:12*j+60] = DD

    D[12*N-24:12*N-21, np.r_[0:12, 12*N-48:12*N]] = np.hstack([DA[:, 48:], DA[:, :48]])
    D[12*N-21:12*N-18, np.r_[0:12, 12*N-48:12*N]] = np.hstack([DB[:, 48:], DB[:, :48]])
    D[12*N-18:12*N-15, np.r_[0:12, 12*N-48:12*N]] = np.hstack([DC[:, 48:], DC[:, :48]])
    D[12*N-15:12*N-12, np.r_[0:12, 12*N-48:12*N]] = np.hstack([DD[:, 48:], DD[:, :48]])

    D[12*N-12:12*N-9, np.r_[0:24, 12*N-36:12*N]] = np.hstack([DA[:, 36:], DA[:, :36]])
    D[12*N-9:12*N-6, np.r_[0:24, 12*N-36:12*N]] = np.hstack([DB[:, 36:], DB[:, :36]])
    D[12*N-6:12*N-3, np.r_[0:24, 12*N-36:12*N]] = np.hstack([DC[:, 36:], DC[:, :36]])
    D[12*N-3:12*N, np.r_[0:24, 12*N-36:12*N]] = np.hstack([DD[:, 36:], DD[:, :36]])

    if N == 4:
        D[0:3, :] = np.hstack([DA[:, 24:48], DA[:, :12] + DA[:, 48:], DA[:, 12:24]])
        D[3:6, :] = np.hstack([DB[:, 24:48], DB[:, :12] + DB[:, 48:], DB[:, 12:24]])
        D[6:9, :] = np.hstack([DC[:, 24:48], DC[:, :12] + DC[:, 48:], DC[:, 12:24]])
        D[9:12, :] = np.hstack([DD[:, 24:48], DD[:, :12] + DD[:, 48:], DD[:, 12:24]])

        D[12:15, :] = np.hstack([DA[:, 12:48], DA[:, :12] + DA[:, 48:]])
        D[15:18, :] = np.hstack([DB[:, 12:48], DB[:, :12] + DB[:, 48:]])
        D[18:21, :] = np.hstack([DC[:, 12:48], DC[:, :12] + DC[:, 48:]])
        D[21:24, :] = np.hstack([DD[:, 12:48], DD[:, :12] + DD[:, 48:]])

        D[24:27, :] = np.hstack([DA[:, :12] + DA[:, 48:], DA[:, 12:48]])
        D[27:30, :] = np.hstack([DB[:, :12] + DB[:, 48:], DB[:, 12:48]])
        D[30:33, :] = np.hstack([DC[:, :12] + DC[:, 48:], DC[:, 12:48]])
        D[33:36, :] = np.hstack([DD[:, :12] + DD[:, 48:], DD[:, 12:48]])

        D[36:39, :] = np.hstack([DA[:, 36:48], DA[:, :12] + DA[:, 48:], DA[:, 12:36]])
        D[39:42, :] = np.hstack([DB[:, 36:48], DB[:, :12] + DB[:, 48:], DB[:, 12:36]])
        D[42:45, :] = np.hstack([DC[:, 36:48], DC[:, :12] + DC[:, 48:], DC[:, 12:36]])
        D[45:48, :] = np.hstack([DD[:, 36:48], DD[:, :12] + DD[:, 48:], DD[:, 12:36]])

    e = np.linalg.eigvals(D)
    w = np.sort(np.real(np.sqrt(e)))
    omega[:, i] = w / np.sqrt(m)

omega_new = omega.copy()
omega1 = omega[:, 0]
omega2 = omega[:, 1]
for i in range(2, len(klist)):
    omega3 = 2 * omega2 - omega1
    for j in range(len_omega):
        ind = np.argmin(np.abs(omega[:, i] - omega3[j]))
        omega_new[j, i] = omega[ind, i]
    omega1 = omega_new[:, i-1]
    omega2 = omega_new[:, i]

v_g = np.zeros_like(omega_new)

for i in range(len(omega_new)):
    v_g[i, :] = np.abs(np.gradient(omega_new[i, :])) / (klist[1] - klist[0])

# 计算声子通道数
frequency = np.linspace(0, 3.2e14, 100)
phonon_channels = np.zeros_like(frequency)

for i, f in enumerate(frequency):
    omega0 = frequency[i]  # Example value, replace with the desired frequency
    intersections = np.zeros(len_omega)
    for j in range(len_omega):
        omega_vals = omega_new[j, :]
        intersections[j] = np.sum((omega_vals[:-1] < omega0) & (omega_vals[1:] >= omega0)) + \
                       np.sum((omega_vals[:-1] >= omega0) & (omega_vals[1:] < omega0))
    total_intersections = np.sum(intersections)
    phonon_channels[i] = total_intersections

# 计算声子态密度
frequency_bins = np.linspace(0, 3.2e14, 1000)
phonon_DOS = np.zeros(len(frequency_bins) - 1)

for i in range(len_omega):
    omega_vals = omega_new[i, :]
    hist, _ = np.histogram(omega_vals, bins=frequency_bins)
    phonon_DOS += hist

# 绘制色散关系图
plt.figure(figsize=(16, 9))

# 子图1：绘制色散关系
plt.subplot(1, 4, 1)
for i in range(len_omega):
    plt.plot(klist / k_max, omega_new[i, :] / 1e12, 'r-', linewidth=2)
plt.xlabel('$k_x/k_{max}$', fontsize=20)
plt.ylabel('$\omega$ (THz)', fontsize=20)
plt.xlim([0, 1])
plt.ylim([0, 310])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14, width=2)

# 子图2：绘制群速度
plt.subplot(1, 4, 2)
for i in range(len_omega):
    plt.plot(v_g[i, 1:] / 1e3, omega_new[i, 1:] / 1e12, 'r-', linewidth=2)
plt.xlabel('$v_g$ (km/s)', fontsize=20)
plt.ylabel('$\omega$ (THz)', fontsize=20)
plt.xlim([0, 25])
plt.ylim([0, 310])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14, width=2)

# 子图3：绘制声子通道数
plt.subplot(1, 4, 3)
plt.plot(phonon_channels, frequency / 1e12, 'r-', linewidth=2)
plt.xlabel('$\\tau$', fontsize=20)
plt.ylabel('$\omega$ (THz)', fontsize=20)
plt.xlim([0, len_omega/2])
plt.ylim([0, 310])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14, width=2)

# 子图4：绘制声子态密度
plt.subplot(1, 4, 4)
plt.plot(phonon_DOS/(len_omega*10), frequency_bins[:-1] / 1e12, 'r-', linewidth=2)
plt.xlabel('DOS (a.u.)', fontsize=20)
plt.ylabel('$\omega$ (THz)', fontsize=20)
plt.xlim([0, 1])
plt.ylim([0, 310])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14, width=2)
plt.tight_layout()

plt.savefig('Fig_disp_AGNR.png', dpi=300)

plt.show()

# Define data for exporting to CSV
data = {
    'frequency_bins': frequency_bins[:-1],
    'phonon_DOS': phonon_DOS / (len_omega * 10)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Export to CSV
output_path = 'Result_phonon_DOS_AGNR.csv'
df.to_csv(output_path, index=False)

output_path

# 比热容计算函数
def heat_capacity(omega, dos, T):
    def integrand(omega, dos, T):
        x = h_bar * omega / (kB * T)
        x_clipped = np.clip(x, None, 600)  # 防止溢出
        return (x**2 * np.exp(x_clipped) / (np.exp(x_clipped) - 1)**2) * dos

    # 计算被积函数的值
    integrand_values = integrand(omega, dos, T)
    Cv = kB * np.trapz(integrand_values, omega)
    return Cv


# 计算调整系数，使得在5000K时比热容为2078 J/K/kg
target_heat_capacity = 2078  # J/K/kg
temperature_high = 5000  # K

# 原始比热容
Cv_original = heat_capacity(frequency_bins[1:], phonon_DOS, temperature_high)

# 计算调整系数
adjustment_factor = target_heat_capacity / Cv_original

# 计算不同温度下的比热容并应用调整系数
temperatures = np.linspace(1, 3000, 3000)
heat_capacities = [heat_capacity(frequency_bins[1:], phonon_DOS, T) * adjustment_factor for T in temperatures]

# 导出比热容数据到CSV
heat_capacity_data = {
    'Temperature (K)': temperatures,
    'Heat Capacity (J/(K·kg))': heat_capacities
}
df_heat_capacity = pd.DataFrame(heat_capacity_data)
df_heat_capacity.to_csv('Heat_capacity_AGNR.csv', index=False)

# 绘制比热容随温度变化图
plt.figure(figsize=(8, 6))
plt.loglog(temperatures, heat_capacities, 'r-', linewidth=2, label='Calculated Heat Capacity')

plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('Heat Capacity $C_v$ (J/(K·kg))', fontsize=14)
# plt.title('Heat Capacity vs Temperature with Low Temperature Fit', fontsize=16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('Heat_capacity_AGNR.png', dpi=300)
plt.show()
