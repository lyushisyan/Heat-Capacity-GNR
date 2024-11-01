import numpy as np
import matplotlib.pyplot as plt

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

# 参数设置
calculation_type = 'high_symmetry_path'  # 'high_symmetry_path' or 'k_grid'
#calculation_type = 'k_grid'

a = 1.42e-10
a0 = a*np.sqrt(3)
m = 1.99e-26  # 质量（千克）
pi = np.pi
h_bar = 1.0546e-34  # 普朗克常数（Js）
kB = 1.38065e-23  # 玻尔兹曼常数（J/K）
f = np.diag([25.5452, 8.1228, 6.1784, 3.7730, -2.4664, -0.4502, -2.2194, 3.5381, 0.2783, 0.6546, 0.1865, -0.4609, 0.6218, 0.2200, 0.1554]) * 16.0217662

# 计算旋转后的向量和矩阵
A_I, B_I = rotate(2 / 3 * np.pi, np.array([[1, 0, 0]]).T * a *2/np.sqrt(3))
A_II, B_II = rotate(1 / 3 * np.pi, np.array([[3 / 2, np.sqrt(3) / 2, 0]]).T * a *2/np.sqrt(3))
A_III, B_III = rotate(2 / 3 * np.pi, np.array([[1, np.sqrt(3), 0]]).T * a *2/np.sqrt(3))
A_IV1, B_IV1 = rotate(2 / 3 * np.pi, np.array([[2.5, np.sqrt(3) / 2, 0]]).T * a *2/np.sqrt(3))
A_IV2, B_IV2 = rotate(2 / 3 * np.pi, np.array([[2.5, -np.sqrt(3) / 2, 0]]).T * a *2/np.sqrt(3))
A_V, B_V = rotate(1 / 3 * np.pi, np.array([[3, 0, 0]]).T * a *2/np.sqrt(3))

A_IV = np.concatenate((A_IV1, A_IV2), axis=1)
B_IV = np.concatenate((B_IV1, B_IV2), axis=1)

[KAB1, KBA1] = rotate(2 / 3 * pi, f[:3, :3])
[KAA, KBB] = rotate(1 / 3 * pi, K(1 / 6 * pi, f[3:6, 3:6]))
[KAB3, KBA3] = rotate(2 / 3 * pi, K(1 / 3 * pi, f[6:9, 6:9]))
[KAB4f, KBA4f] = rotate(2 / 3 * pi, K(0.333473, f[9:12, 9:12]))
[KAB4s, KBA4s] = rotate(2 / 3 * pi, K(0.333473, f[9:12, 9:12]))
[KAA5, KBB5] = rotate(1 / 3 * pi, f[12:15, 12:15])
KAB4 = np.concatenate((KAB4f, KAB4s), axis=2)
KBA4 = np.concatenate((KBA4f, KBA4s), axis=2)

b_1 = 2 * np.pi / a0 * np.array([1/2, -np.sqrt(3)/2, 0])
b_2 = 2 * np.pi / a0 * np.array([1/2, np.sqrt(3)/2, 0])

# 计算色散关系函数
def calculate_dispersion(path):
    W = np.zeros((6, len(path)))
    for idx, k in enumerate(path):
        D = np.zeros((6, 6), dtype=complex)
        DAAs = np.zeros((3, 3), dtype=complex)
        DBBs = np.zeros((3, 3), dtype=complex)
        DBAs = np.zeros((3, 3), dtype=complex)
        DABs = np.zeros((3, 3), dtype=complex)
        
        for ll in range(3):
            DAAs += (KAA[:, :, ll] * np.exp(1j * np.dot(k, -A_II[:, ll])) + 
                     KAA[:, :, ll+3] * np.exp(1j * np.dot(k, -A_II[:, ll+3])) +
                     KAA5[:, :, ll] * np.exp(1j * np.dot(k, -A_V[:, ll])) + 
                     KAA5[:, :, ll+3] * np.exp(1j * np.dot(k, -A_V[:, ll+3])))
            
            DBBs += (KBB[:, :, ll] * np.exp(1j * np.dot(k, -B_II[:, ll])) + 
                     KBB[:, :, ll+3] * np.exp(1j * np.dot(k, -B_II[:, ll+3])) +
                     KBB5[:, :, ll] * np.exp(1j * np.dot(k, -B_V[:, ll])) + 
                     KBB5[:, :, ll+3] * np.exp(1j * np.dot(k, -B_V[:, ll+3])))
            
            DABs += (KAB1[:, :, ll] * np.exp(1j * np.dot(k, -A_I[:, ll])) + 
                     KAB3[:, :, ll] * np.exp(1j * np.dot(k, -A_III[:, ll])) +
                     KAB4[:, :, ll] * np.exp(1j * np.dot(k, -A_IV[:, ll])) + 
                     KAB4[:, :, ll + 3] * np.exp(1j * np.dot(k, -A_IV[:, ll + 3])))
            
            DBAs += (KBA1[:, :, ll] * np.exp(1j * np.dot(k, -B_I[:, ll])) + 
                     KBA3[:, :, ll] * np.exp(1j * np.dot(k, -B_III[:, ll])) +
                     KBA4[:, :, ll] * np.exp(1j * np.dot(k, -B_IV[:, ll])) + 
                     KBA4[:, :, ll + 3] * np.exp(1j * np.dot(k, -B_IV[:, ll + 3])))

        D[0:3, 3:6] = -DABs
        D[3:6, 0:3] = -DBAs
        D[0:3, 0:3] = (np.sum(KAB1, axis=2) + np.sum(KAA, axis=2) + np.sum(KAB3, axis=2) + 
                       np.sum(KAB4f, axis=2) + np.sum(KAB4s, axis=2) + np.sum(KAA5, axis=2) - DAAs)
        D[3:6, 3:6] = (np.sum(KAB1, axis=2) + np.sum(KAA, axis=2) + np.sum(KAB3, axis=2) + 
                       np.sum(KAB4f, axis=2) + np.sum(KAB4s, axis=2) + np.sum(KAA5, axis=2) - DBBs)
        
        D_out = np.array([[D[2,2],D[2,5]],[D[5,2],D[5,5]]])
        D_in = np.array([[D[0,0],D[0,1],D[0,3],D[0,4]],
                         [D[1,0],D[1,1],D[1,3],D[1,4]],
                         [D[3,0],D[3,1],D[3,3],D[3,4]],
                         [D[4,0],D[4,1],D[4,3],D[4,4]]])

        e1 = np.linalg.eigvals(D_out)
        e2 = np.linalg.eigvals(D_in)
        w1 = np.sort(e1)
        w2 = np.sort(e2)
        W[:, idx] = np.real(np.sqrt(np.concatenate((w1,w2)))) / np.sqrt(m)
    return W

def cartesian_to_basis(k_point, b1, b2):
    A = np.array([b1[:2], b2[:2]]).T
    coeffs = np.linalg.solve(A, k_point[:2])
    return coeffs


# 高对称路径计算
if calculation_type == 'high_symmetry_path':
    # 定义高对称点
    Gamma = np.array([0, 0, 0])
    K = (1/3) * b_1 + (2/3) * b_2
    M = (1/2) * b_1 + (1/2) * b_2

    def generate_path(start, end, num_points=100):
        return np.linspace(start, end, num_points)

    # 生成高对称路径
    Gamma_M = generate_path(Gamma, M)
    M_K = generate_path(M, K)
    K_Gamma = generate_path(K, Gamma)
    path = np.vstack((Gamma_M, M_K, K_Gamma))

    # 计算色散关系
    W = calculate_dispersion(path)

    # 转换路径为 b1 和 b2 基矢量的系数表示
    path_in_basis = np.array([cartesian_to_basis(k, b_1, b_2) for k in path])


    # 保存到txt文件
    output_data = np.hstack((path_in_basis, np.zeros((len(path_in_basis), 1)), W.T))  # 将路径的b1,b2坐标与频率值拼接
    np.savetxt('band.txt', output_data, fmt='%.6e')

    # 绘制色散关系
    distances = np.zeros(len(path))
    for i in range(1, len(path)):
        distances[i] = distances[i - 1] + np.linalg.norm(path[i] - path[i - 1])

    plt.figure(figsize=(8, 6))
    for i in range(6):  # 6个分支
        plt.plot(distances, W[i, :], label=f'Branch {i+1}')

    high_symmetry_points = [0, distances[len(Gamma_M)-1], distances[len(Gamma_M) + len(M_K) - 1], distances[-1]]
    high_symmetry_labels = ['$\Gamma$', 'M', 'K', '$\Gamma$']
    plt.xticks(high_symmetry_points, high_symmetry_labels)
    plt.xlabel('k', fontsize=16)
    plt.ylabel('ω (THz)', fontsize=16)
    plt.grid(True)
    plt.show()


# k-网格计算
elif calculation_type == 'k_grid':
    n_points = 16
    u = np.linspace(0, 1, n_points)
    v = np.linspace(0, 1, n_points)
    U, V = np.meshgrid(u, v)
    k_points = np.array([U.ravel(), V.ravel()]).T
    k_grid = np.array([u * b_1 + v * b_2 for u, v in k_points])

    # 计算色散关系
    W = calculate_dispersion(k_grid)

    # 保存频率数据到txt文件，按分支顺序排列
    W_flattened = W.T.reshape(-1, 1)  # 将W拉平为单列，按顺序排列所有分支的10000行数据
    np.savetxt('frequency.txt', W_flattened, fmt='%.6e')

    kx = k_grid[:, 0]
    ky = k_grid[:, 1]

    kx_grid = kx.reshape((n_points, n_points))
    ky_grid = ky.reshape((n_points, n_points))
     
    # 计算声子的群速度
    v_gx = np.zeros_like(W)  # 存储群速度的x分量
    v_gy = np.zeros_like(W)  # 存储群速度的y分量
    v_gz = np.zeros_like(W)
    
    # 对每个分支计算群速度
    for i in range(6):  # 6个分支
        W_grid = W[i, :].reshape((n_points, n_points))  # 将频率数据reshape为二维网格
        
        # 计算频率相对于k_x和k_y的梯度（群速度）
        v_gx_grid, v_gy_grid = np.gradient(W_grid, kx_grid[0, :], ky_grid[:, 0])
        
        # 将群速度存储回原数组，展平以与W数据一致
        v_gx[i, :] = v_gx_grid.ravel()
        v_gy[i, :] = v_gy_grid.ravel()
        
        # 将群速度v_gx和v_gy保存到txt文件，按列保存
        v_g_combined = np.column_stack((v_gx.T.reshape(-1, 1), v_gy.T.reshape(-1, 1), v_gz.T.reshape(-1, 1)))  # 合并两列数据
        np.savetxt('velocity.txt', v_g_combined, fmt='%.6e')

    # 生成6个子图的布局
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))  # 3行2列布局

    # 遍历每个分支并绘制等高线图
    for i in range(6):  # 6个分支
        W_grid = W[i, :].reshape((n_points, n_points))  # 对应分支的频率数据
        row, col = divmod(i, 2)  # 确定每个分支的子图位置

        # 绘制等高线图
        contour = axs[row, col].contourf(kx_grid, ky_grid, W_grid, levels=50, cmap='viridis')
    
        # 添加颜色条和标签
        plt.colorbar(contour, ax=axs[row, col], label=f'Frequency (W{i+1})')
        axs[row, col].set_xlabel('kx')
        axs[row, col].set_ylabel('ky')
        axs[row, col].set_title(f'Branch {i+1}')
        axs[row, col].set_aspect('equal', adjustable='box')

    # 自动调整布局以避免子图重叠
    plt.tight_layout()
    plt.savefig('Frequency.png', dpi=600)
    plt.show()

