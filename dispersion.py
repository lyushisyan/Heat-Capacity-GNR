import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rotation import rotate, K

def compute_dispersion(N, GNR):
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
    d_list = 1000


    len_omega = 12 * N
    omega = np.zeros((len_omega, d_list))

    if GNR == 'AGNR':

        k_max = np.pi / 3 / a
        klist = np.linspace(0, k_max, d_list)

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

    elif GNR == 'ZGNR':

        k_max = np.sqrt(3)*np.pi /3/a
        klist = np.linspace(0, k_max, d_list)

        for i in range(len(klist)):
            kx = klist[i]
            D = np.zeros((len_omega, len_omega), dtype=complex)
            D00 = np.zeros((3, 3))
            Dxx = np.sum(KBA1, axis=2) + np.sum(KBB, axis=2) + np.sum(KBA3, axis=2) + np.sum(KBA4, axis=2)

            # D5
            D52 = np.dot(KBA4[:, :, 2], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBA4[:, :, 5], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D53 = np.dot(KBB[:, :, 4], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KBB[:, :, 5], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))
            D54 = np.dot(KBA1[:, :, 2], np.exp(1j * kx * 0)) + np.dot(KBA3[:, :, 1],
                                                                      np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(
                KBA3[:, :, 2], np.exp(1j * kx * (np.sqrt(3) * a)))
            D55 = np.dot(KBB[:, :, 0], np.exp(1j * kx * (np.sqrt(3) * a))) + np.dot(KBB[:, :, 3],
                                                                                    np.exp(1j * kx * (-np.sqrt(3) * a)))
            D56 = np.dot(KBA1[:, :, 0], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBA1[:, :, 1], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KBA4[:, :, 1],
                                                           np.exp(1j * kx * (-3 * np.sqrt(3) / 2 * a))) + np.dot(
                KBA4[:, :, 3], np.exp(1j * kx * (3 * np.sqrt(3) / 2 * a)))
            D57 = np.dot(KBB[:, :, 1], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBB[:, :, 2], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D58 = np.dot(KBA3[:, :, 0], np.exp(1j * kx * 0)) + np.dot(KBA4[:, :, 0],
                                                                      np.exp(1j * kx * (np.sqrt(3) * a))) + np.dot(
                KBA4[:, :, 4], np.exp(1j * kx * (-np.sqrt(3) * a)))

            # D6
            D63 = np.dot(KAB3[:, :, 0], np.exp(1j * kx * 0)) + np.dot(KAB4[:, :, 0],
                                                                      np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(
                KAB4[:, :, 4], np.exp(1j * kx * (np.sqrt(3) * a)))
            D64 = np.dot(KAA[:, :, 1], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAA[:, :, 2], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))
            D65 = np.dot(KAB1[:, :, 0], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAB1[:, :, 1], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KAB4[:, :, 1],
                                                          np.exp(1j * kx * (3 * np.sqrt(3) / 2 * a))) + np.dot(
                KAB4[:, :, 3], np.exp(1j * kx * (-3 * np.sqrt(3) / 2 * a)))
            D66 = np.dot(KAA[:, :, 0], np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(KAA[:, :, 3],
                                                                                     np.exp(1j * kx * np.sqrt(3) * a))
            D67 = np.dot(KAB1[:, :, 2], np.exp(1j * kx * 0)) + np.dot(KAB3[:, :, 1],
                                                                      np.exp(1j * kx * (np.sqrt(3) * a))) + np.dot(
                KAB3[:, :, 2], np.exp(1j * kx * (-np.sqrt(3) * a)))
            D68 = np.dot(KAA[:, :, 4], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KAA[:, :, 5], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D69 = np.dot(KAB4[:, :, 2], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAB4[:, :, 5], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))

            # D7
            D74 = np.dot(KBA4[:, :, 2], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBA4[:, :, 5], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D75 = np.dot(KBB[:, :, 4], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KBB[:, :, 5], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))
            D76 = np.dot(KBA1[:, :, 2], np.exp(1j * kx * 0)) + np.dot(KBA3[:, :, 1],
                                                                      np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(
                KBA3[:, :, 2], np.exp(1j * kx * (np.sqrt(3) * a)))
            D77 = np.dot(KBB[:, :, 0], np.exp(1j * kx * (np.sqrt(3) * a))) + np.dot(KBB[:, :, 3],
                                                                                    np.exp(1j * kx * (-np.sqrt(3) * a)))
            D78 = np.dot(KBA1[:, :, 0], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBA1[:, :, 1], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KBA4[:, :, 1],
                                                           np.exp(1j * kx * (-3 * np.sqrt(3) / 2 * a))) + np.dot(
                KBA4[:, :, 3], np.exp(1j * kx * (3 * np.sqrt(3) / 2 * a)))
            D79 = np.dot(KBB[:, :, 1], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KBB[:, :, 2], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D710 = np.dot(KBA3[:, :, 0], np.exp(1j * kx * 0)) + np.dot(KBA4[:, :, 0],
                                                                       np.exp(1j * kx * (np.sqrt(3) * a))) + np.dot(
                KBA4[:, :, 4], np.exp(1j * kx * (-np.sqrt(3) * a)))

            # D8
            D85 = np.dot(KAB3[:, :, 0], np.exp(1j * kx * 0)) + np.dot(KAB4[:, :, 0],
                                                                      np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(
                KAB4[:, :, 4], np.exp(1j * kx * (np.sqrt(3) * a)))
            D86 = np.dot(KAA[:, :, 1], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAA[:, :, 2], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))
            D87 = np.dot(KAB1[:, :, 0], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAB1[:, :, 1], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KAB4[:, :, 1],
                                                          np.exp(1j * kx * (3 * np.sqrt(3) / 2 * a))) + np.dot(
                KAB4[:, :, 3], np.exp(1j * kx * (-3 * np.sqrt(3) / 2 * a)))
            D88 = np.dot(KAA[:, :, 0], np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(KAA[:, :, 3],
                                                                                     np.exp(1j * kx * (np.sqrt(3) * a)))
            D89 = np.dot(KAB1[:, :, 2], np.exp(1j * kx * 0)) + np.dot(KAB3[:, :, 2],
                                                                      np.exp(1j * kx * (-np.sqrt(3) * a))) + np.dot(
                KAB3[:, :, 1], np.exp(1j * kx * (np.sqrt(3) * a)))
            D810 = np.dot(KAA[:, :, 4], np.exp(1j * kx * (np.sqrt(3) / 2 * a))) + np.dot(KAA[:, :, 5], np.exp(
                1j * kx * (-np.sqrt(3) / 2 * a)))
            D811 = np.dot(KAB4[:, :, 2], np.exp(1j * kx * (-np.sqrt(3) / 2 * a))) + np.dot(KAB4[:, :, 5], np.exp(
                1j * kx * (np.sqrt(3) / 2 * a)))

            D[0:3, np.r_[0:12, 12 * N - 9:12 * N]] = np.hstack([Dxx - D55, -D56, -D57, -D58, -D52, -D53, -D54])
            D[3:6, np.r_[0:15, 12 * N - 6:12 * N]] = np.hstack([-D65, Dxx - D66, -D67, -D68, -D69, -D63, -D64])
            D[6:9, np.r_[0:18, 12 * N - 3:12 * N]] = np.hstack([-D75, -D76, Dxx - D77, -D78, -D79, -D710, -D74])
            D[9:12, 0:21] = np.hstack([-D85, -D86, -D87, Dxx - D88, -D89, -D810, -D811])

            for j in range(N - 2):
                D[12 * j + 12:12 * j + 15, 12 * j + 3:12 * j + 24] = np.hstack(
                    [-D52, -D53, -D54, Dxx - D55, -D56, -D57, -D58])
                D[12 * j + 15:12 * j + 18, 12 * j + 6:12 * j + 27] = np.hstack(
                    [-D63, -D64, -D65, Dxx - D66, -D67, -D68, -D69])
                D[12 * j + 18:12 * j + 21, 12 * j + 9:12 * j + 30] = np.hstack(
                    [-D74, -D75, -D76, Dxx - D77, -D78, -D79, -D710])
                D[12 * j + 21:12 * j + 24, 12 * j + 12:12 * j + 33] = np.hstack(
                    [-D85, -D86, -D87, Dxx - D88, -D89, -D810, -D811])

            D[12 * N - 12:12 * N - 9, 12 * N - 21:12 * N] = np.hstack([-D52, -D53, -D54, Dxx - D55, -D56, -D57, -D58])
            D[12 * N - 9:12 * N - 6, np.r_[0:3, 12 * N - 18:12 * N]] = np.hstack(
                [-D69, -D63, -D64, -D65, Dxx - D66, -D67, -D68])
            D[12 * N - 6:12 * N - 3, np.r_[0:6, 12 * N - 15:12 * N]] = np.hstack(
                [-D79, -D710, -D74, -D75, -D76, Dxx - D77, -D78])
            D[12 * N - 3:12 * N, np.r_[0:9, 12 * N - 12:12 * N]] = np.hstack(
                [-D89, -D810, -D811, -D85, -D86, -D87, Dxx - D88])

            e = np.linalg.eigvals(D)
            w = np.sort(np.real(np.sqrt(e)))
            omega[:, i] = w / np.sqrt(m)


    else:
        raise ValueError("Invalid GNR type. Please choose 'AGNR' or 'ZGNR'.")

    # Flatten frequencies from all modes and k-points for DOS calculation
    all_frequencies = omega.flatten()

    # Define frequency bins for the histogram
    frequency_bins = 200  # Number of bins for the histogram
    frequency_min = np.min(all_frequencies)
    frequency_max = np.max(all_frequencies)
    frequency_edges = np.linspace(frequency_min, frequency_max, frequency_bins + 1)
    delta_frequency = frequency_edges[1] - frequency_edges[0]

    # Calculate DOS using histogram method
    DOS_counts, _ = np.histogram(all_frequencies, bins=frequency_edges)

    # Compute DOS without normalization
    DOS = DOS_counts / delta_frequency  # Units: states per rad/s

    # Compute normalized DOS (optional)
    total_states = len(all_frequencies)
    DOS_normalized = DOS_counts / (total_states * delta_frequency)  # Units: 1 per rad/s

    # Compute frequency centers for plotting
    frequency_centers = (frequency_edges[:-1] + frequency_edges[1:]) / 2

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), sharey=True)

    # Subplot 1: Plot dispersion relations
    for i in range(len_omega):
        ax1.plot(klist / k_max, omega[i, :] / 1e12, 'k-', linewidth=1)
    ax1.set_xlabel('$k_x/k_{max}$', fontsize=18)
    ax1.set_ylabel('$\omega$, rad/ps', fontsize=18)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 310])
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Subplot 2: Plot phonon density of states (DOS)
    ax2.plot(DOS_normalized * 1e12, frequency_centers / 1e12, 'k-', linewidth=1)
    ax2.set_xlabel('DOS, ps/rad', fontsize=18)
    ax2.set_xlim([0, 0.02])
    ax2.set_ylim([0, 310])
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'Fig_dispersion_and_DOS_{GNR}_N{N}.png', dpi=600)
    plt.show()

    return klist, omega, frequency_centers, DOS
