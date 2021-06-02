import numpy as np

x = []
y = []
z = []
w = []

fx = open("x.txt")
fy = open("y.txt")
fz = open("z.txt")
fw = open("w.txt")

linesx = fx.readlines()
for line1 in linesx:
    x.append(float(line1))
linesy = fy.readlines()
for line2 in linesy:
    y.append(float(line2))
linesz = fz.readlines()
for line3 in linesz:
    z.append(float(line3))
linesw = fw.readlines()
for line4 in linesw:
    w.append(float(line4))

fx.close()
fy.close()
fz.close()
fw.close()
x = np.array(x)
y = np.array(y)
z = np.array(z)
w = np.array(w)
xyz = np.array(list(zip(x, y, z)))
index = []
index = [9, 17, 35, 66, 80]#[71.24145725] [71.24145725]     [71.21581402] [71.26013255]  [71.2162956] [71.25987262]
                        #[71.24145725] [71.24145725]  [837.1937944058782, 917.8196687586696, 799.8892563327363, 417.20353599282447][7, 10, 35, 66]
#for i in range(45):
#    index.append(i)
# index.append(104)

print(index)

xyz_0 = np.array([0, 0, 0])
locate = []
for i in index:
    xyz_1 = np.array(xyz[i][:])
    dist = np.linalg.norm(xyz_1 - xyz_0)
    locate.append(dist)
print(locate)

M = len(index)  # 发射天线数目    需要与index中的数量一致！！！

N = M # 接收天线
BT = 1000000 * np.ones((M, 1))
R_t_case1_4 = 1000 * np.ones((1, M), dtype=int)
R_t_case5_8 = locate
R_r_case1_4 = 1000 * np.ones((1, N), dtype=int)
R_r_case5_8 = locate

phi_t_case1 = np.pi / 32 * np.ones((1, M)) + np.arange(0, (2 * np.pi - 2 * np.pi / M + 2 * np.pi / M), 2 * np.pi / M)
phi_r_case1 = np.arange(0, (2 * np.pi - 2 * np.pi / N) + 2 * np.pi / N, 2 * np.pi / N)
pos_t_case1_x = [a * b for a, b in zip(R_t_case1_4, np.cos(phi_t_case1))]
pos_t_case1_y = [a * b for a, b in zip(R_t_case1_4, np.sin(phi_t_case1))]
pos_r_case1_x = [a * b for a, b in zip(R_r_case1_4[0, :], np.cos(phi_r_case1))]
pos_r_case1_y = [a * b for a, b in zip(R_r_case1_4[0, :], np.sin(phi_r_case1))]

pos_t_case1_x = np.array(pos_t_case1_x)
pos_t_case1_y = np.array(pos_t_case1_y)
pos_r_case1_x = np.array(pos_r_case1_x)
pos_r_case1_y = np.array(pos_r_case1_y)

phi_t_case1 = np.pi / 32 * np.ones((1, M)) + np.arange(0, (2 * np.pi - 2 * np.pi / M + 2 * np.pi / M), 2 * np.pi / M)
phi_r_case1 = np.arange(0, (2 * np.pi - 2 * np.pi / N) + 2 * np.pi / N, 2 * np.pi / N)

pos_t_case5_x = [a * b for a, b in zip(R_t_case5_8, np.cos(phi_t_case1))]
pos_t_case5_y = [a * b for a, b in zip(R_t_case5_8, np.sin(phi_t_case1))]
pos_r_case5_x = [a * b for a, b in zip(R_r_case5_8, np.cos(phi_r_case1))]
pos_r_case5_y = [a * b for a, b in zip(R_r_case5_8, np.sin(phi_r_case1))]
pos_t_case5_x = np.array(pos_t_case5_x)
pos_t_case5_y = np.array(pos_t_case5_y)
pos_r_case5_x = np.array(pos_r_case5_x)
pos_r_case5_y = np.array(pos_r_case5_y)

phi_t_case2 = np.pi / 32 * np.ones((1, M)) + np.arange(0, (np.pi / 2 - np.pi / (2 * M)) + np.pi / (2 * M),
                                                       np.pi / (2 * M))
phi_r_case2 = np.arange(0, (np.pi / 2 - np.pi / (2 * N)) + np.pi / (2 * N), np.pi / (2 * N))

pos_t_case6_x = [a * b for a, b in zip(R_t_case5_8, np.cos(phi_t_case2))]
pos_t_case6_y = [a * b for a, b in zip(R_t_case5_8, np.sin(phi_t_case2))]
pos_r_case6_x = [a * b for a, b in zip(R_r_case5_8, np.cos(phi_r_case2))]
pos_r_case6_y = [a * b for a, b in zip(R_r_case5_8, np.sin(phi_r_case2))]
pos_t_case6_x = np.array(pos_t_case6_x)
pos_t_case6_y = np.array(pos_t_case6_y)
pos_r_case6_x = np.array(pos_r_case6_x)
pos_r_case6_y = np.array(pos_r_case6_y)

phi_t_case3 = np.pi / 32 * np.ones((1, M)) + np.arange(0, (np.pi - np.pi / M) + np.pi / M, np.pi / M)
phi_r_case3 = np.arange(0, (np.pi - np.pi / N) + np.pi / N, np.pi / N)

pos_t_case7_x = [a * b for a, b in zip(R_t_case5_8, np.cos(phi_t_case3))]
pos_t_case7_y = [a * b for a, b in zip(R_t_case5_8, np.sin(phi_t_case3))]
pos_r_case7_x = [a * b for a, b in zip(R_r_case5_8, np.cos(phi_r_case3))]
pos_r_case7_y = [a * b for a, b in zip(R_r_case5_8, np.sin(phi_r_case3))]
pos_t_case7_x = np.array(pos_t_case7_x)
pos_t_case7_y = np.array(pos_t_case7_y)
pos_r_case7_x = np.array(pos_r_case7_x)
pos_r_case7_y = np.array(pos_r_case7_y)

phi_t_case4 = np.pi / 32 * np.ones((1, M)) + np.arange(0, (np.pi - np.pi / M) + np.pi / M, np.pi / M)
phi_r_case4 = (np.pi / 32 + np.pi) * np.ones((1, N)) + np.arange(0, (np.pi - np.pi / N) + np.pi / N, np.pi / N)

pos_t_case8_x = [a * b for a, b in zip(R_t_case5_8, np.cos(phi_t_case4))]
pos_t_case8_y = [a * b for a, b in zip(R_t_case5_8, np.sin(phi_t_case4))]
pos_r_case8_x = [a * b for a, b in zip(R_r_case5_8, np.cos(phi_r_case4))]
pos_r_case8_y = [a * b for a, b in zip(R_r_case5_8, np.sin(phi_r_case4))]
pos_t_case8_x = np.array(pos_t_case8_x)
pos_t_case8_y = np.array(pos_t_case8_y)
pos_r_case8_x = np.array(pos_r_case8_x)
pos_r_case8_y = np.array(pos_r_case8_y)

# 目标信息
target_x = 10 # 目标默认取原点  目标位置可以改变
target_y = 0
SNR_dB = 0
pos_target = [target_x, target_y]
noise_power = pow(10, (-SNR_dB / 10))  # 信噪比为1
c = 3e8  # 光速
# target_RCS=5+1i*4,非相干不考虑雷达散射截面，非相干使用发射-接收通道散射系数h来表示雷达散射截面，h为雷达散射截面乘以发射-接收通道的相位变化。

h1 = np.ones((N, M))


def CRLB_nc(noise_power, c, BT, alpha, pos_T_nc, pos_R_nc, pos_target):
    BT_mean = np.mean(BT)
    BT_M = BT / BT_mean
    yita_nc = pow(c, 2) * noise_power / (8 * pow(np.pi, 2) * pow(BT_mean, 2))
    gx = 0
    gy = 0
    hc = 0
    M_nc = np.size(pos_T_nc, 1)
    N_nc = np.size(pos_R_nc, 1)
    pos_T_nc = np.array(pos_T_nc)
    pos_R_nc = np.array(pos_R_nc)
    target_x = pos_target[0]
    target_y = pos_target[1]
    target_y = pos_target[1]
    alpha = np.array(alpha)
    for m in range(M_nc):
        for n in range(N_nc):
            R_t_target = np.linalg.norm([pos_T_nc[:, m] - pos_target], ord=2)
            R_r_target = np.linalg.norm([pos_R_nc[:, n] - pos_target], ord=2)
            atrx = ((pos_T_nc[0, m] - target_x) / R_t_target + (pos_R_nc[0, n] - target_x) / R_r_target)
            btrx = ((pos_T_nc[1, m] - target_y) / R_t_target + (pos_R_nc[1, n] - target_y) / R_r_target)
            gx = gx + btrx * btrx * pow(alpha[m, n], 2) * pow(BT_M[m], 2)
            gy = gy + atrx * atrx * pow(alpha[m, n], 2) * pow(BT_M[m], 2)
            hc = hc - atrx * btrx * pow(alpha[m, n], 2) * pow(BT_M[m], 2)
    x_CRB_nc = yita_nc * gx / (gx * gy - pow(hc, 2))
    y_CRB_nc = yita_nc * gy / (gx * gy - pow(hc, 2))
    return x_CRB_nc, y_CRB_nc


# 相参处理
alpha = h1
pos_T_nc = [np.conjugate(pos_t_case5_x[0, 0:M]), np.conjugate(pos_t_case5_y[0, 0:M])]
pos_R_nc = [np.conjugate(pos_r_case5_x[0:N]), np.conjugate(pos_r_case5_y[0:N])]
# print(pos_T_nc)
# print(pos_R_nc)

[x_CRB_nc, y_CRB_nc] = CRLB_nc(noise_power, c, BT, alpha, pos_T_nc, pos_R_nc, pos_target)
print(x_CRB_nc, y_CRB_nc)