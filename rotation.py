import numpy as np

# Define rotation function
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


# Define stiffness matrix rotation function
def K(theta, k):
    U = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    inv_U = np.linalg.inv(U)
    k_rotated = np.dot(np.dot(inv_U, k), U)
    return k_rotated
