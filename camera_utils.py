import numpy as np


tol = 1e-7

def skew_mat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def hat(tau):
    omega = tau[:3]
    v = tau[3:]
    omega_hat = skew_mat(omega)
    tau_hat = np.block([[omega_hat, v.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def exp(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    u_hat = skew_mat(u)
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    V = np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)
    M = np.block([[Exp_theta, V @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def Exp(tau):
    v = tau[:3]
    rho = tau[3:]
    theta = np.linalg.norm(v)
    if abs(theta) < tol:
        return rho, 0., np.array([0., 0., 1.])
    u = v / theta
    u_hat = np.array([[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]])
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    V = np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)
    M = np.block([[Exp_theta, V @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def Log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    if theta < tol:
        w = np.array([(R[2,1] - R[1,2]) / 2, (R[0,2] - R[2,0]) / 2, (R[1,0] - R[0,1]) / 2])
        V_inv = np.eye(3)
    else:
        u_hat = theta * (R - R.T) / (2 * np.sin(theta))
        w = np.array([u_hat[2,1], u_hat[0,2], u_hat[1,0]])
        V = np.eye(3) + (1 - np.cos(theta)) / theta**2 * u_hat + (theta - np.sin(theta)) / theta**3 * (u_hat @ u_hat)
        V_inv = np.linalg.inv(V)
    rho = V_inv @ t
    return np.block([w, rho])

def Log2(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    if abs(theta) < tol:
        return np.block([theta * np.array([0., 0., 1.]), t])
    v_hat = theta * (R - R.T) / (2. * np.sin(theta))
    u = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]]) / theta
    u_hat = np.array([[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]])
    V = np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)
    rho = np.linalg.inv(V) @ t if abs(theta) >= tol else t
    return np.block([theta * u, rho])

def adjoint(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew_mat(t) @ R, R]])
