import numpy as np
# import pinocchio as pin
from .poe_fkine import poe_fk
from .SE3_utils import hat, exp, adjoint, Log
from .parameters import screws, Tws, Tsb, q_lb, q_ub


def jacobian(q, frame_in = "space", frame_ref = "world"):
    if frame_in == "space":
        jac_adjoints = [np.eye(6)]
        for k in range(len(screws[frame_in]) - 1):
            jac_adjoint_prev = jac_adjoints[-1]
            jac_adjoints.append(jac_adjoint_prev @ adjoint(exp(hat(screws[frame_in][k]) * q[k])))
    elif frame_in == "body":
        # jac_adjoints = [adjoint(exp(-hat(screws[frame_in][-1]) * q[-1]))]
        # for k in range(len(screws[frame_in]) - 2, -1, -1):
        #     jac_adjoint_prev = jac_adjoints[0]
        #     jac_adjoints.insert(0, jac_adjoint_prev @ adjoint(exp(-hat(screws[frame_in][k]) * q[k])))
        jac_adjoints = [np.eye(6)]
        for k in range(len(screws[frame_in]) - 1, 0, -1):
            jac_adjoint_prev = jac_adjoints[0]
            jac_adjoints.insert(0, jac_adjoint_prev @ adjoint(exp(-hat(screws[frame_in][k]) * q[k])))
    jac_screws = []
    for k in range(len(jac_adjoints)):
        jac_screws.append(jac_adjoints[k] @ screws[frame_in][k])
    jac = np.zeros((len(screws[frame_in][0]), len(q)))
    for k in range(len(jac_screws)):
        jac[:, k] = jac_screws[k]
    if frame_in == "space":
        if frame_ref == "space":
            return jac
        elif frame_ref == "body":
            Twb_q = poe_fk(q, "space", "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(np.linalg.inv(Tsb_q)) @ jac
        elif frame_ref == "world":
            return adjoint(Tws) @ jac
    elif frame_in == "body":
        if frame_ref == "body":
            return jac
        elif frame_ref == "space":
            Twb_q = poe_fk(q, "space", "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(Tsb_q) @ jac
        elif frame_ref == "world":
            Twb_q = poe_fk(q, "space", "world")
            return adjoint(Twb_q) @ jac
    return jac  # default is the world frame

def poe_ik(q0, Td, frame_in = "space", frame_ref = "world", max_iter = 100, error_tol = 1e-5, avoid_box = False):
    q = np.copy(q0)
    
    # CBF Parameters for Box Avoidance
    alpha = 5.0
    s_safe = 0.015  # 1.5 cm περιθώριο ασφαλείας
    box_x_min = 0.125 - 0.06
    box_x_max = 0.125 + 0.06
    box_y_min = 0.125 - 0.055
    box_y_max = 0.125 + 0.055
    box_z_max = 0.06
    
    for _ in range(max_iter):
        if frame_ref == "world" or frame_ref == "space":
            Twref = poe_fk(q, frame_in, frame_ref)
            error = Log(Td @ np.linalg.inv(Twref))
            # error = (lambda e: np.block([e[3:], e[:3]]))(np.array(pin.log(Td @ np.linalg.inv(Twref))))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(error) < error_tol:
            break
        jac_pinv = np.linalg.pinv(jacobian(q, frame_in, frame_ref))
        dq = jac_pinv @ error
        
        # Εφαρμογή CBF για την αποφυγή των τοιχωμάτων του κουτιού
        if avoid_box:
            Twref = poe_fk(q, frame_in, "world")
            p = Twref[:3, 3]
            
            # Ελέγχουμε αν το EE είναι κάτω από το χείλος του κουτιού (+ ένα μικρό περιθώριο)
            if p[2] < box_z_max + s_safe:
                # Ορίζουμε τα 4 τοιχώματα (απόσταση προς τα ΜΕΣΑ του κουτιού, + κανονικό διάνυσμα)
                walls = [
                    (p[0] - box_x_min, np.array([1, 0, 0])),
                    (box_x_max - p[0], np.array([-1, 0, 0])),
                    (p[1] - box_y_min, np.array([0, 1, 0])),
                    (box_y_max - p[1], np.array([0, -1, 0]))
                ]
                
                # Βρίσκουμε το πιο κοντινό τοίχωμα
                s, grad_s = min(walls, key=lambda w: w[0])
                
                # Υπολογισμός Numerical Position Jacobian (Jp) για ασφάλεια (αγνοούμε διαφορές convention)
                Jp = np.zeros((3, len(q)))
                eps = 1e-5
                for i in range(len(q)):
                    q_eps = np.copy(q)
                    q_eps[i] += eps
                    p_eps = poe_fk(q_eps, frame_in, "world")[:3, 3]
                    Jp[:, i] = (p_eps - p) / eps
                
                # CBF constraint: grad_s^T * Jp * dq >= -alpha * (s - s_safe)
                s_dot_nom = grad_s.T @ Jp @ dq
                required_s_dot = -alpha * (s - s_safe)
                
                # Αν η ονομαστική ταχύτητα παραβιάζει το constraint, προβάλλουμε (QP/Active Set update)
                if s_dot_nom < required_s_dot:
                    A = grad_s.T @ Jp  # (1, N)
                    # Δq_safe = Δq_nom + lambda * A^T
                    # A * (dq + lambda * A^T) = required
                    # lambda * (A * A^T) = required - A*dq
                    A_norm_sq = np.dot(A, A)
                    if A_norm_sq > 1e-6:
                        lam = (required_s_dot - s_dot_nom) / A_norm_sq
                        dq = dq + lam * A

        q = q + dq
        q = np.minimum(np.maximum(q, q_lb), q_ub)
    return q
