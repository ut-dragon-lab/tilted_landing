import numpy as np
import rosbag
from scipy.optimize import minimize
import yaml

# --- Configurações do Usuário ---
calibration_file = 'data/calibration.bag'
bundle_name = 'my_bundle'
master_id = 0
topic_name = '/tag_detections'

# --- Funções Matemáticas Locais ---

def quat2rotmat(q):
    """Cria uma matriz de rotação ATIVA a partir de um quat [w, x, y, z]"""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x**2+y**2)]
    ])

def rotmat2quat(R):
    """Converte matriz de rotação para quat [w, x, y, z]"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R(2,1) - R(1,2)) / S # Note: ajustado para índices 0-based
    # Simplificado usando lógica robusta:
    q = np.empty((4,))
    t = np.trace(R)
    if t > 0:
        t = np.sqrt(t + 1)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]: i = 1
        if R[2, 2] > R[i, i]: i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[i+1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j+1] = (R[j, i] + R[i, j]) * t
        q[k+1] = (R[k, i] + R[i, k]) * t
    return q

def invertT(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ p
    return T_inv

def createT(p, q):
    T = np.eye(4)
    T[0:3, 0:3] = quat2rotmat(q)
    T[0:3, 3] = p
    return T

# --- Carregamento dos Dados ---

bag = rosbag.Bag(calibration_file)
other_ids = []
other_sizes = []
rel_p = {} # Dicionário para armazenar listas de posições
rel_q = {} # Dicionário para armazenar listas de quaterniões
master_size = None

print(f"Lendo bag: {calibration_file}...")

for topic, msg, t in bag.read_messages(topics=[topic_name]):
    detections = msg.detections
    
    # Encontrar o master na detecção atual
    mi = next((i for i, d in enumerate(detections) if d.id[0] == master_id), None)
    
    if mi is None:
        continue
    
    # Pose do Master (ROS usa [x,y,z,w], convertendo para [w,x,y,z])
    m_pose = detections[mi].pose.pose.pose
    p_cm = np.array([m_pose.position.x, m_pose.position.y, m_pose.position.z])
    q_cm = np.array([m_pose.orientation.w, m_pose.orientation.x, m_pose.orientation.y, m_pose.orientation.z])
    T_cm = createT(p_cm, q_cm)
    
    if master_size is None:
        master_size = detections[mi].size[0]

    for j, det in enumerate(detections):
        if j == mi or len(det.id) > 1:
            continue
            
        tag_id = det.id[0]
        if tag_id not in other_ids:
            other_ids.append(tag_id)
            other_sizes.append(det.size[0])
            rel_p[tag_id] = []
            rel_q[tag_id] = []
            
        # Pose da tag j
        p_cj = np.array([det.pose.pose.pose.position.x, det.pose.pose.pose.position.y, det.pose.pose.pose.position.z])
        q_cj = np.array([det.pose.pose.pose.orientation.w, det.pose.pose.pose.orientation.x, det.pose.pose.pose.orientation.y, det.pose.pose.pose.orientation.z])
        T_cj = createT(p_cj, q_cj)
        
        # T_mj = T_mc * T_cj
        T_mj = invertT(T_cm) @ T_cj
        
        rel_p[tag_id].append(T_mj[0:3, 3])
        rel_q[tag_id].append(rotmat2quat(T_mj[0:3, 0:3]))

bag.close()

# --- Cálculos Estatísticos ---

def geometric_median_cost(x, points):
    return np.sum(np.linalg.norm(points - x, axis=1))

rel_p_median = {}
rel_q_mean = {}

for tid in other_ids:
    # Mediana Geométrica (Posição)
    points = np.array(rel_p[tid])
    p0 = np.mean(points, axis=0)
    res = minimize(geometric_median_cost, p0, args=(points,), method='Nelder-Mead')
    rel_p_median[tid] = res.x
    
    # Média de Quaterniões (Orientação - Método Autovalores)
    Q = np.array(rel_q[tid]) # Shape (N, 4)
    # No MATLAB era Q*Q', aqui fazemos a matriz de covariância
    M = Q.T @ Q
    eigenvalues, eigenvectors = np.linalg.eig(M)
    q_mean = eigenvectors[:, np.argmax(eigenvalues)]
    if q_mean[0] < 0: q_mean *= -1
    rel_q_mean[tid] = q_mean

# --- Saída em Formato YAML ---

print("\n--- Copie o conteúdo abaixo para o seu tags.yaml ---\n")
print(f"tag_bundles:")
print(f"  [")
print(f"    {{")
print(f"      name: '{bundle_name}',")
print(f"      layout:")
print(f"        [")

# Master Tag
print(f"          {{id: {master_id}, size: {master_size:.2f}, x: 0.0000, y: 0.0000, z: 0.0000, qw: 1.0000, qx: 0.0000, qy: 0.0000, qz: 0.0000}},")

# Outras Tags
for i, tid in enumerate(other_ids):
    p = rel_p_median[tid]
    q = rel_q_mean[tid]
    comma = "," if i < len(other_ids) - 1 else ""
    print(f"          {{id: {tid}, size: {other_sizes[i]:.2f}, x: {p[0]:.4f}, y: {p[1]:.4f}, z: {p[2]:.4f}, qw: {q[0]:.4f}, qx: {q[1]:.4f}, qy: {q[2]:.4f}, qz: {q[3]:.4f}}}{comma}")

print("        ]")
print("    }")
print("  ]")
