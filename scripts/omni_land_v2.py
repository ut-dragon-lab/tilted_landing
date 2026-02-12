#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub
from aerial_robot_planning.trajs import BaseTraj

# CONFIGURAÇÕES DE MISSÃO
APPROACH_SPEED = 0.3  
DESCENT_SPEED = 0.10
PRESS_THRESHOLD = 0.18
PRESS_TARGET_DIST = 0.14
PRESS_DURATION = 2.0 
WAIT_FOR_STABILITY = 1.0 

class TagObserver:
    """
    Rastreador de Tag usando Filtro de Kalman para uma base estática.
    Estado: [x, y, z] (posição fixa).
    """
    def __init__(self, bundle_frame="land_mark_bundle", small_tag_frame="small_tag"):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bundle_frame = bundle_frame
        self.small_tag_frame = small_tag_frame
        
        # Como a tag não se move, o estado é apenas [x, y, z]
        self.state = np.zeros(3)
        self.P = np.eye(3) * 1.0  # Incerteza inicial
        self.Q = np.eye(3) * 1e-5 # Ruído de processo quase nulo (tag estática)
        self.R = np.eye(3) * 0.1 # Ruído de medição da câmera
        
        self.last_valid_quat = [0, 0, 0, 1]
        self.initialized = False

    def update(self, measurement):
        # Matriz de Observação Identidade (medimos x, y, z diretamente)
        H = np.eye(3)
        
        # Ganho de Kalman
        S = self.P + self.R
        K = np.dot(self.P, np.linalg.inv(S))
        
        # Atualização (Inovação)
        y = measurement - self.state
        self.state = self.state + np.dot(K, y)
        self.P = (np.eye(3) - K) @ self.P + self.Q

    def get_smoothed_transform(self):
        target = self._check_frame(self.bundle_frame)
        if target is None:
            target = self._check_frame(self.small_tag_frame)
        if target is None: return None

        raw_pos, raw_quat = target

        if not self.initialized:
            self.state = raw_pos
            self.last_valid_quat = raw_quat
            self.initialized = True
            return self.state, raw_quat

        # Atualiza o filtro com a nova medição
        self.update(raw_pos)
        self.last_valid_quat = raw_quat
        
        return self.state, self.last_valid_quat

    def _check_frame(self, frame_name):
        try:
            trans = self.tf_buffer.lookup_transform("world", frame_name, rospy.Time(0), rospy.Duration(0.01))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            return pos, quat
        except: return None

class LandingTraj(BaseTraj):
    def __init__(self, tag_observer, robot_name):
        super().__init__()
        self.tag_obs = tag_observer
        self.robot_name = robot_name
        self.phase = "APPROACH" 
        self.is_touchdown = False
        
        self.start_xyz = None
        self.phase_start_time = None
        self.curr_odom = None
        rospy.Subscriber(f"/{robot_name}/uav/cog/odom", Odometry, self._odom_cb)

    def _odom_cb(self, msg):
        self.curr_odom = msg

    def get_3d_pt(self, t_cal):
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None or self.curr_odom is None:
            return np.zeros(9)

        t_pos, t_quat = tag_data
        r_mat = R.from_quat(t_quat)
        normal_vec = r_mat.apply([0, 0, 1])
        
        if self.phase_start_time is None:
            self.phase_start_time = rospy.Time.now()
            self.start_xyz = np.array([self.curr_odom.pose.pose.position.x, 
                                       self.curr_odom.pose.pose.position.y, 
                                       self.curr_odom.pose.pose.position.z])

        # Tempo real decorrido desde o início da fase
        elapsed = (rospy.Time.now() - self.phase_start_time).to_sec()
        
        # Tempo projetado para o horizonte do MPC
        t_lookahead = elapsed + t_cal
        
         # FASE 1: NAVEGAÇÃO OMNI ATÉ 1m DA TAG
        if self.phase == "APPROACH":
            target_xyz = t_pos + (normal_vec * 0.9)
            diff = target_xyz - self.start_xyz
            dist_total = np.linalg.norm(diff)
            duration = dist_total / APPROACH_SPEED
            
            if elapsed < duration:
                direction = diff / dist_total
                pos = self.start_xyz + (direction * APPROACH_SPEED * elapsed)
                vel = direction * APPROACH_SPEED
            else:
                pos = target_xyz
                vel = np.zeros(3)
                if elapsed > (duration + WAIT_FOR_STABILITY):
                    self.phase = "DESCEND"
                    self.phase_start_time = rospy.Time.now()
            return np.concatenate([pos, vel, np.zeros(3)])
        
        # --- FASE 2: DESCIDA COM PROJEÇÃO (Sua ideia original) ---
        if self.phase == "DESCEND":
            # Usa t_lookahead para projetar a descida no horizonte
            target_dist = max(PRESS_TARGET_DIST, 0.9 - (DESCENT_SPEED * t_lookahead))
            pos = t_pos + (normal_vec * target_dist)
            vel = -normal_vec * DESCENT_SPEED
            
            # Verificação de contato usa a posição ATUAL do robô
            curr_p = np.array([self.curr_odom.pose.pose.position.x,
                               self.curr_odom.pose.pose.position.y,
                               self.curr_odom.pose.pose.position.z])
            z_dist = np.dot(curr_p - t_pos, normal_vec)
            
            if z_dist < PRESS_THRESHOLD:
                rospy.loginfo(">>> CONTATO DETECTADO <<<")
                self.phase = "PRESS"
                self.phase_start_time = rospy.Time.now()
        
        elif self.phase == "PRESS":
            pos = t_pos + (normal_vec * PRESS_TARGET_DIST)
            vel = np.zeros(3)
            if elapsed > PRESS_DURATION:
                self.is_touchdown = True
        
        return np.concatenate([pos, vel, np.zeros(3)])


    def get_3d_orientation(self, t_cal):
        # Sempre copia a atitude da tag para garantir pouso paralelo à rampa
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None: return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _, t_quat = tag_data
        return [t_quat[3], t_quat[0], t_quat[1], t_quat[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def check_finished(self, t_elapsed):
        return self.is_touchdown

    def get_frame_id(self): return "world"
    def get_child_frame_id(self): return "cog"

if __name__ == "__main__":
    rospy.init_node('beetle_omni_visual_landing')
    robot_name = "beetle1"
    
    observer = TagObserver()
    landing_traj = LandingTraj(observer, robot_name)
    mpc_node = MPCTrajPtPub(robot_name, landing_traj)
    
    halt_pub = rospy.Publisher(f"/{robot_name}/teleop_command/halt", Empty, queue_size=1)
    
    rospy.loginfo("Iniciando Missão: Aproximação via Omni-Nav + KF")
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if landing_traj.is_touchdown:
            rospy.logwarn("Touchdown! Desligando motores.")
            for _ in range(10): halt_pub.publish(Empty())
            break
        rate.sleep()
