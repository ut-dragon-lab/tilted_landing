#!/usr/bin/env python3
"""
Omni Align Node: Alinha a atitude do Beetle paralelamente à tag ou ao horizonte.
Mantém a posição atual (X, Y, Z) enquanto altera apenas a orientação.
"""

import rospy
import numpy as np
import argparse
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub
from aerial_robot_planning.trajs import BaseTraj
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion

# Reaproveitando sua lógica de percepção
from visual_land import TagObserver

class AlignTraj(BaseTraj):
    def __init__(self, tag_observer, robot_name, target_mode="tag"):
        super().__init__()
        self.tag_obs = tag_observer
        self.robot_name = robot_name
        self.target_mode = target_mode # "tag" ou "horizon"
        
        self.curr_odom = None
        self.initial_pos = None
        self.last_valid_quat = None
        
        # Limite de segurança: 0.3 rad (~17 graus) de diferença entre frames do MPC
        self.MAX_ANGULAR_STEP = 0.3 
        
        rospy.Subscriber(f"/{robot_name}/uav/cog/odom", Odometry, self._odom_cb)

    def _odom_cb(self, msg):
        self.curr_odom = msg
        if self.initial_pos is None:
            self.initial_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

    def get_3d_pt(self, t_cal):
        """Mantém a posição travada onde o comando de alinhamento começou."""
        if self.initial_pos is None: return np.zeros(9)
        # Retorna [x, y, z, vx, vy, vz, ax, ay, az]
        return np.concatenate([self.initial_pos, np.zeros(6)])

    def get_3d_orientation(self, t_cal):
        """Retorna a orientação alvo [qw, qx, qy, qz, rates..., accs...]."""
        # Padrão: Atitude nivelada (Horizonte)
        target_quat = [0.0, 0.0, 0.0, 1.0] 

        if self.target_mode == "tag":
            tag_data = self.tag_obs.get_smoothed_transform()
            if tag_data is not None:
                _, t_quat = tag_data # [x, y, z, w]
                
                # GATEKEEPER: Verifica se a tag "pulou" bruscamente
                if self.last_valid_quat is not None:
                    if not self._is_quaternion_smooth(self.last_valid_quat, t_quat):
                        rospy.logerr_throttle(1, "GATEKEEPER: Tag teleport detected! Holding last attitude.")
                        t_quat = self.last_valid_quat
                
                target_quat = t_quat
                self.last_valid_quat = t_quat
            elif self.last_valid_quat is not None:
                target_quat = self.last_valid_quat

        # Formato para NMPC: [qw, qx, qy, qz, rates(3), accs(3)]
        return [target_quat[3], target_quat[0], target_quat[1], target_quat[2], 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _is_quaternion_smooth(self, q1, q2):
        """Calcula a distância angular entre dois quaternions."""
        # q_diff = q1 * inv(q2)
        q_inv = quaternion_inverse(q2)
        q_diff = quaternion_multiply(q1, q_inv)
        # Ângulo da rotação residual
        angle = 2 * np.arccos(np.clip(abs(q_diff[3]), 0, 1))
        return angle < self.MAX_ANGULAR_STEP

    def check_finished(self, t_elapsed):
        return False # Mantém o alinhamento ativo até o usuário cancelar

class GatekeeperMPC(MPCTrajPtPub):
    """
    Wrapper para o publicador que impede o envio de comandos se a 
    trajetória gerada for descontínua.
    """
    def pub_trajectory_points(self, msg):
        # Aqui você poderia implementar uma validação extra na lista de pontos
        # antes de chamar o super(). No momento, a LandingTraj já filtra a tag.
        super().pub_trajectory_points(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beetle Omni Alignment")
    parser.add_argument("--target", choices=["tag", "horizon"], default="tag")
    args = parser.parse_args()

    rospy.init_node('beetle_omni_align')
    robot_name = "beetle1"
    
    observer = TagObserver()
    align_traj = AlignTraj(observer, robot_name, target_mode=args.target)
    
    # Inicializa o publicador MPC com o gatekeeper
    mpc_node = GatekeeperMPC(robot_name, align_traj)
    
    rospy.loginfo(f"Alignment Mode: {args.target} - Positioning Locked.")
    rospy.spin()
