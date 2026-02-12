#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry

class Calibrator:
    def __init__(self, bundle_frame="land_mark_bundle", small_tag_frame="small_tag"):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bundle_frame = bundle_frame
        self.small_tag_frame = small_tag_frame
        
        # Filtro de Kalman (copiado do seu script para manter consistência)
        self.state = np.zeros(3)
        self.P = np.eye(3) * 1.0
        self.Q = np.eye(3) * 1e-5
        self.R = np.eye(3) * 0.1
        self.initialized = False

        self.curr_odom = None
        rospy.Subscriber("/beetle1/uav/cog/odom", Odometry, self._odom_cb)

    def _odom_cb(self, msg):
        self.curr_odom = msg

    def update_kf(self, measurement):
        H = np.eye(3)
        S = self.P + self.R
        K = np.dot(self.P, np.linalg.inv(S))
        y = measurement - self.state
        self.state = self.state + np.dot(K, y)
        self.P = (np.eye(3) - K) @ self.P + self.Q

    def get_tag_data(self):
        # Tenta primeiro o bundle, depois a tag pequena
        target = self._check_frame(self.bundle_frame)
        if target is None:
            target = self._check_frame(self.small_tag_frame)
        
        if target is None: return None
        
        raw_pos, raw_quat = target
        if not self.initialized:
            self.state = raw_pos
            self.initialized = True
        else:
            self.update_kf(raw_pos)
        
        return self.state, raw_quat

    def _check_frame(self, frame_name):
        try:
            # Tempo 0 para pegar a transformação mais recente disponível
            trans = self.tf_buffer.lookup_transform("world", frame_name, rospy.Time(0), rospy.Duration(0.01))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            return pos, quat
        except:
            return None

    def run(self):
        rate = rospy.Rate(10) # 10Hz é suficiente para leitura visual
        rospy.loginfo("--- AGUARDANDO DADOS (TAG E ODOMETRIA) ---")
        
        while not rospy.is_shutdown():
            tag_data = self.get_tag_data()
            
            if tag_data is not None and self.curr_odom is not None:
                t_pos, t_quat = tag_data
                r_mat = R.from_quat(t_quat)
                normal_vec = r_mat.apply([0, 0, 1])
                
                # Posição atual do Beetle
                curr_p = np.array([self.curr_odom.pose.pose.position.x,
                                   self.curr_odom.pose.pose.position.y,
                                   self.curr_odom.pose.pose.position.z])
                
                # Cálculo da distância Z perpendicular (Z_dist)
                z_dist = np.dot(curr_p - t_pos, normal_vec)
                
                # Limpa a tela e imprime para facilitar a leitura
                print(f"\r[CALIBRAÇÃO] Z_DIST ATUAL: {z_dist:.4f} m", end="")
            else:
                if tag_data is None:
                    print("\r[AVISO] Tag não detectada...", end="")
                elif self.curr_odom is None:
                    print("\r[AVISO] Odometria não recebida...", end="")
            
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('threshold_calibrator')
    calib = Calibrator()
    try:
        calib.run()
    except rospy.ROSInterruptException:
        pass
