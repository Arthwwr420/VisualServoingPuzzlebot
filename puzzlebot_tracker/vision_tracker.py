#!/usr/bin/env python3
"""
vision_tracker.py  —  Puzzlebot Visual Servoing
================================================
Nodo ROS2 de percepción. Suscribe /image_raw, detecta el objetivo
(ArUco marker o blob HSV), calcula el error normalizado y la distancia
estimada, y publica en /detection/error (geometry_msgs/Point).

  point.x  = error horizontal normalizado [-1, +1]   (+ = derecha)
  point.y  = error vertical   normalizado [-1, +1]   (+ = abajo)
  point.z  = distancia estimada en metros             (-1 = perdido)

Autor:  Puzzlebot Team
ROS2:   Humble / Iron / Jazzy
OpenCV: 4.x
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import cv2
import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
#  Filtro de media móvil exponencial (EMA) para suavizar detecciones
# ─────────────────────────────────────────────────────────────────────────────
class EMAFilter:
    def __init__(self, alpha: float = 0.35, size: int = 3):
        self.alpha = alpha
        self._state = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = value.copy()
        else:
            self._state = self.alpha * value + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self):
        self._state = None


# ─────────────────────────────────────────────────────────────────────────────
#  Nodo principal
# ─────────────────────────────────────────────────────────────────────────────
class VisionTracker(Node):

    def __init__(self):
        super().__init__('vision_tracker')

        # ── Parámetros declarados (modificables desde params.yaml o CLI) ──
        self.declare_parameter('detection_mode', 'hybrid')
        #   'aruco'  → solo marcadores ArUco
        #   'hsv'    → solo blob de color
        #   'hybrid' → ArUco primario, HSV como respaldo

        self.declare_parameter('aruco_dict_id', 0)   # DICT_4X4_50
        self.declare_parameter('aruco_target_id', 0) # ID del marcador a seguir
        self.declare_parameter('aruco_marker_size', 0.10)  # metros (lado)

        self.declare_parameter('hsv_lower', [35, 60, 60])   # Verde (H,S,V)
        self.declare_parameter('hsv_upper', [85, 255, 255])
        self.declare_parameter('hsv_min_area', 800)          # px²

        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fx', 554.0)  # Focal length px (horizontal)
        self.declare_parameter('camera_fy', 554.0)

        self.declare_parameter('ema_alpha', 0.35)    # Suavizado (0=máx suave, 1=sin filtro)
        self.declare_parameter('lost_frames_max', 12)
        self.declare_parameter('publish_debug', True)

        # ── Leer parámetros ──
        self.mode         = self.get_parameter('detection_mode').value
        self.target_id    = self.get_parameter('aruco_target_id').value
        self.marker_size  = self.get_parameter('aruco_marker_size').value
        self.img_w        = self.get_parameter('image_width').value
        self.img_h        = self.get_parameter('image_height').value
        self.fx           = self.get_parameter('camera_fx').value
        self.fy           = self.get_parameter('camera_fy').value
        self.lost_max     = self.get_parameter('lost_frames_max').value
        self.pub_debug    = self.get_parameter('publish_debug').value

        # ── ArUco Detector (OpenCV 4.7+) ──
        if self.mode in ('aruco', 'hybrid'):
            dict_id = self.get_parameter('aruco_dict_id').value
            self.aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
            params            = cv2.aruco.DetectorParameters()
        # Mejorar detección en condiciones variables de iluminación
            params.adaptiveThreshWinSizeMin  = 3
            params.adaptiveThreshWinSizeMax  = 23
            params.adaptiveThreshWinSizeStep = 10
            params.minMarkerPerimeterRate    = 0.03
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        else:
            self.aruco_detector = None
            self.get_logger().info(f'Modo {self.mode}: ArUco desactivado')

        # ── HSV bounds ──
        lo = self.get_parameter('hsv_lower').value
        hi = self.get_parameter('hsv_upper').value
        self.hsv_lower = np.array(lo, dtype=np.uint8)
        self.hsv_upper = np.array(hi, dtype=np.uint8)
        self.hsv_min_area = self.get_parameter('hsv_min_area').value

        # ── Kernel morfológico para limpieza de máscara ──
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # ── Estado ──
        self.bridge       = CvBridge()
        self.ema          = EMAFilter(alpha=self.get_parameter('ema_alpha').value)
        self.lost_frames  = 0
        self.detected_prev = False

        # ── ROS2 I/O ──
        qos_sensor = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT)

        self.sub_img = self.create_subscription(
            Image, '/image_raw', self._image_cb, qos_sensor)

        self.pub_error   = self.create_publisher(Point, '/detection/error', 10)
        self.pub_detected = self.create_publisher(Bool,  '/detection/active', 10)

        if self.pub_debug:
            self.pub_dbg = self.create_publisher(Image, '/detection/debug', 5)

        self.get_logger().info(
            f'VisionTracker iniciado | modo={self.mode} | '
            f'aruco_id={self.target_id} | res={self.img_w}x{self.img_h}')

    # ─────────────────────────────────────────────────────────────────────────
    #  Callback principal
    # ─────────────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        detected = False
        cx = cy = 0.0
        dist = -1.0

        # ── Detección primaria: ArUco ──
        if self.mode in ('aruco', 'hybrid'):
            detected, cx, cy, dist = self._detect_aruco(frame)

        # ── Detección secundaria: HSV blob ──
        if not detected and self.mode in ('hsv', 'hybrid'):
            detected, cx, cy, dist = self._detect_hsv(frame)

        # ── Calcular y publicar error ──
        error_msg = Point()

        if detected:
            self.lost_frames = 0

            # Normalizar a [-1, +1]
            ex = (cx - self.img_w * 0.5) / (self.img_w  * 0.5)
            ey = (cy - self.img_h * 0.5) / (self.img_h * 0.5)

            raw = np.array([ex, ey, dist])
            filtered = self.ema.update(raw)

            error_msg.x = float(np.clip(filtered[0], -1.0, 1.0))
            error_msg.y = float(np.clip(filtered[1], -1.0, 1.0))
            error_msg.z = float(max(0.0, filtered[2]))

        else:
            self.lost_frames += 1
            if self.lost_frames > self.lost_max:
                self.ema.reset()
                error_msg.z = -1.0  # Señal de objetivo perdido

        self.pub_error.publish(error_msg)

        detected_msg = Bool()
        detected_msg.data = detected
        self.pub_detected.publish(detected_msg)

        # ── Imagen de debug ──
        if self.pub_debug:
            self._publish_debug(frame, detected, cx, cy, error_msg)

    # ─────────────────────────────────────────────────────────────────────────
    #  Detección ArUco
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_aruco(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ecualización adaptativa para iluminación variable
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return False, 0.0, 0.0, -1.0

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_id:
                continue

            c  = corners[i][0]          # shape (4, 2)
            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))

            # Distancia por tamaño aparente:  d = (M * fx) / w_px
            # donde M = tamaño real del marcador (metros), w_px = ancho en píxeles
            w_px = np.linalg.norm(c[0] - c[1])
            dist = (self.marker_size * self.fx) / w_px if w_px > 0 else -1.0

            # Dibujar marcador en frame de debug
            cv2.aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[marker_id]]))

            return True, cx, cy, float(dist)

        return False, 0.0, 0.0, -1.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Detección HSV (blob de color)
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_hsv(self, frame: np.ndarray):
        # Desenfoque Gaussiano para reducir ruido de sensor
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Limpieza morfológica: elimina ruido y cierra huecos
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  self.morph_kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, 0.0, 0.0, -1.0

        # Tomar el contorno más grande
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)

        if area < self.hsv_min_area:
            return False, 0.0, 0.0, -1.0

        M  = cv2.moments(best)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Estimación de distancia por área (modelo pinhole inverso)
        #   area ≈ (A_real * fx²) / d²  → d ≈ sqrt(A_ref / area) * d_ref
        # Calibrar A_ref y d_ref experimentalmente con tu objeto.
        A_ref, d_ref = 40000.0, 0.30   # área en px² a 30 cm de distancia
        dist = d_ref * np.sqrt(A_ref / area)

        cv2.drawContours(frame, [best], -1, (0, 255, 0), 2)

        return True, cx, cy, float(dist)

    # ─────────────────────────────────────────────────────────────────────────
    #  Debug overlay
    # ─────────────────────────────────────────────────────────────────────────
    def _publish_debug(self, frame, detected, cx, cy, err: Point):
        dbg = frame  # ya modificado in-place por los detectores

        cx_img = self.img_w // 2
        cy_img = self.img_h // 2

        # Líneas de guía (crosshair)
        cv2.line(dbg, (cx_img, 0), (cx_img, self.img_h), (0, 0, 200), 1)
        cv2.line(dbg, (0, cy_img), (self.img_w, cy_img), (0, 0, 200), 1)

        if detected:
            cv2.circle(dbg, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            cv2.line(dbg, (cx_img, cy_img), (int(cx), int(cy)), (0, 255, 255), 2)
            label = f'ex={err.x:+.2f}  ey={err.y:+.2f}  d={err.z:.2f}m'
            cv2.putText(dbg, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(dbg, 'SEARCHING...', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = VisionTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
