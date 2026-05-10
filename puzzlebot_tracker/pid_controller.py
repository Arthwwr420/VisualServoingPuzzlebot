#!/usr/bin/env python3
"""
pid_controller.py  —  Puzzlebot Visual Servoing Controller
===========================================================
Nodo ROS2 de control. Implementa dos lazos PID independientes:

  LAZO ANGULAR:  error_x (posición horizontal del objetivo)  → ω (rad/s)
  LAZO LINEAL:   error_d (distancia estimada vs deseada)     → v (m/s)

Características:
  • Anti-windup por clamping de integral
  • Filtro paso-bajo en la derivada (evita spike noise)
  • Dead-zone por eje (evita micro-vibraciones en equilibrio)
  • Gain scheduling: reduce velocidad lineal si no está centrado
  • Watchdog: detiene el robot si se pierde el objetivo
  • State machine: SEARCHING → TRACKING → LOST

Publicaciones:
  /cmd_vel  (geometry_msgs/Twist)

Subscripciones:
  /detection/error  (geometry_msgs/Point)
  /detection/active (std_msgs/Bool)

Autor:  Puzzlebot Team
ROS2:   Humble / Iron / Jazzy
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, String

import numpy as np
from enum import Enum, auto
import time


# ─────────────────────────────────────────────────────────────────────────────
#  Estados del robot
# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    SEARCHING = auto()   # Girando en búsqueda del objetivo
    TRACKING  = auto()   # Seguimiento activo con PID
    LOST      = auto()   # Objetivo perdido, detenido


# ─────────────────────────────────────────────────────────────────────────────
#  Controlador PID con filtro de derivada y anti-windup
# ─────────────────────────────────────────────────────────────────────────────
class PIDController:
    """
    Implementación discreta del PID con:
      - Deadzone:     ignora errores menores a un umbral
      - Anti-windup:  clamping de integral en [-max_i, +max_i]
      - Derivada LPF: α * D_prev + (1-α) * D_raw
      - Salida acotada a [-max_out, +max_out]
    """

    def __init__(self,
                 kp: float, ki: float, kd: float,
                 max_output: float,
                 max_integral: float,
                 deadzone: float = 0.0,
                 deriv_alpha: float = 0.65):

        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_output   = max_output
        self.max_integral = max_integral
        self.deadzone     = deadzone
        self.d_alpha      = deriv_alpha

        self._integral  = 0.0
        self._prev_err  = 0.0
        self._deriv_flt = 0.0

    def reset(self):
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._deriv_flt = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Calcula salida del PID dado el error y el intervalo de tiempo dt (s)."""
        if abs(error) < self.deadzone:
            error = 0.0

        # Integral con anti-windup
        self._integral += error * dt
        self._integral  = float(np.clip(self._integral,
                                        -self.max_integral, self.max_integral))

        # Derivada filtrada (evita amplificación de ruido de cuantización)
        raw_deriv       = (error - self._prev_err) / max(dt, 1e-4)
        self._deriv_flt = (self.d_alpha * self._deriv_flt +
                           (1.0 - self.d_alpha) * raw_deriv)
        self._prev_err  = error

        output = (self.kp * error +
                  self.ki * self._integral +
                  self.kd * self._deriv_flt)

        return float(np.clip(output, -self.max_output, self.max_output))

    def set_gains(self, kp: float, ki: float, kd: float):
        """Permite ajuste de ganancias en caliente (gain scheduling)."""
        self.kp, self.ki, self.kd = kp, ki, kd


# ─────────────────────────────────────────────────────────────────────────────
#  Nodo controlador
# ─────────────────────────────────────────────────────────────────────────────
class VisualServoController(Node):

    def __init__(self):
        super().__init__('visual_servo_controller')

        # ── Parámetros ──
        self.declare_parameter('desired_distance', 0.50)   # m
        self.declare_parameter('v_max',   0.25)            # m/s
        self.declare_parameter('omega_max', 1.20)          # rad/s

        # PID Angular (ex → ω)
        self.declare_parameter('ang_kp', 0.65)
        self.declare_parameter('ang_ki', 0.008)
        self.declare_parameter('ang_kd', 0.10)

        # PID Lineal (dist_error → v)
        self.declare_parameter('lin_kp', 0.45)
        self.declare_parameter('lin_ki', 0.005)
        self.declare_parameter('lin_kd', 0.06)

        self.declare_parameter('deadzone_angular', 0.04)   # normalizado [-1,1]
        self.declare_parameter('deadzone_linear',  0.04)   # metros
        self.declare_parameter('lost_timeout',     1.5)    # segundos
        self.declare_parameter('search_omega',     0.35)   # rad/s al buscar
        self.declare_parameter('centering_gain',   1.2)    # factor de atenuación lineal

        # ── Leer valores ──
        self.d_goal    = self.get_parameter('desired_distance').value
        self.v_max     = self.get_parameter('v_max').value
        self.w_max     = self.get_parameter('omega_max').value
        self.timeout   = self.get_parameter('lost_timeout').value
        self.search_w  = self.get_parameter('search_omega').value
        self.c_gain    = self.get_parameter('centering_gain').value

        # ── Instanciar PIDs ──
        self.ang_pid = PIDController(
            kp=self.get_parameter('ang_kp').value,
            ki=self.get_parameter('ang_ki').value,
            kd=self.get_parameter('ang_kd').value,
            max_output=self.w_max,
            max_integral=0.40,
            deadzone=self.get_parameter('deadzone_angular').value,
            deriv_alpha=0.65
        )
        self.lin_pid = PIDController(
            kp=self.get_parameter('lin_kp').value,
            ki=self.get_parameter('lin_ki').value,
            kd=self.get_parameter('lin_kd').value,
            max_output=self.v_max,
            max_integral=0.25,
            deadzone=self.get_parameter('deadzone_linear').value,
            deriv_alpha=0.70
        )

        # ── Estado ──
        self.state              = State.SEARCHING
        self.last_detect_time   = None
        self.last_cb_time       = self.get_clock().now()
        self._search_dir        = 1.0  # +1 derecha, -1 izquierda

        # ── ROS2 I/O ──
        self.sub_err = self.create_subscription(
            Point, '/detection/error', self._error_cb, 10)
        self.sub_det = self.create_subscription(
            Bool, '/detection/active', self._detected_cb, 10)

        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel',         10)
        self.pub_state = self.create_publisher(String, '/controller/state', 10)

        # Watchdog a 10 Hz
        self.watchdog_timer = self.create_timer(0.10, self._watchdog_cb)

        self.get_logger().info(
            f'VisualServoController listo | d_goal={self.d_goal}m | '
            f'v_max={self.v_max} m/s | ω_max={self.w_max} rad/s')

    # ─────────────────────────────────────────────────────────────────────────
    #  Callback de error de visión
    # ─────────────────────────────────────────────────────────────────────────
    def _error_cb(self, msg: Point):
        now = self.get_clock().now()
        dt  = (now - self.last_cb_time).nanoseconds * 1e-9
        self.last_cb_time = now
        dt  = float(np.clip(dt, 0.005, 0.200))  # 5 ms – 200 ms

        # ── Objetivo perdido ──
        if msg.z < 0.0:
            if self.state == State.TRACKING:
                self._transition(State.LOST)
            self._stop()
            self.ang_pid.reset()
            self.lin_pid.reset()
            return

        # ── Objetivo detectado ──
        self.last_detect_time = now
        if self.state != State.TRACKING:
            self._transition(State.TRACKING)

        # ── Control angular: centrar objetivo horizontalmente ──
        # msg.x > 0 → objetivo a la derecha → girar derecha (ω negativo en ROS)
        omega = -self.ang_pid.compute(msg.x, dt)

        # ── Control lineal: mantener distancia deseada ──
        # dist_error > 0 → robot lejos → avanzar (v positiva)
        dist_err = msg.z - self.d_goal
        v_raw    = self.lin_pid.compute(dist_err, dt)

        # ── Gain scheduling: reducir v cuando no está centrado ──
        # Si el objetivo no está centrado, prioriza girar antes de avanzar.
        centering_factor = max(0.0, 1.0 - self.c_gain * abs(msg.x))
        v = v_raw * centering_factor

        self._send_cmd(v, omega)

        self.get_logger().debug(
            f'[TRACKING] ex={msg.x:+.3f} dist={msg.z:.3f}m '
            f'→ v={v:+.3f} ω={omega:+.3f}')

    def _detected_cb(self, msg: Bool):
        # Podría usarse para transiciones adicionales si necesario
        pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Watchdog: detecta timeout de pérdida de objetivo
    # ─────────────────────────────────────────────────────────────────────────
    def _watchdog_cb(self):
        if self.last_detect_time is None:
            # Nunca detectó → modo búsqueda
            if self.state != State.SEARCHING:
                self._transition(State.SEARCHING)
            self._search_spin()
            return

        elapsed = (self.get_clock().now() - self.last_detect_time).nanoseconds * 1e-9

        if elapsed > self.timeout and self.state == State.TRACKING:
            self.get_logger().warn(
                f'Objetivo perdido ({elapsed:.1f}s) → SEARCHING')
            self._transition(State.SEARCHING)
            self.ang_pid.reset()
            self.lin_pid.reset()

        if self.state == State.SEARCHING:
            self._search_spin()

    # ─────────────────────────────────────────────────────────────────────────
    #  Comportamiento de búsqueda: girar suavemente
    # ─────────────────────────────────────────────────────────────────────────
    def _search_spin(self):
        self._send_cmd(0.0, self._search_dir * self.search_w)

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _send_cmd(self, v: float, omega: float):
        cmd = Twist()
        cmd.linear.x  = float(np.clip(v,     -self.v_max, self.v_max))
        cmd.angular.z = float(np.clip(omega, -self.w_max,  self.w_max))
        self.pub_cmd.publish(cmd)

    def _stop(self):
        self.pub_cmd.publish(Twist())

    def _transition(self, new_state: State):
        self.get_logger().info(f'Estado: {self.state.name} → {new_state.name}')
        self.state = new_state
        msg = String()
        msg.data = new_state.name
        self.pub_state.publish(msg)

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = VisualServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
