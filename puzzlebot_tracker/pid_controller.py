#!/usr/bin/env python3
"""
pid_controller.py — Controlador PID de visual servoing para Puzzlebot.

Implementa dos lazos PID independientes en lazo cerrado:
  - Lazo angular:  error horizontal (e_x) → velocidad angular ω [rad/s]
  - Lazo lineal:   error de distancia (e_d) → velocidad lineal v [m/s]

Características:
  - Anti-windup por clamping de la integral
  - Filtro paso-bajo en la derivada (reduce amplificación de ruido)
  - Deadzone por canal (evita micro-oscilaciones en equilibrio)
  - Gain scheduling: atenúa v cuando el objetivo no está centrado
  - Watchdog: detiene el robot ante pérdida del objetivo
  - Máquina de estados: SEARCHING → TRACKING → LOST

Tópicos suscritos:
  /detection/error  (geometry_msgs/Point)
  /detection/active (std_msgs/Bool)

Tópicos publicados:
  /cmd_vel          (geometry_msgs/Twist)
  /controller/state (std_msgs/String)

ROS2: Humble / Iron / Jazzy
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, String

import numpy as np
from enum import Enum, auto


class State(Enum):
    SEARCHING = auto()  # Girando en búsqueda del objetivo
    TRACKING  = auto()  # Seguimiento activo con PID
    LOST      = auto()  # Objetivo perdido, detenido


class PIDController:
    """Controlador PID discreto con deadzone, anti-windup y filtro derivativo.

    La derivada se filtra con un paso-bajo de primer orden para evitar
    la amplificación de ruido de cuantización:
        D_filt = α · D_prev + (1 - α) · D_raw

    Args:
        kp, ki, kd:      Ganancias proporcional, integral y derivativa.
        max_output:      Límite simétrico de la salida.
        max_integral:    Límite anti-windup de la integral.
        deadzone:        Umbral de error bajo el cual la salida es cero.
        deriv_alpha:     Factor del filtro paso-bajo derivativo [0, 1].
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
        """Reinicia el estado interno del controlador."""
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._deriv_flt = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Calcula la salida del PID para el error y paso de tiempo dados.

        Args:
            error: Error de seguimiento actual.
            dt:    Intervalo de tiempo desde el último cómputo [s].

        Returns:
            Señal de control acotada a [-max_output, +max_output].
        """
        if abs(error) < self.deadzone:
            error = 0.0

        # Integral con anti-windup por clamping
        self._integral += error * dt
        self._integral  = float(np.clip(self._integral,
                                        -self.max_integral, self.max_integral))

        # Derivada filtrada
        raw_deriv       = (error - self._prev_err) / max(dt, 1e-4)
        self._deriv_flt = (self.d_alpha * self._deriv_flt +
                           (1.0 - self.d_alpha) * raw_deriv)
        self._prev_err  = error

        output = (self.kp * error +
                  self.ki * self._integral +
                  self.kd * self._deriv_flt)

        return float(np.clip(output, -self.max_output, self.max_output))

    def set_gains(self, kp: float, ki: float, kd: float):
        """Actualiza ganancias en tiempo real sin reiniciar el estado."""
        self.kp, self.ki, self.kd = kp, ki, kd


class VisualServoController(Node):
    """Nodo ROS2 de control PID para visual servoing del Puzzlebot.

    Centra el objetivo horizontalmente (lazo angular) y mantiene
    la distancia deseada al mismo (lazo lineal).
    """

    def __init__(self):
        super().__init__('visual_servo_controller')

        # Parámetros de movimiento
        self.declare_parameter('desired_distance', 0.50)   # [m]
        self.declare_parameter('v_max',   0.25)            # [m/s]
        self.declare_parameter('omega_max', 1.20)          # [rad/s]

        # Ganancias PID angular (e_x → ω)
        self.declare_parameter('ang_kp', 0.65)
        self.declare_parameter('ang_ki', 0.008)
        self.declare_parameter('ang_kd', 0.10)

        # Ganancias PID lineal (e_d → v)
        self.declare_parameter('lin_kp', 0.45)
        self.declare_parameter('lin_ki', 0.005)
        self.declare_parameter('lin_kd', 0.06)

        # Parámetros de comportamiento
        self.declare_parameter('deadzone_angular', 0.04)   # normalizado [-1,1]
        self.declare_parameter('deadzone_linear',  0.04)   # [m]
        self.declare_parameter('lost_timeout',     1.5)    # [s]
        self.declare_parameter('search_omega',     0.35)   # [rad/s]
        self.declare_parameter('centering_gain',   1.2)    # factor de atenuación lineal

        # Leer parámetros
        self.d_goal   = self.get_parameter('desired_distance').value
        self.v_max    = self.get_parameter('v_max').value
        self.w_max    = self.get_parameter('omega_max').value
        self.timeout  = self.get_parameter('lost_timeout').value
        self.search_w = self.get_parameter('search_omega').value
        self.c_gain   = self.get_parameter('centering_gain').value

        # Instanciar controladores PID
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

        # Estado interno
        self.state            = State.SEARCHING
        self.last_detect_time = None
        self.last_cb_time     = self.get_clock().now()
        self._search_dir      = 1.0  # +1 = derecha, -1 = izquierda

        # ROS2 I/O
        self.sub_err = self.create_subscription(
            Point, '/detection/error', self._error_cb, 10)
        self.sub_det = self.create_subscription(
            Bool, '/detection/active', self._detected_cb, 10)

        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.pub_state = self.create_publisher(String, '/controller/state', 10)

        self.watchdog_timer = self.create_timer(0.10, self._watchdog_cb)

        self.get_logger().info(
            f'VisualServoController listo | d_goal={self.d_goal}m | '
            f'v_max={self.v_max} m/s | ω_max={self.w_max} rad/s'
        )

    def _error_cb(self, msg: Point):
        """Callback de error de visión: computa y publica comandos PID."""
        now = self.get_clock().now()
        dt  = (now - self.last_cb_time).nanoseconds * 1e-9
        self.last_cb_time = now
        dt  = float(np.clip(dt, 0.005, 0.200))

        # Objetivo perdido
        if msg.z < 0.0:
            if self.state == State.TRACKING:
                self._transition(State.LOST)
            self._stop()
            self.ang_pid.reset()
            self.lin_pid.reset()
            return

        # Objetivo detectado
        self.last_detect_time = now
        if self.state != State.TRACKING:
            self._transition(State.TRACKING)

        # Lazo angular: msg.x > 0 → objetivo a la derecha → ω negativo (giro derecha)
        omega = -self.ang_pid.compute(msg.x, dt)

        # Lazo lineal: dist_error > 0 → robot demasiado lejos → avanzar
        dist_err = msg.z - self.d_goal
        v_raw    = self.lin_pid.compute(dist_err, dt)

        # Gain scheduling: reducir v cuando el objetivo no está centrado
        centering_factor = max(0.0, 1.0 - self.c_gain * abs(msg.x))
        v = v_raw * centering_factor

        self._send_cmd(v, omega)

        self.get_logger().debug(
            f'[TRACKING] ex={msg.x:+.3f} dist={msg.z:.3f}m '
            f'→ v={v:+.3f} ω={omega:+.3f}'
        )

    def _detected_cb(self, msg: Bool):
        """Callback de bandera de detección (disponible para lógica adicional)."""
        pass

    def _watchdog_cb(self):
        """Watchdog a 10 Hz: detecta timeout y activa modo búsqueda."""
        if self.last_detect_time is None:
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

    def _search_spin(self):
        """Gira a velocidad constante para buscar el objetivo."""
        self._send_cmd(0.0, self._search_dir * self.search_w)

    def _send_cmd(self, v: float, omega: float):
        """Publica comando de velocidad acotado en /cmd_vel."""
        cmd = Twist()
        cmd.linear.x  = float(np.clip(v,     -self.v_max, self.v_max))
        cmd.angular.z = float(np.clip(omega, -self.w_max,  self.w_max))
        self.pub_cmd.publish(cmd)

    def _stop(self):
        """Publica Twist vacío para detener el robot."""
        self.pub_cmd.publish(Twist())

    def _transition(self, new_state: State):
        """Registra la transición de estado y la publica."""
        self.get_logger().info(f'Estado: {self.state.name} → {new_state.name}')
        self.state = new_state
        msg = String()
        msg.data = new_state.name
        self.pub_state.publish(msg)

    def destroy_node(self):
        self._stop()
        super().destroy_node()


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
