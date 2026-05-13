#!/usr/bin/env python3
"""
lqr_controller.py — Controlador óptimo LQR para seguimiento visual del Puzzlebot.

Implementa un controlador LQR discreto con estado aumentado (integradores incluidos).
Resuelve la ecuación algebraica de Riccati discreta (DARE) en la inicialización
para obtener la ganancia óptima K, y aplica la ley de control u = -K·x en cada
detección recibida.

Vector de estado:   x  = [e_x, e_d, ∫e_x, ∫e_d]ᵀ
Vector de control:  u  = [ω, v]ᵀ

Tópicos suscritos:
  /detection/error  (geometry_msgs/Point)

Tópicos publicados:
  /VelocitySetL, /VelocitySetR  (std_msgs/Float32) — velocidades de rueda [rad/s]
  /controller/state             (std_msgs/String)
  /controller/gains             (std_msgs/String)

ROS2: Humble / Iron / Jazzy
"""

import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String, Float32
from enum import Enum, auto


class State(Enum):
    SEARCHING = auto()  # Girando en búsqueda del objetivo
    TRACKING  = auto()  # Seguimiento activo con LQR
    LOST      = auto()  # Objetivo perdido, detenido


class LQRVisualController(Node):
    """Controlador LQR de visual servoing para robot diferencial.

    Calcula ganancias LQR óptimas al iniciar y las aplica para generar
    velocidades de rueda a partir del error de detección visual.
    """

    WHEEL_RADIUS = 0.05   # Radio de rueda [m]
    WHEEL_BASE   = 0.19   # Distancia entre ruedas [m]

    def __init__(self):
        super().__init__('lqr_visual_controller')

        # Parámetros de movimiento
        self.declare_parameter('desired_distance', 0.50)
        self.declare_parameter('v_max',            0.25)
        self.declare_parameter('omega_max',        1.20)
        self.declare_parameter('sample_time',      1.0 / 30.0)
        self.declare_parameter('model_k_w',        1.0)  # Ganancia del modelo angular
        self.declare_parameter('model_k_v',        1.0)  # Ganancia del modelo lineal

        # Matrices de peso LQR (Q: costo de estado, R: costo de control)
        self.declare_parameter('q_ex',    12.0)  # Penalización error horizontal
        self.declare_parameter('q_ed',     6.0)  # Penalización error de distancia
        self.declare_parameter('q_iex',    0.8)  # Penalización integral e_x
        self.declare_parameter('q_ied',    0.4)  # Penalización integral e_d
        self.declare_parameter('r_omega',  0.15) # Penalización esfuerzo angular
        self.declare_parameter('r_v',      0.20) # Penalización esfuerzo lineal

        # Límites de los integradores
        self.declare_parameter('int_ex_max', 2.0)
        self.declare_parameter('int_ed_max', 1.5)

        # Parámetros de comportamiento
        self.declare_parameter('lost_timeout',   1.5)
        self.declare_parameter('search_omega',   0.35)
        self.declare_parameter('centering_gain', 1.2)
        self.declare_parameter('deadzone_ex',    0.04)
        self.declare_parameter('deadzone_ed',    0.04)

        # Límites de aceleración para ramp-up suave
        self.declare_parameter('max_accel_v', 0.15)
        self.declare_parameter('max_accel_w', 0.20)

        # Leer parámetros
        self.d_goal      = self.get_parameter('desired_distance').value
        self.v_max       = self.get_parameter('v_max').value
        self.w_max       = self.get_parameter('omega_max').value
        self.dt          = self.get_parameter('sample_time').value
        self.k_w         = self.get_parameter('model_k_w').value
        self.k_v         = self.get_parameter('model_k_v').value
        self.timeout     = self.get_parameter('lost_timeout').value
        self.search_w    = self.get_parameter('search_omega').value
        self.c_gain      = self.get_parameter('centering_gain').value
        self.dz_ex       = self.get_parameter('deadzone_ex').value
        self.dz_ed       = self.get_parameter('deadzone_ed').value
        self.int_ex_max  = self.get_parameter('int_ex_max').value
        self.int_ed_max  = self.get_parameter('int_ed_max').value
        self.max_accel_v = self.get_parameter('max_accel_v').value
        self.max_accel_w = self.get_parameter('max_accel_w').value

        # Calcular ganancia LQR
        self.F, self.G, self.K = self._build_lqr()

        # Estado interno del controlador
        self._x         = np.zeros(4)  # [e_x, e_d, ∫e_x, ∫e_d]
        self.prev_v_cmd = 0.0
        self.prev_w_cmd = 0.0

        self._last_time             = self.get_clock().now()
        self._last_detect           = None
        self.state                  = State.SEARCHING
        self._detection_this_cycle  = False

        # ROS2 I/O
        self.sub = self.create_subscription(
            Point, '/detection/error', self._error_cb, 10)

        self.pub_left  = self.create_publisher(Float32, '/VelocitySetL',      10)
        self.pub_right = self.create_publisher(Float32, '/VelocitySetR',      10)
        self.pub_state = self.create_publisher(String,  '/controller/state',  10)
        self.pub_gains = self.create_publisher(String,  '/controller/gains',  10)

        self.create_timer(0.10, self._watchdog_cb)
        self._publish_gains_info()

        self.get_logger().info(
            f'LQRVisualController listo | dt={self.dt:.4f}s | '
            f'd_goal={self.d_goal}m\n'
            f'  K =\n{np.round(self.K, 4)}'
        )

    def _build_lqr(self):
        """Construye el modelo discreto y calcula la ganancia LQR óptima.

        Modelo de primer orden para cada canal (linealización del servoing):
          ė_x = -k_w · ω
          ė_d = -k_v · v

        La discretización se hace por el método de Euler de primer orden.
        La ganancia se obtiene resolviendo la DARE con SciPy.

        Returns:
            Tuple (F, G, K): matrices del sistema discreto y ganancia LQR.

        Raises:
            Exception: si la DARE no converge.
        """
        dt  = self.dt
        k_w = self.k_w
        k_v = self.k_v

        A = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        B = np.array([
            [-k_w,  0.  ],
            [ 0.,  -k_v ],
            [ 0.,   0.  ],
            [ 0.,   0.  ]
        ])

        F = np.eye(4) + A * dt
        G = (np.eye(4) * dt + A * (dt ** 2) / 2.0) @ B

        Q = np.diag([
            self.get_parameter('q_ex').value,
            self.get_parameter('q_ed').value,
            self.get_parameter('q_iex').value,
            self.get_parameter('q_ied').value
        ])
        R = np.diag([
            self.get_parameter('r_omega').value,
            self.get_parameter('r_v').value
        ])

        try:
            P = solve_discrete_are(F, G, Q, R)
        except Exception as e:
            self.get_logger().error(f'DARE no converge: {e}')
            raise

        K = np.linalg.inv(G.T @ P @ G + R) @ (G.T @ P @ F)

        # Verificar estabilidad del sistema en lazo cerrado
        Acl  = F - G @ K
        eigs = np.abs(np.linalg.eigvals(Acl))
        stable = bool(np.all(eigs < 1.0))

        self.get_logger().info(
            f'DARE resuelta | Estable={stable} | |λ|_max={np.max(eigs):.4f}\n'
            f'  Autovalores: {np.round(eigs, 4)}'
        )

        if not stable:
            self.get_logger().error('¡Sistema en lazo cerrado inestable! Revisar Q y R.')

        return F, G, K

    def _error_cb(self, msg: Point):
        """Callback de detección: aplica la ley de control LQR y publica velocidades."""
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        dt  = float(np.clip(dt, 0.005, 0.200))

        self._detection_this_cycle = True

        # Objetivo perdido
        if msg.z < 0.0:
            if self.state == State.TRACKING:
                self._transition(State.LOST)
            self._reset_integrators()
            self._stop()
            return

        # Objetivo detectado
        self._last_detect = now
        if self.state != State.TRACKING:
            self._transition(State.TRACKING)

        e_x = float(msg.x)
        # e_d > 0: robot más cerca de lo deseado (retroceder)
        # e_d < 0: robot más lejos de lo deseado (avanzar)
        e_d = float(self.d_goal - msg.z)

        # Aplicar deadzone
        if abs(e_x) < self.dz_ex:
            e_x = 0.0
        if abs(e_d) < self.dz_ed:
            e_d = 0.0

        # Actualizar estado: errores actuales e integrales acotadas
        self._x[0] = e_x
        self._x[1] = e_d
        self._x[2] = float(np.clip(self._x[2] + e_x * dt, -self.int_ex_max, self.int_ex_max))
        self._x[3] = float(np.clip(self._x[3] + e_d * dt, -self.int_ed_max, self.int_ed_max))

        # Ley de control LQR: u = -K·x
        u          = -self.K @ self._x
        omega_star = float(u[0])
        v_star     = float(u[1])

        # Gain scheduling: reducir velocidad lineal cuando el objetivo no está centrado
        centering = float(np.clip(1.0 - self.c_gain * abs(e_x), 0.0, 1.0))
        v_target  = v_star * centering

        # Acotar y aplicar rate limiting para suavizar transiciones
        omega_target = float(np.clip(omega_star, -self.w_max, self.w_max))
        v_target     = float(np.clip(v_target,   -self.v_max, self.v_max))

        omega_cmd = self._rate_limit(omega_target, self.prev_w_cmd, self.max_accel_w)
        v_cmd     = self._rate_limit(v_target,     self.prev_v_cmd, self.max_accel_v)

        self.prev_w_cmd = omega_cmd
        self.prev_v_cmd = v_cmd

        self._send_cmd(v_cmd, omega_cmd)

        self.get_logger().info(
            f'[TRACK] dist={msg.z:.2f}m e_d={e_d:+.3f} e_x={e_x:+.3f} '
            f'centering={centering:.2f} '
            f'v*={v_star:+.3f} → v_cmd={v_cmd:+.3f}  '
            f'ω={omega_cmd:+.3f}'
        )

    def _watchdog_cb(self):
        """Watchdog a 10 Hz: detecta pérdida de objetivo y activa modo búsqueda."""
        # Si hubo detección en este ciclo, el comando LQR ya fue publicado.
        if self._detection_this_cycle:
            self._detection_this_cycle = False
            return

        self._detection_this_cycle = False

        if self._last_detect is None:
            if self.state != State.SEARCHING:
                self._transition(State.SEARCHING)
            self._search_spin()
            return

        elapsed = (self.get_clock().now() - self._last_detect).nanoseconds * 1e-9

        if elapsed > self.timeout:
            if self.state == State.TRACKING:
                self.get_logger().warn(f'Objetivo perdido ({elapsed:.1f}s) → SEARCHING')
                self._transition(State.SEARCHING)
                self._reset_integrators()
                self.prev_v_cmd = 0.0
                self.prev_w_cmd = 0.0

        if self.state == State.SEARCHING:
            self._search_spin()

    def _search_spin(self):
        """Gira a velocidad constante para buscar el objetivo."""
        self._send_cmd(0.0, self.search_w)

    def _reset_integrators(self):
        """Reinicia los términos integrales del estado."""
        self._x[2] = 0.0
        self._x[3] = 0.0

    def _rate_limit(self, target: float, previous: float, max_delta: float) -> float:
        """Limita la tasa de cambio de un comando para suavizar transiciones."""
        delta = float(np.clip(target - previous, -max_delta, max_delta))
        return previous + delta

    def _send_cmd(self, v: float, omega: float):
        """Convierte (v, ω) a velocidades individuales de rueda y publica.

        Modelo cinemático diferencial:
          v_L = (v - ω·b/2) / r
          v_R = (v + ω·b/2) / r
        """
        half_base = self.WHEEL_BASE / 2.0
        v_left    = (v - omega * half_base) / self.WHEEL_RADIUS
        v_right   = (v + omega * half_base) / self.WHEEL_RADIUS

        msg_l      = Float32(); msg_l.data = float(v_left)
        msg_r      = Float32(); msg_r.data = float(v_right)

        self.pub_left.publish(msg_l)
        self.pub_right.publish(msg_r)

        self.get_logger().debug(f'Ruedas → L={v_left:+.3f}  R={v_right:+.3f} rad/s')

    def _stop(self):
        """Publica velocidad cero en ambas ruedas."""
        zero = Float32()
        zero.data = 0.0
        self.pub_left.publish(zero)
        self.pub_right.publish(zero)
        self.prev_v_cmd = 0.0
        self.prev_w_cmd = 0.0

    def _transition(self, new_state: State):
        """Registra la transición de estado y la publica."""
        self.get_logger().info(f'Estado: {self.state.name} → {new_state.name}')
        self.state = new_state
        msg      = String()
        msg.data = new_state.name
        self.pub_state.publish(msg)

    def _publish_gains_info(self):
        """Publica la matriz de ganancias LQR calculada en /controller/gains."""
        info = (
            f'LQR K =\n'
            f'  ω = {-self.K[0,0]:.4f}·e_x + {-self.K[0,1]:.4f}·e_d + '
            f'{-self.K[0,2]:.4f}·∫e_x + {-self.K[0,3]:.4f}·∫e_d\n'
            f'  v = {-self.K[1,0]:.4f}·e_x + {-self.K[1,1]:.4f}·e_d + '
            f'{-self.K[1,2]:.4f}·∫e_x + {-self.K[1,3]:.4f}·∫e_d'
        )
        msg      = String()
        msg.data = info
        self.pub_gains.publish(msg)

    def destroy_node(self):
        self._stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LQRVisualController()
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
