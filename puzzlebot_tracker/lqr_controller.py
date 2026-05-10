#!/usr/bin/env python3
"""
lqr_controller.py  —  Puzzlebot Visual Servoing con Control Óptimo LQR
=======================================================================

FORMULACIÓN DEL PROBLEMA ÓPTIMO
─────────────────────────────────────────────────────────────────────────
Estado aumentado con integrales de error (acción integral sin heurística):

  x = [e_x,   e_d,   ∫e_x·dt,   ∫e_d·dt]ᵀ

  e_x  : error horizontal normalizado ∈ [-1, +1]   (positivo = objetivo a la derecha)
  e_d  : error de distancia (d_medida − d_deseada)  [m]
  ∫e_x : integral de e_x                            [s]
  ∫e_d : integral de e_d                            [m·s]

Control:
  u = [ω, v]ᵀ   (velocidad angular [rad/s], lineal [m/s])

Dinámica linealizada en torno al equilibrio (e=0, d=d*):
  ė_x = −k_w · ω     (rotación del robot corrige el centrado)
  ė_d = −k_v · v     (velocidad lineal corrige la distancia)

Modelo continuo  ẋ = A·x + B·u :

  A = ⌈ 0  0  0  0 ⌉     B = ⌈ −k_w   0   ⌉
      | 0  0  0  0 |         |   0   −k_v |
      | 1  0  0  0 |         |   0    0   |
      ⌊ 0  1  0  0 ⌋         ⌊   0    0   ⌋

Discretización exacta (ZOH, A es nilpotente → F = I + A·dt):

  F = I + A·dt          (matriz de transición discreta)
  G = (I·dt + A·dt²/2)·B   (matriz de entrada discreta)

Función de costo de horizonte infinito:

  J = Σ_{k=0}^{∞} [ xₖᵀ Q xₖ  +  uₖᵀ R uₖ ]

  Q = diag(q_ex, q_ed, q_iex, q_ied)   ← penaliza errores de estado
  R = diag(r_w,  r_v)                  ← penaliza esfuerzo de control

Solución via DARE (Discrete Algebraic Riccati Equation):

  P = FᵀPF − (FᵀPG)(GᵀPG + R)⁻¹(GᵀPF) + Q    [solve_discrete_are]

Ganancia óptima (constante, calculada UNA VEZ al inicio):

  K  = (GᵀPG + R)⁻¹ · GᵀPF           [2 × 4]

Ley de control óptima (lineal en el estado):

  u* = −K · x
  ω* = −K[0,:]·x    →  centra el objetivo en imagen
  v* = −K[1,:]·x    →  regula la distancia

Esta ley minimiza J de forma garantizada para el modelo lineal y
estabiliza el sistema cuando los autovalores de (F − G·K) están dentro
del círculo unitario (verificado en __init__).

─────────────────────────────────────────────────────────────────────────
Subscripciones:  /detection/error  (geometry_msgs/Point)
Publicaciones:   /cmd_vel          (geometry_msgs/Twist)
                 /controller/state (std_msgs/String)
                 /controller/gains (std_msgs/String)  ← debug

Autor:   Puzzlebot Team
ROS2:    Humble / Iron / Jazzy
Deps:    numpy, scipy
─────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    SEARCHING = auto()
    TRACKING  = auto()
    LOST      = auto()


# ─────────────────────────────────────────────────────────────────────────────
class LQRVisualController(Node):
    """
    Controlador óptimo LQR de visual servoing para el Puzzlebot.

    Diseño:
      • Estado de 4 dimensiones con términos integrales
      • Matrices F, G obtenidas por discretización exacta (ZOH)
      • Ganancia K calculada offline resolviendo la DARE
      • Anti-windup: saturación de los estados integrales
      • Gain scheduling físico: reduce v cuando no está centrado
      • Máquina de estados: SEARCHING / TRACKING / LOST
    """

    def __init__(self):
        super().__init__('lqr_visual_controller')

        # ── Parámetros de la plataforma ──────────────────────────────────────
        self.declare_parameter('desired_distance', 0.50)
        self.declare_parameter('v_max',   0.25)
        self.declare_parameter('omega_max', 1.20)
        self.declare_parameter('sample_time', 1.0 / 30.0)

        # Sensibilidades del modelo linealizado
        # k_w: cuánto cambia e_x por unidad de omega (normalizado)
        # k_v: cuánto cambia e_d por unidad de v
        self.declare_parameter('model_k_w', 1.0)
        self.declare_parameter('model_k_v', 1.0)

        # ── Matrices de costo Q y R ──────────────────────────────────────────
        # Q = diag(q_ex, q_ed, q_iex, q_ied)
        # Aumentar q_ex → respuesta angular más agresiva
        # Aumentar q_ed → respuesta lineal más agresiva
        # Aumentar r_w / r_v → control más suave (menos agresivo)
        self.declare_parameter('q_ex',   12.0)   # penalización error horizontal
        self.declare_parameter('q_ed',    6.0)   # penalización error de distancia
        self.declare_parameter('q_iex',   0.8)   # penalización integral e_x
        self.declare_parameter('q_ied',   0.4)   # penalización integral e_d
        self.declare_parameter('r_omega', 0.15)  # costo de velocidad angular
        self.declare_parameter('r_v',     0.20)  # costo de velocidad lineal

        # ── Límites de estados integrales (anti-windup) ──────────────────────
        self.declare_parameter('int_ex_max', 2.0)   # [s]  integra ex
        self.declare_parameter('int_ed_max', 1.5)   # [m·s] integra ed

        # ── Comportamiento de pérdida ────────────────────────────────────────
        self.declare_parameter('lost_timeout',  1.5)
        self.declare_parameter('search_omega',  0.35)
        self.declare_parameter('centering_gain', 1.2)
        self.declare_parameter('deadzone_ex',   0.04)
        self.declare_parameter('deadzone_ed',   0.04)

        # ── Leer parámetros ──────────────────────────────────────────────────
        self.d_goal   = self.get_parameter('desired_distance').value
        self.v_max    = self.get_parameter('v_max').value
        self.w_max    = self.get_parameter('omega_max').value
        self.dt       = self.get_parameter('sample_time').value
        self.k_w      = self.get_parameter('model_k_w').value
        self.k_v      = self.get_parameter('model_k_v').value
        self.timeout  = self.get_parameter('lost_timeout').value
        self.search_w = self.get_parameter('search_omega').value
        self.c_gain   = self.get_parameter('centering_gain').value
        self.dz_ex    = self.get_parameter('deadzone_ex').value
        self.dz_ed    = self.get_parameter('deadzone_ed').value

        self.int_ex_max = self.get_parameter('int_ex_max').value
        self.int_ed_max = self.get_parameter('int_ed_max').value

        # ── Construir y resolver el LQR ──────────────────────────────────────
        self.F, self.G, self.K = self._build_lqr()

        # ── Estado del controlador ───────────────────────────────────────────
        # x = [e_x, e_d, ∫e_x, ∫e_d]
        self._x = np.zeros(4)
        self._last_time       = self.get_clock().now()
        self._last_detect     = None
        self.state            = State.SEARCHING

        # ── ROS2 I/O ─────────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Point, '/detection/error', self._error_cb, 10)
        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.pub_state = self.create_publisher(String, '/controller/state',  10)
        self.pub_gains = self.create_publisher(String, '/controller/gains',  10)

        # Watchdog a 10 Hz
        self.create_timer(0.10, self._watchdog_cb)

        # Publicar info de las ganancias una vez
        self._publish_gains_info()

        self.get_logger().info(
            f'LQRVisualController listo | dt={self.dt:.4f}s | '
            f'd_goal={self.d_goal}m\n'
            f'  K =\n{np.round(self.K, 4)}')

    # ─────────────────────────────────────────────────────────────────────────
    #  Construcción del LQR
    # ─────────────────────────────────────────────────────────────────────────
    def _build_lqr(self):
        """
        Construye las matrices del sistema discreto y resuelve la DARE
        para obtener la ganancia óptima K.

        Retorna: (F, G, K)
        """
        dt   = self.dt
        k_w  = self.k_w
        k_v  = self.k_v

        # ── Modelo continuo ──────────────────────────────────────────────────
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

        # ── Discretización exacta (A es nilpotente de orden 2) ───────────────
        # F = expm(A·dt) = I + A·dt  (serie exacta: A² = 0)
        # G = ∫₀^dt expm(A·τ)dτ · B = (I·dt + A·dt²/2)·B
        F = np.eye(4) + A * dt
        G = (np.eye(4) * dt + A * (dt ** 2) / 2.0) @ B

        # ── Matrices de costo ────────────────────────────────────────────────
        q_ex  = self.get_parameter('q_ex').value
        q_ed  = self.get_parameter('q_ed').value
        q_iex = self.get_parameter('q_iex').value
        q_ied = self.get_parameter('q_ied').value
        r_w   = self.get_parameter('r_omega').value
        r_v   = self.get_parameter('r_v').value

        Q = np.diag([q_ex, q_ed, q_iex, q_ied])
        R = np.diag([r_w,  r_v])

        # ── Resolver DARE ────────────────────────────────────────────────────
        #   P = FᵀPF − (FᵀPG)(GᵀPG + R)⁻¹(GᵀPF) + Q
        try:
            P = solve_discrete_are(F, G, Q, R)
        except Exception as e:
            self.get_logger().error(f'DARE no converge: {e}')
            raise

        # ── Ganancia óptima ──────────────────────────────────────────────────
        #   K = (GᵀPG + R)⁻¹ · GᵀPF
        K = np.linalg.inv(G.T @ P @ G + R) @ (G.T @ P @ F)

        # ── Verificar estabilidad: |λ(F - G·K)| < 1 ─────────────────────────
        Acl  = F - G @ K
        eigs = np.abs(np.linalg.eigvals(Acl))
        stable = bool(np.all(eigs < 1.0))

        self.get_logger().info(
            f'DARE resuelta | Estable={stable} | |λ|_max={np.max(eigs):.4f}\n'
            f'  Autovalores lazo cerrado: {np.round(eigs, 4)}')

        if not stable:
            self.get_logger().error('¡Sistema inestable! Revisar Q y R.')

        return F, G, K

    # ─────────────────────────────────────────────────────────────────────────
    #  Callback de error visual
    # ─────────────────────────────────────────────────────────────────────────
    def _error_cb(self, msg: Point):
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        dt  = float(np.clip(dt, 0.005, 0.200))

        # ── Objetivo perdido ─────────────────────────────────────────────────
        if msg.z < 0.0:
            if self.state == State.TRACKING:
                self._transition(State.LOST)
            self._reset_integrators()
            self._stop()
            return

        # ── Objetivo detectado ───────────────────────────────────────────────
        self._last_detect = now
        if self.state != State.TRACKING:
            self._transition(State.TRACKING)

        # ── Construir vector de estado x ─────────────────────────────────────
        e_x = float(msg.x)
        e_d = float(msg.z - self.d_goal)

        # Aplicar dead-zone
        if abs(e_x) < self.dz_ex:
            e_x = 0.0
        if abs(e_d) < self.dz_ed:
            e_d = 0.0

        # Actualizar integrales (estados x[2] y x[3]) con anti-windup
        self._x[0] = e_x
        self._x[1] = e_d
        self._x[2] = float(np.clip(
            self._x[2] + e_x * dt, -self.int_ex_max, self.int_ex_max))
        self._x[3] = float(np.clip(
            self._x[3] + e_d * dt, -self.int_ed_max, self.int_ed_max))

        # ── Ley de control óptima: u* = −K·x ─────────────────────────────────
        u = -self.K @ self._x          # [ω*, v*]
        omega_star = float(u[0])
        v_star     = float(u[1])

        # ── Gain scheduling: reducir v cuando el objetivo no está centrado ───
        # Esto NO altera la optimalidad del LQR; es una restricción física
        # que evita que el robot avance mientras aún debe rotar.
        centering = max(0.0, 1.0 - self.c_gain * abs(e_x))
        v_cmd     = v_star * centering

        # ── Saturar a límites físicos del actuador ───────────────────────────
        omega_cmd = float(np.clip(omega_star, -self.w_max,  self.w_max))
        v_cmd     = float(np.clip(v_cmd,      -self.v_max,  self.v_max))

        self._send_cmd(v_cmd, omega_cmd)

        self.get_logger().debug(
            f'[LQR] x=[{e_x:+.3f}, {e_d:+.3f}, {self._x[2]:+.3f}, {self._x[3]:+.3f}] '
            f'→ ω={omega_cmd:+.3f} v={v_cmd:+.3f}')

    # ─────────────────────────────────────────────────────────────────────────
    #  Watchdog y búsqueda
    # ─────────────────────────────────────────────────────────────────────────
    def _watchdog_cb(self):
        if self._last_detect is None:
            if self.state != State.SEARCHING:
                self._transition(State.SEARCHING)
            self._search_spin()
            return

        elapsed = (self.get_clock().now() - self._last_detect).nanoseconds * 1e-9
        if elapsed > self.timeout and self.state == State.TRACKING:
            self.get_logger().warn(f'Objetivo perdido ({elapsed:.1f}s) → SEARCHING')
            self._transition(State.SEARCHING)
            self._reset_integrators()

        if self.state == State.SEARCHING:
            self._search_spin()

    def _search_spin(self):
        """Girar suavemente buscando el objetivo."""
        self._send_cmd(0.0, self.search_w)

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_integrators(self):
        """Resetear solo los estados integrales (x[2], x[3])."""
        self._x[2] = 0.0
        self._x[3] = 0.0

    def _send_cmd(self, v: float, omega: float):
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = omega
        self.pub_cmd.publish(cmd)

    def _stop(self):
        self.pub_cmd.publish(Twist())

    def _transition(self, new_state: State):
        self.get_logger().info(f'Estado: {self.state.name} → {new_state.name}')
        self.state = new_state
        msg = String()
        msg.data = new_state.name
        self.pub_state.publish(msg)

    def _publish_gains_info(self):
        """Publicar resumen de ganancias para debug / reporte."""
        info = (
            f'LQR K =\n'
            f'  ω = {-self.K[0,0]:.4f}·e_x + {-self.K[0,1]:.4f}·e_d + '
            f'{-self.K[0,2]:.4f}·∫e_x + {-self.K[0,3]:.4f}·∫e_d\n'
            f'  v = {-self.K[1,0]:.4f}·e_x + {-self.K[1,1]:.4f}·e_d + '
            f'{-self.K[1,2]:.4f}·∫e_x + {-self.K[1,3]:.4f}·∫e_d'
        )
        msg = String()
        msg.data = info
        self.pub_gains.publish(msg)

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
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
