#!/usr/bin/env python3
"""
kalman_tracker.py  —  Filtro de Kalman 2D para seguimiento visual
==================================================================
Extiende el pipeline con un Kalman Filter que:
  • Suaviza las detecciones ruidosas (cx, cy)
  • Predice la posición durante pérdidas temporales de visión
  • Estima velocidad del objetivo para anticipar movimiento

Estado del filtro:  [cx, cy, vx, vy]  (posición + velocidad en píxeles)
Observación:        [cx, cy]

Uso:
  tracker = KalmanVisualTracker(img_w=640, img_h=480, dt=1/30)
  cx_f, cy_f = tracker.update(cx_measured, cy_measured)
  cx_pred, cy_pred = tracker.predict_only()  # cuando no hay detección

Autor: Puzzlebot Team
"""

import numpy as np


class KalmanVisualTracker:
    """
    Filtro de Kalman lineal 2D para tracking de centroide en imagen.

    Modelo de movimiento de velocidad constante (CWNA):
        x_{k+1} = F * x_k + w_k     (proceso)
        z_k     = H * x_k + v_k     (observación)

    Estado:  x = [cx, cy, vx, vy]^T
    Medida:  z = [cx, cy]^T
    """

    def __init__(self,
                 img_w: float = 640.0,
                 img_h: float = 480.0,
                 dt: float    = 1.0 / 30.0,
                 process_noise_pos:  float = 2.0,
                 process_noise_vel:  float = 10.0,
                 measure_noise:      float = 5.0):

        self.img_w = img_w
        self.img_h = img_h
        self.dt    = dt
        self._initialized = False

        # ── Matrices del sistema ──
        # F: Matriz de transición de estado (vel. constante)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float64)

        # H: Matriz de observación (solo observamos posición)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Q: Ruido de proceso (incertidumbre del modelo dinámico)
        q_p = process_noise_pos ** 2
        q_v = process_noise_vel ** 2
        self.Q = np.diag([q_p, q_p, q_v, q_v])

        # R: Ruido de medición (ruido del detector visual)
        r = measure_noise ** 2
        self.R = np.diag([r, r])

        # Estado inicial y covarianza
        self.x = np.zeros((4, 1))         # [cx, cy, vx, vy]
        self.P = np.eye(4) * 500.0        # Incertidumbre inicial alta

    def initialize(self, cx: float, cy: float):
        """Inicializar el filtro con una primera detección."""
        self.x = np.array([[cx], [cy], [0.0], [0.0]])
        self.P = np.eye(4) * 100.0
        self._initialized = True

    def update(self, cx: float, cy: float):
        """
        Ciclo completo: predicción + corrección con nueva medida.
        Retorna (cx_filtrado, cy_filtrado).
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return cx, cy

        # ── Predicción ──
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Actualización (corrección con medida z) ──
        z = np.array([[cx], [cy]])
        y = z - self.H @ x_pred                    # innovación
        S = self.H @ P_pred @ self.H.T + self.R    # covarianza de innovación
        K = P_pred @ self.H.T @ np.linalg.inv(S)   # ganancia de Kalman

        self.x = x_pred + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict_only(self):
        """
        Solo predicción, sin corrección (para frames sin detección).
        Retorna (cx_pred, cy_pred) acotado al frame.
        """
        if not self._initialized:
            return self.img_w * 0.5, self.img_h * 0.5

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx = float(np.clip(self.x[0, 0], 0, self.img_w))
        cy = float(np.clip(self.x[1, 0], 0, self.img_h))
        return cx, cy

    @property
    def velocity(self):
        """Velocidad estimada del objetivo en píxeles/segundo."""
        if not self._initialized:
            return 0.0, 0.0
        return float(self.x[2, 0]) / self.dt, float(self.x[3, 0]) / self.dt

    @property
    def uncertainty(self):
        """Desviación estándar de posición estimada (px)."""
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))

    def reset(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 500.0
        self._initialized = False


# ─────────────────────────────────────────────────────────────────────────────
#  Integración con el nodo de visión (ejemplo de uso)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    tracker = KalmanVisualTracker(img_w=640, img_h=480, dt=1/30)

    # Simular 60 frames con ruido y una pérdida temporal (frames 30-40)
    print('frame | cx_true | cx_meas | cx_filt | vx_est')
    print('-' * 55)
    for i in range(60):
        cx_true = 320 + i * 2.0            # target moviendose a la derecha
        cy_true = 240.0

        if 30 <= i <= 40:
            # Simular pérdida de visión → solo predicción
            cx_f, cy_f = tracker.predict_only()
            cx_meas = None
        else:
            cx_meas = cx_true + np.random.normal(0, 4.0)
            cx_f, cy_f = tracker.update(cx_meas, cy_true)

        vx, _ = tracker.velocity
        print(f'{i:5d} | {cx_true:7.1f} | '
              f'{"---" if cx_meas is None else f"{cx_meas:7.1f}"} | '
              f'{cx_f:7.1f} | {vx:7.1f}')
        time.sleep(0.01)
