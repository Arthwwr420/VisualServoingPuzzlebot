#!/usr/bin/env python3
"""
kalman_tracker.py — Filtro de Kalman 2D para seguimiento visual.

Módulo standalone (no nodo ROS2) que puede integrarse en el pipeline
de visión para suavizar detecciones ruidosas y predecir la posición
del objetivo durante pérdidas temporales de visión.

Modelo de movimiento: velocidad constante (CWNA — Constant Velocity, White Noise Acceleration).
  Estado:     x = [cx, cy, vx, vy]ᵀ   (posición y velocidad en píxeles)
  Observación: z = [cx, cy]ᵀ

Uso básico:
    tracker = KalmanVisualTracker(img_w=640, img_h=480, dt=1/30)

    # Con detección disponible
    cx_f, cy_f = tracker.update(cx_measured, cy_measured)

    # Sin detección (predicción pura)
    cx_pred, cy_pred = tracker.predict_only()
"""

import numpy as np


class KalmanVisualTracker:
    """Filtro de Kalman lineal 2D para tracking de centroide en imagen.

    Modelo de velocidad constante:
        x_{k+1} = F · x_k + w_k     (dinámica del proceso)
        z_k     = H · x_k + v_k     (modelo de medición)

    Args:
        img_w, img_h:        Dimensiones de la imagen [px] (para clipping en predict_only).
        dt:                  Paso de tiempo entre frames [s].
        process_noise_pos:   Desviación estándar del ruido de proceso en posición [px].
        process_noise_vel:   Desviación estándar del ruido de proceso en velocidad [px/s].
        measure_noise:       Desviación estándar del ruido de medición [px].
    """

    def __init__(self,
                 img_w: float = 640.0,
                 img_h: float = 480.0,
                 dt: float    = 1.0 / 30.0,
                 process_noise_pos: float = 2.0,
                 process_noise_vel: float = 10.0,
                 measure_noise:     float = 5.0):

        self.img_w = img_w
        self.img_h = img_h
        self.dt    = dt
        self._initialized = False

        # Matriz de transición de estado (modelo de velocidad constante)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float64)

        # Matriz de observación (solo posición es observable)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Covarianza del ruido de proceso Q
        q_p = process_noise_pos ** 2
        q_v = process_noise_vel ** 2
        self.Q = np.diag([q_p, q_p, q_v, q_v])

        # Covarianza del ruido de medición R
        r = measure_noise ** 2
        self.R = np.diag([r, r])

        # Estado inicial y covarianza (incertidumbre alta hasta primera detección)
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 500.0

    def initialize(self, cx: float, cy: float):
        """Inicializa el filtro con la primera detección.

        Args:
            cx, cy: Centroide inicial [px].
        """
        self.x = np.array([[cx], [cy], [0.0], [0.0]])
        self.P = np.eye(4) * 100.0
        self._initialized = True

    def update(self, cx: float, cy: float):
        """Ciclo completo: predicción + corrección Kalman con nueva medida.

        Args:
            cx, cy: Posición medida del centroide [px].

        Returns:
            Tuple (cx_filtrado, cy_filtrado).
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return cx, cy

        # Predicción
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Corrección con la medida z
        z = np.array([[cx], [cy]])
        y = z - self.H @ x_pred                    # Innovación
        S = self.H @ P_pred @ self.H.T + self.R    # Covarianza de innovación
        K = P_pred @ self.H.T @ np.linalg.inv(S)   # Ganancia de Kalman

        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict_only(self):
        """Solo predicción, sin corrección (para frames sin detección).

        Returns:
            Tuple (cx_pred, cy_pred) acotado al tamaño del frame.
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
        """Velocidad estimada del objetivo [px/s]."""
        if not self._initialized:
            return 0.0, 0.0
        return float(self.x[2, 0]) / self.dt, float(self.x[3, 0]) / self.dt

    @property
    def uncertainty(self):
        """Desviación estándar de la posición estimada [px]."""
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))

    def reset(self):
        """Reinicia el filtro a su estado inicial no-inicializado."""
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 500.0
        self._initialized = False


if __name__ == '__main__':
    import time

    # Demo: simula 60 frames con ruido y pérdida temporal (frames 30-40)
    tracker = KalmanVisualTracker(img_w=640, img_h=480, dt=1/30)

    print('frame | cx_true | cx_meas |  cx_filt | vx_est')
    print('-' * 55)
    for i in range(60):
        cx_true = 320 + i * 2.0
        cy_true = 240.0

        if 30 <= i <= 40:
            cx_f, cy_f = tracker.predict_only()
            cx_meas = None
        else:
            cx_meas = cx_true + np.random.normal(0, 4.0)
            cx_f, cy_f = tracker.update(cx_meas, cy_true)

        vx, _ = tracker.velocity
        meas_str = '---' if cx_meas is None else f'{cx_meas:7.1f}'
        print(f'{i:5d} | {cx_true:7.1f} | {meas_str} | {cx_f:8.1f} | {vx:7.1f}')
        time.sleep(0.01)
