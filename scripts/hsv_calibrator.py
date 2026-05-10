#!/usr/bin/env python3
"""
hsv_calibrator.py  —  Herramienta de calibración de color HSV
==============================================================
Abre la cámara y muestra sliders interactivos para ajustar los límites
HSV hasta aislar perfectamente el objeto a seguir.

Al terminar, imprime los valores para copiar en params.yaml.

Uso:
  python3 scripts/hsv_calibrator.py
  python3 scripts/hsv_calibrator.py --device 0      # cámara USB índice 0
  python3 scripts/hsv_calibrator.py --device /dev/video0
"""

import cv2
import numpy as np
import argparse


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        help='Índice o path de la cámara (default: 0)')
    parser.add_argument('--width',  type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f'ERROR: No se pudo abrir la cámara: {device}')
        return

    # ── Ventana con trackbars ──
    win = 'HSV Calibrator  [Q para salir y guardar]'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 120)

    # Valores iniciales (verde)
    cv2.createTrackbar('H low',  win,  35, 179, nothing)
    cv2.createTrackbar('H high', win,  85, 179, nothing)
    cv2.createTrackbar('S low',  win,  60, 255, nothing)
    cv2.createTrackbar('S high', win, 255, 255, nothing)
    cv2.createTrackbar('V low',  win,  60, 255, nothing)
    cv2.createTrackbar('V high', win, 255, 255, nothing)

    morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    print('\n=== HSV Calibrator ===')
    print('Ajusta los sliders hasta que el objeto se vea BLANCO en la máscara.')
    print('Presiona Q para salir e imprimir los valores finales.\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_lo = cv2.getTrackbarPos('H low',  win)
        h_hi = cv2.getTrackbarPos('H high', win)
        s_lo = cv2.getTrackbarPos('S low',  win)
        s_hi = cv2.getTrackbarPos('S high', win)
        v_lo = cv2.getTrackbarPos('V low',  win)
        v_hi = cv2.getTrackbarPos('V high', win)

        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  morph)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph)

        # Encontrar y dibujar el contorno más grande
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()
        if contours:
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            cv2.drawContours(result, [best], -1, (0, 255, 0), 2)
            M = cv2.moments(best)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result, f'area={area:.0f}px²',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

        # Mostrar frame original + máscara + resultado
        mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined  = np.hstack([result, mask_bgr])
        scale     = min(1.0, 1200 / combined.shape[1])
        combined  = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.putText(combined,
                    f'H:[{h_lo},{h_hi}] S:[{s_lo},{s_hi}] V:[{v_lo},{v_hi}]',
                    (10, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow(win, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('=' * 50)
    print('Copia esto en config/params.yaml:')
    print(f'  hsv_lower: [{h_lo}, {s_lo}, {v_lo}]')
    print(f'  hsv_upper: [{h_hi}, {s_hi}, {v_hi}]')
    print('=' * 50)


if __name__ == '__main__':
    main()
