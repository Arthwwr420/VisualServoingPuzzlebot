#!/usr/bin/python3
"""
raw_cam.py — Nodo de cámara CSI standalone para Puzzlebot (Jetson Nano).

Captura video de la cámara CSI vía GStreamer (NVMM) y lo publica como
sensor_msgs/CompressedImage. Útil para diagnóstico de la cámara y
pruebas independientes del pipeline de visión.

Basado en: https://github.com/JetsonHacksNano/CSI-Camera

Tópico publicado: /image/back/image_compressed (sensor_msgs/CompressedImage)
"""

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=4032,
    capture_height=3040,
    display_width=504,
    display_height=380,
    framerate=20,
    flip_method=1,
):
    """Construye el pipeline GStreamer para la cámara CSI del Jetson Nano.

    Captura en resolución nativa (NVMM) y escala con nvvidconv.
    """
    return (
        "nvarguscamerasrc sensor_mode=0 sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CameraNode(Node):
    """Nodo ROS2 que captura frames de la cámara CSI y los publica como CompressedImage.

    La captura se hace a 60 Hz (timer rápido) y la publicación a 10 Hz,
    desacoplando la adquisición del envío de mensajes.
    """

    def __init__(self):
        super().__init__('back_camera_node')

        image_topic = self.declare_parameter(
            'image_topic', '/image/back/image_compressed').value
        self.frame_id = self.declare_parameter('frame_id', 'camera').value

        self.image_publisher = self.create_publisher(CompressedImage, image_topic, 1)
        self.br = CvBridge()

        pipeline = gstreamer_pipeline(flip_method=3, framerate=10)
        self.get_logger().info(f'Iniciando captura:\n  {pipeline}')
        self.video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        self.last_image = None

        # Timer de captura a 60 Hz; publicación a 10 Hz
        self.create_timer(1.0 / 60, self._capture_cb)
        self.create_timer(1.0 / 10, self._publish_cb)

    def close_videocapture(self):
        """Libera el recurso de captura de video."""
        self.video_capture.release()

    def _capture_cb(self):
        """Captura el frame más reciente de la cámara."""
        if self.video_capture.isOpened():
            try:
                _, self.last_image = self.video_capture.read()
            except Exception as e:
                self.get_logger().warn(f'Error en captura: {e}')

    def _publish_cb(self):
        """Publica el último frame capturado como CompressedImage."""
        if self.last_image is None:
            return
        time_msg = self._get_time_msg()
        img_msg  = self._make_image_msg(self.last_image, time_msg)
        self.image_publisher.publish(img_msg)

    def _get_time_msg(self) -> Time:
        """Crea un mensaje Time con el timestamp actual del reloj ROS2."""
        time_msg = Time()
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time_msg.sec    = int(sec)
        time_msg.nanosec = int(nanosec)
        return time_msg

    def _make_image_msg(self, image, time: Time) -> CompressedImage:
        """Convierte un frame de OpenCV a sensor_msgs/CompressedImage.

        Args:
            image: Frame BGR de OpenCV.
            time:  Timestamp ROS2 para el header.

        Returns:
            Mensaje CompressedImage listo para publicar.
        """
        img_msg = self.br.cv2_to_compressed_imgmsg(image)
        img_msg.header.stamp    = time
        img_msg.header.frame_id = self.frame_id
        return img_msg


def main(args=None):
    """Modo standalone de diagnóstico: muestra el stream en ventana local."""
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=3, framerate=10),
        cv2.CAP_GSTREAMER
    )
    if video_capture.isOpened():
        cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
        while True:
            ret_val, img = video_capture.read()
            cv2.imshow('demo', img)
            cv2.waitKey(10)
    else:
        print('Error: no se pudo abrir la cámara.')

    cv2.destroyAllWindows()
    video_capture.release()


if __name__ == '__main__':
    main()
