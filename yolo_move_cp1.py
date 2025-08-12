# yolo_test_jsg.py — rqt 실시간 멀티스레드 발행 + 300x300 ROI + 터미널 매칭 + XY 이동(Z 고정)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 필요 시 제거

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import numpy as np
import threading
from queue import Queue, Empty

import DR_init
from onrobot import RG

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# YOLO_MODEL_PATH = "/home/ohjunseok/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/rokey/rokey/basic/my_best.pt" ### 박스 약
YOLO_MODEL_PATH = "/home/ohjunseok/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/rokey/model/my_best_pill_2.pt" ### 알약


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")
        # --- ROS I/O ---
        self.bridge = CvBridge()
        self.pub_vis = self.create_publisher(Image, "/yolo/vis_image", 10)
        self.sub_color = self.create_subscription(Image, "/camera/camera/color/image_raw", self.on_color, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.on_depth, 10)
        self.sub_info  = self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.on_info, 10)

        # --- Latest frames / intrinsics ---
        self.depth = None
        self.K = None  # {"fx","fy","ppx","ppy"}

        # --- Robot / Gripper ---
        self.gripper2cam = np.load("T_gripper2camera.npy")
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        self.JReady = posj([0, 0, 90, 0, 90, -90])

        # --- YOLO ---
        self.model = YOLO(YOLO_MODEL_PATH)
        names = self.model.names
        self.names = names if isinstance(names, dict) else {i:n for i,n in enumerate(names)}
        self.conf_thres = 0.25

        # --- ROI/State ---
        self.roi_size = 300  # 중앙 300×300
        self.candidates = {}  # {cls_name: (conf,(u,v),(bx1,by1,bx2,by2))}
        self.awaiting_input = False
        self.user_query = None
        self.lock = threading.Lock()

        # --- Publisher thread (실시간 rqt 발행 전용) ---
        self.pub_queue: Queue[tuple] = Queue(maxsize=5)  # (vis_bgr, header)
        self.pub_thread = threading.Thread(target=self._pub_worker, daemon=True)
        self.pub_thread.start()

        # --- Motion thread 상태 ---
        self.moving = False  # 중복 이동 방지

        # --- 입력 스레드 핸들 ---
        self.input_thread = None

    # ===================== Subscriptions =====================
    def on_info(self, msg: CameraInfo):
        self.K = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def on_depth(self, msg: Image):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def on_color(self, msg: Image):
        """새 컬러 프레임마다: YOLO → ROI 300 기억 → 입력 매칭 트리거 → 시각화 큐 적재"""
        if self.depth is None or self.K is None:
            self.get_logger().info("Depth/CameraInfo not yet received, skipping frame.", throttle_duration_sec=5)
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if img is None:
            return

        import cv2
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - self.roi_size // 2), max(0, cy - self.roi_size // 2)
        x2, y2 = min(w - 1, cx + self.roi_size // 2), min(h - 1, cy + self.roi_size // 2)

        res = self.model.predict(img, verbose=False, device="cpu", conf=self.conf_thres)[0]
        boxes = getattr(res, "boxes", None)

        cand = {}
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                cls_name = self.names.get(cls_id, f"id_{cls_id}")
                u = (bx1 + bx2) // 2; v = (by1 + by2) // 2
                if x1 <= u <= x2 and y1 <= v <= y2:
                    prev = cand.get(cls_name)
                    if (prev is None) or (conf > prev[0]):
                        cand[cls_name] = (conf, (u, v), (bx1, by1, bx2, by2))
                # draw
                label = f"{cls_name} {conf*100:.1f}%"
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                yt = max(by1, th + 4)
                cv2.rectangle(img, (bx1, yt - th - 4), (bx1 + tw + 4, yt + bl - 2), (0, 255, 0), -1)
                cv2.putText(img, label, (bx1 + 2, yt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        with self.lock:
            self.candidates = cand

        # 비차단 입력 스레드: 한 번만 대기
        if (not self.awaiting_input) and self.candidates:
            self.awaiting_input = True
            threading.Thread(target=self._read_user_input_once, daemon=True).start()

        # 입력이 도착했다면 처리(이동 스레드 트리거)
        with self.lock:
            q = self.user_query
            if q is not None:
                self._try_move_if_match(q)
                self.user_query = None

        # === 발행 전용 스레드로 전달 (실시간) ===
        try:
            if self.pub_queue.full():
                _ = self.pub_queue.get_nowait()  # 가장 오래된 프레임 폐기
            self.pub_queue.put_nowait((img, msg.header))
        except Exception:
            pass

    # ===================== Publisher thread =====================
    def _pub_worker(self):
        while rclpy.ok():
            try:
                img, header = self.pub_queue.get(timeout=0.1)
            except Empty:
                continue
            msg_out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg_out.header = header
            msg_out.header.frame_id = "yolo_vis"
            self.pub_vis.publish(msg_out)

    # ===================== Non-blocking terminal input =====================
    def _read_user_input_once(self):
        try:
            classes_txt = ", ".join(sorted(self.candidates.keys()))
            print(f"[입력대기] ROI(300x300) 감지: {classes_txt}")
            user_cls = input("[입력] 이동할 클래스명을 입력하세요: ").strip()
        except EOFError:
            user_cls = ""
        with self.lock:
            self.user_query = user_cls
            self.awaiting_input = False

    # ===================== Try motion (in thread) =====================
    def _try_move_if_match(self, user_cls: str):
        if user_cls in self.candidates:
            conf, (u, v), (bx1, by1, bx2, by2) = self.candidates[user_cls]
            u_c = (bx1 + bx2) // 2
            v_c = (by1 + by2) // 2
            if not self.moving:
                self.moving = True
                threading.Thread(target=self._move_xy_thread, args=(u_c, v_c), daemon=True).start()
                self.get_logger().info(f"[OK] '{user_cls}' 일치(conf={conf:.3f}) → bbox({u_c},{v_c})로 XY 이동 시작.")
            else:
                self.get_logger().info("[SKIP] 이미 이동 중.")
        else:
            self.get_logger().info(f"[NG] '{user_cls}' 미일치. 이동 취소.")

    def _move_xy_thread(self, u, v):
        try:
            self.move_to_img_xy(u, v, use_gripper=False)
        finally:
            self.moving = False

    # ===================== Kinematics / Motion =====================
    def get_camera_pos(self, u, v, z, K):
        x = (u - K["ppx"]) * z / K["fx"]
        y = (v - K["ppy"]) * z / K["fy"]
        return (x, y, z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        Rm = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [x, y, z]
        return T

    def cam_to_base(self, Xc):
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        Xc_h = np.append(np.array(Xc), 1.0)
        Xb = base2cam @ Xc_h
        return Xb[:3]

    def move_xy_to_base(self, x_b, y_b):
        cur = get_current_posx()[0]
        target = posx([x_b, y_b, cur[2], cur[3], cur[4], cur[5]])  # Z 유지
        movel(target, 30, 30)

    def move_to_img_xy(self, u, v, use_gripper=False):
        if self.depth is None or self.K is None:
            self.get_logger().warn("Depth 또는 CameraInfo 미수신으로 이동 불가.")
            return
        h, w = self.depth.shape
        u = int(np.clip(u, 0, w - 1)); v = int(np.clip(v, 0, h - 1))
        z = float(self.depth[v, u])
        Xc = self.get_camera_pos(u, v, z, self.K)
        Xb = self.cam_to_base(Xc)
        self.move_xy_to_base(Xb[0], Xb[1])
        if use_gripper:
            self.gripper.close_gripper(); wait(1); movej(self.JReady, 30, 30); self.gripper.open_gripper(); wait(1)

if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        exit(True)

    app = TestNode()
    try:
        # 멀티스레드 실행기: 콜백 병렬 처리
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(app)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        app.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
