import cv2
import numpy as np
import onnxruntime as ort
import serial
import time
from threading import Lock
from PIL import Image, ImageDraw, ImageFont
import json
import os

class DetectionCore:
    def __init__(self):
        self.model_path = "yun1000.onnx"
        
        # 优化ONNX运行时设置
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.intra_op_num_threads = 4  # 根据实际CPU核心数调整
        
        # 创建ONNX会话
        self.net = ort.InferenceSession(self.model_path, sess_options=sess_options)

        self.categories = {
            0: "可回收物",
            1: "有害垃圾",
            2: "厨余垃圾",
            3: "其他垃圾",
            4: "其他垃圾"  # 添加ID为4的类别，您可以根据实际情况修改名称
        }

        # 多垃圾处理相关配置
        self.multi_garbage_mode = False  # 默认关闭多垃圾模式
        self.same_category_threshold = 2  # 检测到多少个相同类别的垃圾才触发多垃圾模式
        self.multi_garbage_radius = 100  # 多个垃圾的最大半径范围（像素）- 降低半径范围
        self.multi_garbage_category = None  # 当前多垃圾模式下的垃圾类别
        self.multi_garbage_items = []  # 多垃圾模式下的垃圾项目列表
        self.same_type_garbage_enabled = True  # 启用同类型垃圾批量处理
        self.multi_garbage_distance_threshold = 80  # 同类型垃圾的距离阈值，小于此值视为同一组 - 降低距离阈值
        self.garbage_groups = {}  # 用于存储分组后的垃圾 {category: [[item1, item2, ...], [item3, item4, ...]]}
        self.garbage_by_category = {
            "可回收物": [],
            "有害垃圾": [],
            "厨余垃圾": [],
            "其他垃圾": []
        }  # 按类别存储的垃圾
        self.min_garbage_size = 20  # 最小垃圾尺寸（像素），小于此值的检测结果会被过滤
        self.force_single_garbage = False  # 强制单垃圾模式，即使检测到多个垃圾也分开处理

        # ROI区域设置
        self.roi_enabled = False  # 是否启用ROI检测
        self.roi = None  # ROI区域 [x, y, width, height]
        self.roi_selecting = False  # 是否正在选择ROI
        self.roi_start_point = None  # ROI选择的起始点
        
        # 小物体检测增强
        self.small_object_mode = True  # 小物体检测增强模式
        
        # 添加静止物体检测增强 - 新增
        self.static_object_mode = True  # 启用静止物体检测增强
        self.motion_detection_frames = 3  # 检测物体运动停止的帧数（从5改为3）
        self.static_confirmation_time = 0.2  # 物体静止确认时间(秒)（从0.5改为0.2）
        self.last_static_check_time = 0  # 上次静止检查时间
        
        # 尝试加载保存的ROI设置
        try:
            self.load_roi_settings()
        except Exception as e:
            print(f"加载ROI设置失败: {e}")

        # 模型参数
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        self.anchors = [[10, 13, 16, 30, 33, 23],
                       [30, 61, 62, 45, 59, 119],
                       [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        # 串口通信
        try:
            self.ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        except:
            self.ser = None
            print("串口打开失败")

        self.camera = None
        self.lock = Lock()
        self.detection_counts = {cat: 0 for cat in self.categories.values()}
        self.last_detection = {}
        self.confirm_threshold = 0.86  # 置信度阈值

        # 添加计数器
        self.recyclable_count = 0  # 可回收物计数
        self.hazardous_count = 0   # 有害垃圾计数
        self.kitchen_count = 0      # 厨余垃圾计数
        self.other_count = 0        # 其他垃圾计数
        self.confirm_threshold = 5  # 确认阈值
        self.has_detected = False   # 是否已检测到物体
        self.last_category = None   # 上一次检测的类别

        # 添加检测状态控制
        self.is_running = False
        self.last_detected_objects = set()  # 记录上一次检测到的物体
        self.current_detected_objects = set()  # 记录当前检测到的物体
        self.object_processed = False  # 标记当前物体是否已处理

        # 添加时间控制
        self.last_detection_time = 0  # 上次检测的时间
        self.detection_interval = 5  # 检测间隔（秒）
        self.confirmed = False  # 是否已确认检测
        
        # 添加STM32响应等待
        self.waiting_stm32 = False  # 是否正在等待STM32响应
        self.stm32_timeout = 2  # STM32响应超时时间（秒）- 从5秒改为2秒
        self.stm32_response_time = 0  # STM32开始响应的时间
        self.retry_count = 0  # 添加重试计数器

        # 垃圾类别统计
        self.total_garbage_counts = {cat: 0 for cat in self.categories.values()}  # 总垃圾数量统计
        
        # 帧率计算相关变量
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # 保存上一次检测结果
        self.last_detections = []
        
        # 全平台垃圾检测相关
        self.platform_scan_mode = False  # 全平台扫描模式
        self.platform_detections = []  # 存储平台上所有检测到的垃圾
        self.current_transmission_index = 0  # 当前传输的垃圾索引
        self.transmission_complete = False  # 传输完成标志
        
        # 自动运行相关
        self.auto_mode = True  # 自动模式标志
        self.scan_duration = 3  # 扫描持续时间（秒）- 改为3秒
        self.scan_start_time = 0  # 扫描开始时间
        self.last_scan_time = 0  # 上次扫描完成的时间，用于控制扫描间隔
        self.auto_state = "scanning"  # 自动状态：scanning, transmitting, waiting
        
        # 添加物体稳定性追踪 - 修改参数以更好支持静止物体检测
        self.tracking_objects = {}  # 用于追踪物体的字典 {id: {category, position, frames, last_seen}}
        self.next_object_id = 0  # 下一个物体ID
        self.stability_threshold = 2  # 降低稳定性阈值以更快识别静止物体(从3降为2)
        self.position_tolerance = 100  # 增加位置容差(从80增加到100)，更好地适应静止物体的微小抖动
        self.max_tracking_age = 20  # 增加物体最大跟踪帧数(从15增加到20)，避免静止物体被过早移除
        
        # 优化性能的帧跳过计数器
        self.frame_skip_counter = 0

        # 添加缓冲设置以提高帧率 - 修改为每帧都检测
        self.frame_buffer = None
        self.processed_buffer = None
        self.skip_frame_count = 0
        self.max_skip_frames = 0  # 修改：不跳过帧
        self.detection_frequency = 1  # 修改：每帧都进行检测
        
        # 性能优化相关 - 对于静止物体检测，减少性能优化以提高准确性
        self.enable_performance_mode = True  # 保持启用性能优化模式
        self.process_resolution_scale = 0.85  # 提高处理分辨率(从0.7增加到0.85)以获得更准确的检测
        self.display_resolution_scale = 0.9  # 显示时的分辨率缩放比例
        
        # 帧率控制与计算
        self.target_fps = 30.0  # 目标帧率
        self.last_frame_time = 0

        # 添加更多重试设置
        self.max_retries = 3  # 最大重试次数，超过后跳过当前垃圾

    def open_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False

    def close_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def start(self):
        """开始检测"""
        self.is_running = True
        self.confirmed = False
        self.waiting_stm32 = False
        self.last_detection_time = 0
        self._reset_counts()

    def stop(self):
        """停止检测"""
        self.is_running = False
        self.confirmed = False
        self.waiting_stm32 = False
        self.last_detection_time = 0
        self._reset_counts()

    def process_frame(self):
        """处理当前帧并返回结果"""
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            print("摄像头未打开")
            return None, []

        # 读取当前帧
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("无法读取视频帧")
            return None, []
        
        # 更新FPS计数
        if hasattr(self, 'update_fps'):
            self.update_fps()
            
            # 检查串口响应
        if hasattr(self, '_check_serial_response'):
            self._check_serial_response()
            
        # 检查STM32响应超时
        if self.waiting_stm32 and hasattr(self, 'stm32_response_time') and time.time() - self.stm32_response_time > self.stm32_timeout:
            print(f"STM32响应超时({self.stm32_timeout}秒)，重试或跳过")
            self.retry_count += 1
            
            if self.retry_count >= self.max_retries:
                print(f"已达到最大重试次数({self.max_retries})，跳过当前垃圾")
                self.waiting_stm32 = False
                self.confirmed = False
                self._reset_counts()
                
                # 如果在平台扫描模式下，继续处理下一个垃圾
                if self.platform_scan_mode and not self.transmission_complete:
                    if hasattr(self, 'current_transmission_index') and hasattr(self, 'transmission_order'):
                        print(f"跳过当前垃圾，继续处理下一个垃圾")
                        self.current_transmission_index += 1
                        if hasattr(self, '_transmit_next'):
                            self._transmit_next()
            else:
                # 重新发送最后一次的指令
                if self.platform_scan_mode and hasattr(self, 'current_transmission_index') and hasattr(self, 'transmission_order'):
                    if 0 <= self.current_transmission_index < len(self.transmission_order):
                        current_category = self.transmission_order[self.current_transmission_index]
                        print(f"重试发送 {current_category} 类别信息，重试次数: {self.retry_count}")
                        
                        if hasattr(self, '_transmit_next'):
                            self._transmit_next()
                self.stm32_response_time = time.time()  # 更新等待响应的开始时间

        # 全平台扫描模式的逻辑
        if self.platform_scan_mode:
            # 自动模式处理逻辑
            if self.auto_mode:
                # 如果已完成传输并且不在等待响应状态，检查是否应该开始新的扫描
                if self.transmission_complete and not self.waiting_stm32:
                    # 检查是否达到扫描间隔时间
                    if hasattr(self, 'last_scan_time') and time.time() - self.last_scan_time > 3:  # 3秒间隔
                        print("开始新一轮自动扫描")
                        # 重置扫描相关变量
                        self.platform_detections = []
                        self.current_transmission_index = 0
                        self.transmission_complete = False
                        self.auto_state = "scanning"
                        self.scan_start_time = time.time()
                        self.last_scan_time = time.time()
                        self.transmission_order = []  # 清空传输顺序
                
                # 扫描状态下的处理
                if self.auto_state == "scanning":
                    if time.time() - self.scan_start_time > self.scan_duration:
                        print(f"完成{self.scan_duration}秒扫描，开始处理扫描结果")
                        self.auto_state = "processing"
                        
                        # 处理检测结果，并计算传输顺序
                        if hasattr(self, 'start_transmission'):
                            self.start_transmission()
                
                # 处理状态下的逻辑
                elif self.auto_state == "processing":
                    # 如果存在传输顺序但尚未开始传输，开始传输第一个垃圾
                    if hasattr(self, 'transmission_order') and self.transmission_order and self.current_transmission_index == 0 and not self.waiting_stm32:
                        print("开始传输第一个垃圾")
                        if hasattr(self, '_transmit_next'):
                            self._transmit_next()
                        self.auto_state = "waiting"
                
                # 等待状态下的逻辑 - 已经由_check_serial_response处理

        # 应用ROI裁剪，如果启用了ROI
        roi_frame = frame
        if hasattr(self, 'roi_enabled') and self.roi_enabled and hasattr(self, 'roi') and self.roi:
            x, y, w, h = self.roi
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                roi_frame = frame[y:y+h, x:x+w]
            
        # 处理鼠标事件，选择ROI区域
        if hasattr(self, 'selecting_roi') and self.selecting_roi:
            if hasattr(self, 'roi_start') and self.roi_start and hasattr(self, 'roi_end') and self.roi_end:
                temp_frame = frame.copy()
                x = min(self.roi_start[0], self.roi_end[0])
                y = min(self.roi_start[1], self.roi_end[1])
                w = abs(self.roi_start[0] - self.roi_end[0])
                h = abs(self.roi_start[1] - self.roi_end[1])
                cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(temp_frame, f"ROI: {w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                return temp_frame, []
        
                # 对当前帧进行目标检测
        detections = []
        if self.platform_scan_mode:
            # 在全平台扫描模式下使用detect_all方法
            if hasattr(self, 'detect_all'):
                detections = self.detect_all(roi_frame)
                
                # 如果在扫描状态，将检测结果添加到platform_detections
                if self.auto_state == "scanning":
                    self.platform_detections.extend(detections)
            else:
                # 如果没有detect_all方法，退回到使用detect方法
                detections = self.detect(roi_frame)
        else:
            # 常规模式下使用detect方法
            detections = self.detect(roi_frame)
            
            # 检查是否有多个相同类型的垃圾
            if hasattr(self, 'check_multi_garbage'):
                has_multi, multi_category, multi_position, is_multi = self.check_multi_garbage(detections)
                if has_multi and not self.confirmed and not self.waiting_stm32:
                    print(f"检测到多个相同类型的{multi_category}，位置: {multi_position}")
        
        # 显示检测结果
        display_frame = frame.copy()
        
        # 在显示帧上绘制检测结果
        if detections:
            for det in detections:
                if hasattr(self, 'draw_detection'):
                    self.draw_detection(display_frame, det)
                    
                # 在非全平台模式下，检查是否符合类别确认条件
                if not self.platform_scan_mode and not self.confirmed and not self.waiting_stm32:
                    category = det['category']
                    if hasattr(self, 'check_category') and self.check_category(category):
                        x, y = det['position']
                        print(f"确认检测到{category}垃圾，位置: ({x}, {y})")
                        
                        # 发送垃圾类别和位置信息
                        if hasattr(self, '_send_category'):
                            self._send_category(category, det['position'])
                            self.confirmed = True
                            self.waiting_stm32 = True
                            self.stm32_response_time = time.time()
            
            # 在全平台扫描模式下显示额外信息
            if self.platform_scan_mode:
                if hasattr(self, 'draw_platform_scan_info'):
                    self.draw_platform_scan_info(display_frame)
                
                # 绘制垃圾分组信息
                if hasattr(self, 'garbage_groups') and self.garbage_groups and hasattr(self, 'draw_garbage_groups'):
                    self.draw_garbage_groups(display_frame)
                
            # 显示FPS
            if hasattr(self, 'draw_fps'):
                self.draw_fps(display_frame)
            
            # 如果有ROI区域，显示
            if hasattr(self, 'roi_enabled') and self.roi_enabled and hasattr(self, 'roi') and self.roi and hasattr(self, 'draw_roi'):
                self.draw_roi(display_frame)
            
            # 确保返回正确的帧和检测结果
            return display_frame, detections
        
        # 如果没有检测结果，也绘制基本信息
        if self.platform_scan_mode and hasattr(self, 'draw_platform_scan_info'):
            self.draw_platform_scan_info(display_frame)
            
        if hasattr(self, 'draw_fps'):
            self.draw_fps(display_frame)
            
        if hasattr(self, 'roi_enabled') and self.roi_enabled and hasattr(self, 'roi') and self.roi and hasattr(self, 'draw_roi'):
            self.draw_roi(display_frame)
            
        return display_frame, detections

    def _check_serial_response(self):
        """检查串口响应，特别是处理STM32返回的'1'信号"""
        if not self.ser or not self.waiting_stm32:
            return
            
        try:
            # 检查串口是否有数据可读
            if self.ser.in_waiting > 0:
                # 读取所有可用数据
                response = self.ser.read(self.ser.in_waiting)
                response_str = response.decode('utf-8', errors='ignore').strip()
                
                print(f"接收到串口响应: '{response_str}'")
                
                # 明确检查是否包含'1'字符
                if '1' in response_str:
                    print("接收到STM32完成信号'1'，可以继续处理下一个垃圾")
                    self.waiting_stm32 = False  # 重置等待状态
                    self.retry_count = 0
                    self.confirmed = False      # 重要：重置确认状态，允许新的垃圾被检测
                    
                    # 重置所有计数，准备处理下一个垃圾
                    self._reset_counts()
                    
                    # 平台扫描模式下，继续处理下一个垃圾
                    if self.platform_scan_mode and not self.transmission_complete:
                        print("平台扫描模式：继续处理下一个垃圾")
                        if hasattr(self, '_transmit_next'):
                            self._transmit_next()
                        else:
                            print("缺少_transmit_next方法，无法继续传输")
                    
                    # 记录日志，便于调试
                    with open("stm32_response.log", "a") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 收到STM32响应: {response_str}\n")
                    return True
                
        except Exception as e:
            print(f"读取串口响应时出错: {e}")
            
        return False

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def cal_outputs(self, outs, nl, na, model_w, model_h, anchor_grid, stride):
        row_ind = 0
        grid = [np.zeros(1)] * nl
        for i in range(nl):
            h, w = int(model_w / stride[i]), int(model_h / stride[i])
            length = int(na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (na, 1))) * int(stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs

    def post_process_opencv(self, outputs, model_h, model_w, img_h, img_w, thred_nms=0.4, thred_cond=0.5):
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / model_w * img_w
        c_y = outputs[:, 1] / model_h * img_h
        w = outputs[:, 2] / model_w * img_w
        h = outputs[:, 3] / model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)

        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
        if len(ids) > 0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        else:
            return [], [], []

    def _get_category_id(self, category):
        """根据类别名称获取类别ID"""
        category_map = {
            "可回收物": 1,
            "有害垃圾": 2,
            "厨余垃圾": 3,
            "其他垃圾": 4
        }
        return category_map.get(category, 0)

    def check_category(self, category):
        """检查类别并计数"""
        # 如果已确认检测或正在等待STM32响应，不进行新的检测
        if self.confirmed or self.waiting_stm32:
            return False

        # 如果类别改变，重置所有计数
        if category != self.last_category:
            self._reset_counts()
            self.last_category = category

        if category == "可回收物":
            self.recyclable_count += 1
            print(f'[{time.strftime("%H:%M:%S")}] 疑似是可回收物垃圾 {self.recyclable_count} 次')
            if self.recyclable_count > 5:
                print(f'[{time.strftime("%H:%M:%S")}] 匹配确定是可回收物垃圾！！！！！')
                return True

        elif category == "有害垃圾":
            self.hazardous_count += 1
            print(f'[{time.strftime("%H:%M:%S")}] 疑似是有害垃圾 {self.hazardous_count} 次')
            if self.hazardous_count > 5:
                print(f'[{time.strftime("%H:%M:%S")}] 匹配确定是有害垃圾！！！！！')
                return True

        elif category == "厨余垃圾":
            self.kitchen_count += 1
            print(f'[{time.strftime("%H:%M:%S")}] 疑似是厨余垃圾 {self.kitchen_count} 次')
            if self.kitchen_count > 5:
                print(f'[{time.strftime("%H:%M:%S")}] 匹配确定是厨余垃圾！！！！！')
                return True

        elif category == "其他垃圾":
            self.other_count += 1
            print(f'[{time.strftime("%H:%M:%S")}] 疑似是其他垃圾 {self.other_count} 次')
            if self.other_count > 5:
                print(f'[{time.strftime("%H:%M:%S")}] 匹配确定是其他垃圾！！！！！')
                return True

        return False

    def check_multi_garbage(self, detections):
        """
        检查是否有多个相同类型的垃圾
        :param detections: 检测到的垃圾列表
        :return: (是否有多个垃圾, 垃圾类别, 位置, 是否多个相同类型)
        """
        if not detections or len(detections) < 2:
            return False, None, None, False
            
        # 对检测到的垃圾进行分组
        garbage_groups = self._group_same_type_garbage(detections)
        
        for category, groups in garbage_groups.items():
            for group in groups:
                if len(group) >= self.same_category_threshold:
                    # 找到多个相同类型的垃圾，计算中心点
                    center_x = sum(item['position'][0] for item in group) / len(group)
                    center_y = sum(item['position'][1] for item in group) / len(group)
                    position = (int(center_x), int(center_y))
                    
                    print(f"检测到{len(group)}个{category}组成的多垃圾组，位置={position}")
                    
                    # 发送多垃圾信息
                    if not self.confirmed and not self.waiting_stm32:
                        self._send_category(category, position, is_multi=True)
                        self.confirmed = True
                        self.waiting_stm32 = True
                        self.stm32_response_time = time.time()
                        
                    return True, category, position, True
                    
        return False, None, None, False

    def _reset_counts(self):
        """重置所有计数器"""
        self.recyclable_count = 0
        self.hazardous_count = 0
        self.kitchen_count = 0
        self.other_count = 0
        self.has_detected = False
        self.last_category = None

    def _send_category(self, category, position=None, is_multi=False):
        """
        发送类别信息，以及物体的位置信息
        使用统一的8位格式：
        z = 类别ID (1位)
        xcv = x坐标 (3位)
        bnm = y坐标 (3位)
        l = 处理模式 (0=单垃圾模式, 1=多垃圾模式)
        """
        if not self.ser:
            return False

        # 根据类别获取ID
        category_id = self._get_category_id(category)

        if position:
            # 提取坐标信息
            x, y = position
            
            # 确保坐标是3位数
            x_str = str(int(x)).zfill(3)[:3]  # 确保是3位，如果超过则截取
            y_str = str(int(y)).zfill(3)[:3]  # 确保是3位，如果超过则截取
            
            # 添加最后一位表示是否是多垃圾模式
            multi_bit = "1" if is_multi else "0"
            
            # 构建8位数据
            data = f"{category_id}{x_str}{y_str}{multi_bit}"
            
            print(f"发送数据: 类别={category}(ID={category_id}), 位置=({x},{y}), " +
                  f"多垃圾模式={is_multi}, 发送格式='{data}'")
        else:
            # 如果没有位置信息，使用默认坐标(中心点)
            x_str = "320"
            y_str = "240"
            multi_bit = "1" if is_multi else "0"
            data = f"{category_id}{x_str}{y_str}{multi_bit}"
            print(f"发送类别数据(默认中心位置): {category}(ID={category_id}), 多垃圾模式={is_multi}, 格式='{data}'")
        
        # 发送数据
        self.ser.write(data.encode())
        
        # 更新统计
        self.total_garbage_counts[category] += 1
        
        return True

    def detect(self, image):
        # 图像预处理 - 降低处理分辨率以提高速度
        scale_factor = 0.85 if self.small_object_mode else 0.75  # 小物体模式下减少缩放以保留更多细节
        resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        img = cv2.resize(resized_image, (self.model_w, self.model_h))
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        # 模型推理
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        # 后处理
        outs = self.cal_outputs(outs, self.nl, self.na, self.model_w, self.model_h,
                              self.anchor_grid, self.stride)

        img_h, img_w = resized_image.shape[:2]
        
        # 调整NMS阈值，小物体模式下降低NMS阈值，减少小物体被过滤的可能
        nms_threshold = 0.3 if self.small_object_mode else 0.4
        confidence_threshold = 0.6 if self.small_object_mode else 0.75  # 小物体模式下降低置信度阈值
        
        boxes, confs, ids = self.post_process_opencv(outs, self.model_h, self.model_w,
                                                   img_h, img_w, thred_nms=nms_threshold, thred_cond=confidence_threshold)

        # 根据缩放比例调整检测框的坐标
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                boxes[i][0] /= scale_factor
                boxes[i][1] /= scale_factor
                boxes[i][2] /= scale_factor
                boxes[i][3] /= scale_factor

        # 转换为检测结果列表
        detections = []
        if len(boxes) > 0:
            for box, conf, id in zip(boxes, confs, ids):
                if conf > confidence_threshold:
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    
                    # 计算检测框的大小
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    box_area = box_width * box_height
                    is_small_object = box_area < (img_h * img_w * 0.01)  # 检测框面积小于图像面积1%认为是小物体
                    
                    # 如果启用了ROI检测，检查检测点是否在ROI区域内
                    if self.roi_enabled and self.roi is not None:
                        roi_x, roi_y, roi_w, roi_h = self.roi
                        # 检查中心点是否在ROI区域内
                        if not (roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h):
                            continue  # 如果不在ROI区域内，跳过这个检测结果
                    
                    category = self.categories[id]

                    # 添加检测结果
                    detection_info = {
                            'box': box,
                            'category': category,
                            'confidence': conf,
                        'position': (center_x, center_y),
                            'is_small': is_small_object
                        }
                    detections.append(detection_info)

        # 如果没有检测到物体，重置状态
        if len(boxes) == 0:
            self._reset_counts()
            self.last_category = None
            return []
        
        # 检查是否有多个垃圾
        has_multi, multi_category, multi_position, is_multi = self.check_multi_garbage(detections)
        
        # 如果没有多个垃圾，使用普通方法处理单个垃圾
        if not has_multi and len(detections) > 0:
            # 获取置信度最高的检测结果
            best_detection = max(detections, key=lambda x: x['confidence'])
            category = best_detection['category']
            position = best_detection['position']
            
            # 使用类别确认机制进行检测
            if self.check_category(category) and not self.confirmed and not self.waiting_stm32:
                # 发送单个垃圾数据
                self._send_category(category, position, is_multi=False)
                self.confirmed = True
                self.waiting_stm32 = True
                self.stm32_response_time = time.time()
                print(f"发送单个垃圾数据: {category}, 位置={position}")

        return detections

    def draw_detection(self, frame, detection):
        box = detection['box']
        category = detection['category']
        confidence = detection['confidence']
        center_x, center_y = detection['position']
        is_small = detection.get('is_small', False)
        is_static = detection.get('is_static', False)  # 获取静止状态
        
        # 获取跟踪ID和帧数信息（如果有）
        tracking_id = detection.get('tracking_id', None)
        frames = detection.get('frames', 0)

        # 绘制检测框 - 根据物体类型使用不同颜色
        # 静止物体使用红色，小物体使用黄色，普通物体使用绿色
        if is_static:
            box_color = (0, 0, 255)  # 静止物体用红色
            box_thickness = 3  # 静止物体用粗线
        elif is_small:
            box_color = (0, 255, 255)  # 小物体用黄色
            box_thickness = 1  # 小物体用细线
        else:
            box_color = (0, 255, 0)  # 常规物体用绿色
            box_thickness = 2  # 常规物体用中等粗细线
        
        cv2.rectangle(frame,
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     box_color, box_thickness)

        # 准备标签文本（使用英文标签）
        category_map = {
            "可回收物": "Recyclable",
            "有害垃圾": "Hazardous",
            "厨余垃圾": "Kitchen",
            "其他垃圾": "Other"
        }
        label = category_map.get(category, category)
        
        # 获取类别ID (用于发送给STM32的)
        category_id = self._get_category_id(category)
        
        # 小物体显示特殊标识
        if is_small:
            label = f"{label} (S)"
            
        # 静止物体显示特殊标识
        if is_static:
            label = f"{label} [STATIC]"

        # 添加类别ID信息
        label = f"{label} (Class ID:{category_id})"
        
        # 添加计数信息到标签
        count = self.total_garbage_counts.get(category, 0)
        if count > 0:
            label = f"{label} Count:{count}"
            
        # 添加跟踪ID和稳定帧数（如果有）
        if tracking_id is not None:
            label = f"{label} [Track ID:{tracking_id}, F:{frames}]"

        # 设置字体参数 - 静止物体使用较大字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        if is_static:
            font_scale = 1.0
            thickness = 2
        elif is_small:
            font_scale = 0.6
            thickness = 1
        else:
            font_scale = 0.7
            thickness = 1

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 绘制标签背景
        cv2.rectangle(frame,
                     (int(box[0]), int(box[1]) - text_height - 10),
                     (int(box[0]) + text_width + 5, int(box[1])),
                     (0, 0, 0), -1)  # 黑色填充背景
                     
        # 绘制标签文本
        cv2.putText(frame,
                    label,
                   (int(box[0]), int(box[1]) - 5),
                    font,
                    font_scale,
                   (255, 255, 255),  # 白色文本
                    thickness)

        # 绘制中心点
        cv2.circle(frame, (center_x, center_y), 3, box_color, -1)
        
        return frame

    def update_fps(self):
        """更新帧率统计，优化计算方式"""
        current_time = time.time()
        time_diff = current_time - self.fps_time
        
        # 避免除零错误，并使用平滑平均
        if time_diff >= 0.5:  # 每0.5秒更新一次FPS计算
            # 计算当前瞬时帧率
            instant_fps = self.frame_count / time_diff
            
            # 平滑处理 (EMA - Exponential Moving Average)
            alpha = 0.3  # 平滑因子
            if self.fps == 0:
                self.fps = instant_fps
            else:
                self.fps = alpha * instant_fps + (1-alpha) * self.fps
            
            # 重置计数
            self.frame_count = 0
            self.fps_time = current_time

    def draw_fps(self, frame):
        """在画面上显示帧率"""
        current_time = time.time()
        
        # 每秒更新一次FPS计算
        if current_time - self.fps_time > 1.0:
            # 计算FPS
            elapsed_frames = self.frame_count
            elapsed_time = current_time - self.fps_time
            self.fps = elapsed_frames / elapsed_time
            
            # 重置计数器
            self.frame_count = 0
            self.fps_time = current_time

        # 绘制FPS信息
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
        
    def draw_platform_scan_info(self, frame):
        """在画面上显示平台扫描相关信息"""
        # 显示平台扫描模式状态
        if self.platform_scan_mode:
            status_text = "全平台扫描模式: 已启用"
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示自动模式状态
            if self.auto_mode:
                auto_text = f"自动模式: {self.auto_state}"
                cv2.putText(frame, auto_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示检测到的垃圾数量
            y_offset = 120
            total_items = 0
            for category, items in self.garbage_by_category.items():
                if items:
                    count_text = f"{category}: {len(items)}个"
                    cv2.putText(frame, count_text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 30
                    total_items += len(items)
            
            # 显示总数量
            if total_items > 0:
                total_text = f"总计: {total_items}个垃圾"
                cv2.putText(frame, total_text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            # 如果在传输模式，显示传输状态
            if hasattr(self, 'transmission_order') and self.transmission_order:
                trans_text = f"传输顺序: {', '.join(self.transmission_order)}"
                cv2.putText(frame, trans_text, (10, frame.shape[0] - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                if hasattr(self, 'current_transmission_index'):
                    if 0 <= self.current_transmission_index < len(self.transmission_order):
                        current_cat = self.transmission_order[self.current_transmission_index]
                        curr_text = f"当前: {current_cat}"
                        cv2.putText(frame, curr_text, (10, frame.shape[0] - 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            status_text = "全平台扫描模式: 未启用"
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return frame

    def start_platform_scan(self):
        """开始全平台垃圾扫描"""
        self.platform_scan_mode = True
        self.platform_detections = []
        self.garbage_by_category = {
            "可回收物": [],
            "有害垃圾": [],
            "厨余垃圾": [],
            "其他垃圾": []
        }
        # 重置跟踪对象
        self.tracking_objects = {}
        self.next_object_id = 0
        
        self.current_transmission_index = 0
        self.transmission_complete = False
        self.scan_start_time = time.time()
        self.last_scan_time = time.time()  # 设置上次扫描时间
        self.auto_state = "scanning"
        self.retry_count = 0  # 重置重试计数
        self.confirmed = False  # 重置确认状态
        self.waiting_stm32 = False  # 重置等待状态
        print("开始全平台垃圾扫描...")

    def stop_platform_scan(self):
        """停止全平台垃圾扫描"""
        self.platform_scan_mode = False
        self._process_platform_detections()
        print("全平台垃圾扫描完成。")

    def _process_platform_detections(self):
        """处理平台上所有检测到的垃圾，按类别分类并统计数量"""
        # 确保garbage_by_category已初始化
        if not hasattr(self, 'garbage_by_category') or self.garbage_by_category is None:
            self.garbage_by_category = {
                "可回收物": [],
                "有害垃圾": [],
                "厨余垃圾": [],
                "其他垃圾": []
            }
        else:
            # 清空之前的分类结果
            for category in self.garbage_by_category:
                self.garbage_by_category[category] = []

        # 获取稳定的检测结果
        if hasattr(self, '_get_stable_detections'):
            stable_detections = self._get_stable_detections()
        else:
            # 如果没有稳定检测方法，使用platform_detections
            stable_detections = self.platform_detections if hasattr(self, 'platform_detections') and self.platform_detections else []

        # 对每个稳定的垃圾进行分类
        for detection in stable_detections:
            category = detection['category']
            position = detection['position']
            confidence = detection['confidence']
            # 将垃圾信息添加到对应类别
            if category in self.garbage_by_category:
                self.garbage_by_category[category].append({
                    'position': position,
                    'box': detection['box'],
                    'confidence': confidence
                })

        # 按置信度对每类垃圾进行排序（从高到低）
        for category in self.garbage_by_category:
            if self.garbage_by_category[category]:
                self.garbage_by_category[category].sort(key=lambda x: x['confidence'], reverse=True)

        # 对垃圾进行分组处理
        if hasattr(self, '_group_same_type_garbage'):
            self.garbage_groups = self._group_same_type_garbage(stable_detections)
        else:
            self.garbage_groups = {}

        # 打印统计信息
        for category, items in self.garbage_by_category.items():
            if items:
                avg_conf = sum(item['confidence'] for item in items) / len(items)
                print(f"{category}: {len(items)}个, 平均置信度: {avg_conf:.3f}")
                
                # 打印分组信息
                if hasattr(self, 'garbage_groups') and category in self.garbage_groups:
                    groups = self.garbage_groups[category]
                    print(f"  分组结果: {len(groups)}个组")
                    for i, group in enumerate(groups):
                        print(f"    组 {i+1}: {len(group)}个垃圾项目")
            else:
                print(f"{category}: 0个")

    def start_transmission(self):
        """开始按照数量从少到多的顺序传输垃圾信息，对于同样数量的类别，优先传输置信度高的"""
        # 确保platform_detections已定义
        if not hasattr(self, 'platform_detections'):
            self.platform_detections = []
            
        if not self.platform_detections:
            print("没有检测到垃圾，无法开始传输")
            if hasattr(self, 'auto_mode') and self.auto_mode:
                # 自动模式下重新开始扫描
                self.platform_detections = []
                if hasattr(self, 'scan_start_time'):
                self.scan_start_time = time.time()
                if hasattr(self, 'auto_state'):
                self.auto_state = "scanning"
            return False

        # 处理检测到的垃圾
        if hasattr(self, '_process_platform_detections'):
        self._process_platform_detections()
        else:
            print("缺少_process_platform_detections方法，无法处理垃圾")
            return False

        # 确保garbage_by_category已初始化
        if not hasattr(self, 'garbage_by_category') or self.garbage_by_category is None:
            self.garbage_by_category = {
                "可回收物": [],
                "有害垃圾": [],
                "厨余垃圾": [],
                "其他垃圾": []
            }
            print("garbage_by_category未初始化，已创建默认值")

        # 计算每类垃圾的平均置信度
        category_confidence = {}
        for category, items in self.garbage_by_category.items():
            if items:
                # 计算该类别的平均置信度
                avg_conf = sum(item['confidence'] for item in items) / len(items)
                category_confidence[category] = avg_conf
            else:
                category_confidence[category] = 0

        # 过滤掉数量为0的类别
        valid_categories = [(cat, len(items), category_confidence[cat]) 
                           for cat, items in self.garbage_by_category.items() 
                           if len(items) > 0]

        # 如果没有检测到任何垃圾，直接返回
        if not valid_categories:
            print("没有检测到垃圾，无法开始传输")
            if hasattr(self, 'auto_mode') and self.auto_mode:
                # 自动模式下重新开始扫描
                self.platform_detections = []
                if hasattr(self, 'scan_start_time'):
                self.scan_start_time = time.time()
                if hasattr(self, 'auto_state'):
                self.auto_state = "scanning"
            return False

        # 按照数量从少到多排序，对于相同数量的类别，按照置信度从高到低排序
        valid_categories.sort(key=lambda x: (x[1], -x[2]))
        
        # 生成传输顺序
        self.transmission_order = [cat for cat, _, _ in valid_categories]
        
        # 初始化传输索引
        self.current_transmission_index = 0
        
        # 初始化组索引字典
        self.current_group_indices = {cat: 0 for cat, _, _ in valid_categories}
        
        # 初始化单垃圾模式的项目索引
        self.current_item_indices = {cat: 0 for cat, _, _ in valid_categories}
        
        # 重置传输完成标志
        self.transmission_complete = False
        
        # 打印传输顺序
        print(f"垃圾传输顺序: {self.transmission_order}")
        
        # 打印每个类别的垃圾数量和分组情况
        for category in self.transmission_order:
            items = self.garbage_by_category.get(category, [])
            print(f"{category}: {len(items)}个垃圾")
            
            # 打印分组情况
            if hasattr(self, 'garbage_groups') and category in self.garbage_groups:
                groups = self.garbage_groups[category]
                print(f"  分组情况: {len(groups)}个组")
                
                # 在强制单垃圾模式下，计算总垃圾数
                if hasattr(self, 'force_single_garbage') and self.force_single_garbage:
                    total_items = sum(len(group) for group in groups)
                    print(f"  强制单垃圾模式: 将处理{total_items}个独立垃圾")
                else:
                    for i, group in enumerate(groups):
                        print(f"    组 {i+1}: {len(group)}个垃圾")
                        
        # 开始传输第一个垃圾
        if hasattr(self, '_transmit_next'):
        return self._transmit_next()
        else:
            print("缺少_transmit_next方法，无法开始传输")
            return False

    def _transmit_next(self):
        """传输下一个垃圾类别的信息"""
        # 确保transmission_order已定义
        if not hasattr(self, 'transmission_order'):
            self.transmission_order = []
            return False
            
        # 确保current_transmission_index已定义
        if not hasattr(self, 'current_transmission_index'):
            self.current_transmission_index = 0
            
        # 确保current_group_indices已定义
        if not hasattr(self, 'current_group_indices'):
            self.current_group_indices = {}
            
        # 确保transmission_complete已定义
        if not hasattr(self, 'transmission_complete'):
            self.transmission_complete = False
            
        # 确保current_item_indices已定义 - 用于跟踪组内当前处理的物体索引
        if not hasattr(self, 'current_item_indices'):
            self.current_item_indices = {}
        
        if self.current_transmission_index >= len(self.transmission_order):
            print("所有垃圾信息已传输完成，重新开始扫描")
            self.transmission_complete = True
            if hasattr(self, 'auto_mode') and self.auto_mode:
                # 自动模式下重新开始扫描
                self.platform_detections = []
                if hasattr(self, 'scan_start_time'):
                self.scan_start_time = time.time()
                if hasattr(self, 'auto_state'):
                self.auto_state = "scanning"
                print("重新进入扫描状态，等待3秒后开始处理新的垃圾")
            return False

        current_category = self.transmission_order[self.current_transmission_index]
        # 确保garbage_by_category已初始化
        if not hasattr(self, 'garbage_by_category') or self.garbage_by_category is None:
            self.garbage_by_category = {
                "可回收物": [],
                "有害垃圾": [],
                "厨余垃圾": [],
                "其他垃圾": []
            }
        
        items = self.garbage_by_category.get(current_category, [])
        
        # 没有下一个类别时，最后一位标记为0
        is_last = (self.current_transmission_index == len(self.transmission_order) - 1)
        
        # 处理垃圾组
        is_multi_garbage = False
        group_position = None
        
        # 强制单垃圾模式处理逻辑
        if hasattr(self, 'force_single_garbage') and self.force_single_garbage:
            # 初始化当前类别的item索引
            if current_category not in self.current_item_indices:
                self.current_item_indices[current_category] = 0
                
            # 获取该类别的所有垃圾项目
            all_items = []
            if hasattr(self, 'garbage_groups') and current_category in self.garbage_groups:
                # 从所有组中提取单个垃圾项目
                for group in self.garbage_groups[current_category]:
                    all_items.extend(group)
            elif items:
                all_items = items
                
            # 如果没有垃圾项目，移动到下一个类别
            if not all_items:
                self.current_transmission_index += 1
                return self._transmit_next()
                
            # 获取当前要处理的垃圾项目索引
            current_item_idx = self.current_item_indices[current_category]
            
            # 检查是否已处理完该类别的所有垃圾
            if current_item_idx >= len(all_items):
                # 该类别的所有垃圾都已处理完，移动到下一个类别
                self.current_transmission_index += 1
                return self._transmit_next()
                
            # 获取当前要处理的垃圾项目
            current_item = all_items[current_item_idx]
            group_position = current_item['position']
            
            print(f"强制单垃圾模式: 处理{current_category}的第{current_item_idx+1}/{len(all_items)}个垃圾，位置={group_position}")
            
            # 更新索引，为下一次传输准备
            self.current_item_indices[current_category] = current_item_idx + 1
            
        else:
            # 原有的多垃圾分组处理逻辑
            if hasattr(self, 'garbage_groups') and current_category in self.garbage_groups and self.garbage_groups[current_category]:
                # 获取当前类别的分组
                groups = self.garbage_groups[current_category]
                
                # 检查当前组索引是否有效
                if current_category in self.current_group_indices:
                    current_group_index = self.current_group_indices[current_category]
                    
                    # 如果当前组索引有效
                    if 0 <= current_group_index < len(groups):
                        current_group = groups[current_group_index]
                        
                        # 如果组内有多个垃圾，设置多垃圾标志
                        if hasattr(self, 'same_category_threshold') and len(current_group) >= self.same_category_threshold:
                            is_multi_garbage = True
                            
                            # 计算该组的中心点作为位置
                            center_x = sum(item['position'][0] for item in current_group) / len(current_group)
                            center_y = sum(item['position'][1] for item in current_group) / len(current_group)
                            group_position = (int(center_x), int(center_y))
                            
                            print(f"发送{current_category}组内的多垃圾，组大小={len(current_group)}，位置={group_position}")
                        else:
                            # 单个垃圾，使用该垃圾的位置
                            highest_conf_item = max(current_group, key=lambda x: x['confidence'])
                            group_position = highest_conf_item['position']
                            
                            print(f"发送{current_category}组内的单个垃圾，位置={group_position}")
                        
                        # 更新组索引，为下一次传输准备
                        self.current_group_indices[current_category] = current_group_index + 1
                    else:
                        # 如果所有组都处理完了，移动到下一个类别
                        self.current_transmission_index += 1
                        return self._transmit_next()
            
            # 如果没有找到有效的组，使用默认方法（最高置信度物体）
            if group_position is None and items:
            # 已经在_process_platform_detections中按置信度排序，直接取第一个
            highest_conf_item = items[0]
                group_position = highest_conf_item['position']
            confidence = highest_conf_item['confidence']
            
                print(f"传输 {current_category} 类别的最高置信度物体: 位置={group_position}, 置信度={confidence:.3f}")
            elif not items and group_position is None:
                # 没有物品，移动到下一个类别
                self.current_transmission_index += 1
                return self._transmit_next()
        
        # 发送数据 - 使用统一格式: zxcvbnml
        # z = 类别ID, xcv = x坐标, bnm = y坐标, l = 多垃圾模式(0=单个,1=多个)
        if hasattr(self, '_send_platform_data'):
            self._send_platform_data(current_category, group_position, is_multi_garbage)
        else:
            print(f"缺少_send_platform_data方法，无法发送数据：类别={current_category}, 位置={group_position}, 多垃圾={is_multi_garbage}")
        
        # 重要：设置确认状态和等待状态，确保收到响应之前不会发送下一个垃圾
        self.confirmed = True
        self.waiting_stm32 = True
        if hasattr(self, 'stm32_response_time'):
            self.stm32_response_time = time.time()
        
        # 在强制单垃圾模式下，不更新类别索引，只有当该类别的所有垃圾都处理完才移动到下一个类别
        if not hasattr(self, 'force_single_garbage') or not self.force_single_garbage:
            # 检查是否所有组都已处理完
            if (hasattr(self, 'current_group_indices') and current_category in self.current_group_indices and 
                hasattr(self, 'garbage_groups') and current_category in self.garbage_groups and 
                self.current_group_indices[current_category] >= len(self.garbage_groups[current_category])):
                # 当前类别的所有组都处理完了，移动到下一个类别
                self.current_transmission_index += 1
            
            # 自动模式下设置为等待状态
        if hasattr(self, 'auto_mode') and self.auto_mode:
            if hasattr(self, 'auto_state'):
                self.auto_state = "waiting"
            if hasattr(self, 'waiting_stm32'):
                self.waiting_stm32 = True
            if hasattr(self, 'stm32_response_time'):
                self.stm32_response_time = time.time()
        
        print(f"正在等待STM32响应，超时时间: {self.stm32_timeout}秒")
        
        # 非自动模式直接返回True
        if not hasattr(self, 'auto_mode') or not self.auto_mode:
            return True
        
        # 自动模式下，已经设置了等待状态，不需要继续处理
        return True

    def _send_platform_data(self, category, position, more_categories):
        """
        使用统一格式发送平台数据
        zxcvbnml格式：
        z = 类别ID
        xcv = x坐标 (3位)
        bnm = y坐标 (3位)
        l = 处理模式 (0=单垃圾模式, 1=多垃圾模式)
        """
        if not self.ser:
            print("串口未连接，无法发送数据")
            return False
            
        # 获取类别ID
        category_id = self._get_category_id(category)
        
        # 提取坐标
        x, y = position
        
        # 确保坐标是3位数
        x_str = str(x).zfill(3)[:3]  # 确保是3位，如果超过则截取
        y_str = str(y).zfill(3)[:3]  # 确保是3位，如果超过则截取
        
        # 添加最后一位表示是否是多垃圾模式
        # 此处将more_categories用作多垃圾指示符
        multi_bit = "1" if more_categories else "0"
        
        # 构建8位数据
        data = f"{category_id}{x_str}{y_str}{multi_bit}"
        
        print(f"发送平台数据: 类别={category}(ID={category_id}), 位置=({x},{y}), " +
              f"多垃圾模式={more_categories}, 发送格式='{data}'")
        
        # 发送数据
        self.ser.write(data.encode())
        
        # 更新统计
        if category in self.total_garbage_counts:
            self.total_garbage_counts[category] += 1
            
        return True

    def detect_all(self, image):
        """检测图像中的所有垃圾，优化版本"""
        # 图像预处理 - 降低处理分辨率以提高速度
        scale_factor = self.process_resolution_scale  # 使用全局设置的缩放比例
        
        # 如果图像太大，额外缩放
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > 800:
            extra_scale = 800 / max_dim
            scale_factor *= extra_scale
            
        resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # 模型处理
        try:
            img = cv2.resize(resized_image, (self.model_w, self.model_h))
            img = img.astype(np.float32) / 255.0
            blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

            # 模型推理 - 性能优化
            outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

            # 后处理
            outs = self.cal_outputs(outs, self.nl, self.na, self.model_w, self.model_h,
                                  self.anchor_grid, self.stride)

            img_h, img_w = resized_image.shape[:2]
            
            # 调整NMS阈值，小物体模式下降低NMS阈值，减少小物体被过滤的可能
            nms_threshold = 0.3 if self.small_object_mode else 0.4
            confidence_threshold = 0.6 if self.small_object_mode else 0.75  # 小物体模式下降低置信度阈值
            
            boxes, confs, ids = self.post_process_opencv(outs, self.model_h, self.model_w,
                                                       img_h, img_w, thred_nms=nms_threshold, thred_cond=confidence_threshold)

            # 根据缩放比例调整检测框的坐标
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    boxes[i][0] /= scale_factor
                    boxes[i][1] /= scale_factor
                    boxes[i][2] /= scale_factor
                    boxes[i][3] /= scale_factor

            # 转换为检测结果列表
            current_detections = []
            if len(boxes) > 0:
                for box, conf, id in zip(boxes, confs, ids):
                    # 仅处理置信度足够高的物体
                    if conf > confidence_threshold:
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        
                        # 计算检测框的大小
                        box_width = box[2] - box[0]
                        box_height = box[3] - box[1]
                        box_area = box_width * box_height
                        is_small_object = box_area < (img_h * img_w * 0.01)  # 检测框面积小于图像面积1%认为是小物体
                        
                        # 如果启用了ROI检测，检查检测点是否在ROI区域内
                        if self.roi_enabled and self.roi is not None:
                            roi_x, roi_y, roi_w, roi_h = self.roi
                            # 检查中心点是否在ROI区域内
                            if not (roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h):
                                continue  # 如果不在ROI区域内，跳过这个检测结果
                                
                        category = self.categories[id]

                        # 在全平台扫描模式下，不需要确认过程，直接添加所有检测结果
                    detection_info = {
                            'box': box,
                            'category': category,
                            'confidence': conf,
                        'position': (center_x, center_y),
                            'is_small': is_small_object
                        }
                    detections.append(detection_info)
        except Exception as e:
            print(f"检测过程发生错误: {e}")
            current_detections = []

        # 更新跟踪对象
        self._update_tracking_objects(current_detections)
        
        # 获取稳定的检测结果
        stable_detections = self._get_stable_detections()
        
        return stable_detections
        
    def _update_tracking_objects(self, current_detections):
        """更新跟踪对象列表"""
        # 记录当前时间，用于静止物体检测
        current_time = time.time()
        
        # 标记所有当前跟踪对象为未匹配
        for obj_id in self.tracking_objects:
            self.tracking_objects[obj_id]['matched'] = False
            
        # 对当前检测到的每个物体
        for detection in current_detections:
            category = detection['category']
            position = detection['position']
            confidence = detection['confidence']
            is_small = detection.get('is_small', False)
            
            # 为小物体调整位置容差
            position_tolerance = self.position_tolerance
            if is_small:
                # 小物体使用更小的位置容差，使跟踪更严格
                position_tolerance = self.position_tolerance * 0.7
            
            # 尝试匹配到现有的跟踪对象
            matched = False
            best_match_id = None
            best_match_distance = float('inf')
            
            # 找出最佳匹配
            for obj_id, obj_data in self.tracking_objects.items():
                obj_position = obj_data['position']
                obj_category = obj_data['category']
                obj_is_small = obj_data.get('is_small', False)
                
                # 对小物体使用调整后的位置容差
                current_tolerance = position_tolerance
                if obj_is_small or is_small:
                    current_tolerance = position_tolerance
                
                # 计算位置差距
                distance = np.sqrt((position[0] - obj_position[0])**2 + 
                                  (position[1] - obj_position[1])**2)
                
                # 如果位置接近且类别相同，认为是同一个物体
                if distance < current_tolerance and category == obj_category:
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_id = obj_id
                        matched = True
            
            # 如果找到了最佳匹配，更新该对象
            if matched and best_match_id is not None:
                self.tracking_objects[best_match_id]['position'] = position
                self.tracking_objects[best_match_id]['confidence'] = confidence
                self.tracking_objects[best_match_id]['box'] = detection['box']
                self.tracking_objects[best_match_id]['frames'] += 1
                self.tracking_objects[best_match_id]['last_seen'] = 0
                self.tracking_objects[best_match_id]['matched'] = True
                self.tracking_objects[best_match_id]['is_small'] = is_small
                
                # 静止物体检测 - 记录位置历史
                if self.static_object_mode:
                    if 'position_history' not in self.tracking_objects[best_match_id]:
                        self.tracking_objects[best_match_id]['position_history'] = []
                    
                    # 添加当前位置到历史记录
                    self.tracking_objects[best_match_id]['position_history'].append(position)
                    
                    # 保持历史记录不超过需要的帧数
                    if len(self.tracking_objects[best_match_id]['position_history']) > self.motion_detection_frames:
                        self.tracking_objects[best_match_id]['position_history'].pop(0)
                        
                    # 检查物体是否静止
                    self._check_if_static(best_match_id)
                
            # 如果没有匹配到现有对象，创建新的跟踪对象
            elif not matched:
                # 对小物体降低稳定性阈值要求
                frames = 1
                if is_small and self.small_object_mode:
                    # 小物体模式下，小物体初始帧数给予奖励
                    frames = 2  # 这样只需要再检测1帧就能达到稳定阈值3
                
                new_obj = {
                    'category': category,
                    'position': position,
                    'confidence': confidence,
                    'box': detection['box'],
                    'frames': frames,
                    'last_seen': 0,
                    'matched': True,
                    'is_small': is_small
                }
                
                # 静止物体检测 - 初始化位置历史
                if self.static_object_mode:
                    new_obj['position_history'] = [position]
                    new_obj['is_static'] = False
                
                self.tracking_objects[self.next_object_id] = new_obj
                self.next_object_id += 1
        
        # 更新未匹配对象的last_seen计数，并移除过期对象
        objects_to_remove = []
        for obj_id, obj_data in self.tracking_objects.items():
            if not obj_data['matched']:
                obj_data['last_seen'] += 1
                # 如果物体长时间未被检测到，移除它
                # 对小物体使用更短的跟踪寿命，避免错误跟踪
                if obj_data.get('is_small', False):
                    max_age = self.max_tracking_age * 0.7  # 小物体跟踪寿命短一些
                else:
                    max_age = self.max_tracking_age
                    
                if obj_data['last_seen'] > max_age:
                    objects_to_remove.append(obj_id)
        
        # 移除过期对象
        for obj_id in objects_to_remove:
            del self.tracking_objects[obj_id]
        
        # 定期检查静止物体 - 每隔static_confirmation_time秒执行一次
        if self.static_object_mode and current_time - self.last_static_check_time > self.static_confirmation_time:
            self.last_static_check_time = current_time
            self._check_all_static_objects()
    
    def _check_if_static(self, obj_id):
        """检查指定ID的物体是否静止"""
        if obj_id not in self.tracking_objects:
            return
            
        obj_data = self.tracking_objects[obj_id]
        if 'position_history' not in obj_data or len(obj_data['position_history']) < self.motion_detection_frames:
            return
            
        # 计算位置历史记录中的最大位移
        positions = obj_data['position_history']
        max_movement = 0
        for i in range(1, len(positions)):
            movement = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                              (positions[i][1] - positions[i-1][1])**2)
            max_movement = max(max_movement, movement)
        
        # 如果最大位移小于阈值，认为物体是静止的
        movement_threshold = 8  # 像素阈值，从10降为8，提高对静止的敏感度
        if max_movement < movement_threshold:
            obj_data['is_static'] = True
            # 对静止物体提高稳定性帧数，使其更容易被检测为稳定物体
            if obj_data['frames'] < self.stability_threshold:
                obj_data['frames'] = self.stability_threshold + 1  # 确保立即达到稳定状态
            print(f"检测到静止物体 ID:{obj_id}, 类别:{obj_data['category']}, 位置:{obj_data['position']}, 移动距离:{max_movement:.2f}px")
        else:
            obj_data['is_static'] = False
    
    def _check_all_static_objects(self):
        """检查所有跟踪对象是否静止"""
        for obj_id in list(self.tracking_objects.keys()):
            self._check_if_static(obj_id)
            
    def _get_stable_detections(self):
        """获取稳定的检测结果"""
        stable_detections = []
        
        # 遍历所有跟踪对象
        for obj_id, obj_data in self.tracking_objects.items():
            # 如果物体已经连续出现足够多的帧数，认为它是稳定的
            if obj_data['frames'] >= self.stability_threshold:
                # 添加静止状态标记
                is_static = obj_data.get('is_static', False)
                
                detection = {
                    'box': obj_data['box'],
                    'category': obj_data['category'],
                    'confidence': obj_data['confidence'],
                    'position': obj_data['position'],
                    'tracking_id': obj_id,  # 添加跟踪ID以便于识别
                    'frames': obj_data['frames'],  # 添加已跟踪帧数
                    'is_static': is_static  # 添加静止状态标记
                }
                stable_detections.append(detection)
                
                # 对于静止物体，优先向舵机发送信息
                if self.static_object_mode and is_static:
                    # 关键修改：只有在不等待STM32响应时才发送新的垃圾信息
                    if self.auto_mode and not self.waiting_stm32:
                        # 检查是否已经有足够帧数确认为稳定的静止物体
                        if obj_data['frames'] > self.stability_threshold:  # 改为立即发送
                            # 向舵机发送类别和位置信息
                            self._send_category(obj_data['category'], obj_data['position'])
                            self.waiting_stm32 = True
                            self.stm32_response_time = time.time()
                            print(f"已向舵机发送静止物体信息: {obj_data['category']}, 位置: {obj_data['position']}")
        
        return stable_detections

    def draw_roi(self, frame):
        """绘制ROI区域"""
        if not self.roi_enabled or self.roi is None:
            return frame
            
        # 解包ROI参数
        x, y, w, h = self.roi
        
        # 绘制ROI边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 添加ROI文字标记
        cv2.putText(frame, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def draw_garbage_groups(self, frame):
        """在画面上绘制垃圾分组信息"""
        if not hasattr(self, 'garbage_groups') or not self.garbage_groups:
            return frame
            
        # 绘制每个类别的分组
        for category, groups in self.garbage_groups.items():
            for i, group in enumerate(groups):
                if len(group) >= self.same_category_threshold:
                    # 多垃圾组，计算中心点
                    center_x = int(sum(item['position'][0] for item in group) / len(group))
                    center_y = int(sum(item['position'][1] for item in group) / len(group))
                    
                    # 绘制多垃圾组标记
                    radius = min(self.multi_garbage_distance_threshold // 2, 100)  # 限制半径大小
                    cv2.circle(frame, (center_x, center_y), radius, 
                              (0, 165, 255), 2)  # 橙色圆圈表示多垃圾组
                    
                    # 绘制组编号
                    cv2.putText(frame, f"G{i+1}:{len(group)}个", 
                               (center_x - 20, center_y - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # 绘制组内物体之间的连线
                    for j in range(len(group)):
                        for k in range(j+1, len(group)):
                            pt1 = (int(group[j]['position'][0]), int(group[j]['position'][1]))
                            pt2 = (int(group[k]['position'][0]), int(group[k]['position'][1]))
                            cv2.line(frame, pt1, pt2, (0, 165, 255), 1, cv2.LINE_AA)
                            
        return frame
        
    def start_roi_selection(self):
        """开始ROI区域选择"""
        self.roi_selecting = True
        self.roi_start_point = None
        print("请在画面中拖动鼠标以选择检测区域")
        
    def save_roi_settings(self):
        """保存ROI设置到文件"""
        # 检查ROI是否存在且有效
        if not self.roi_enabled or self.roi is None:
            return
            
        # 创建设置数据
        settings = {
            "roi_enabled": self.roi_enabled,
            "roi": self.roi
        }
        
        # 保存到文件
        try:
            with open("roi_settings.json", "w") as f:
                json.dump(settings, f)
            print(f"ROI设置已保存: {self.roi}")
        except Exception as e:
            print(f"保存ROI设置失败: {e}")
            
    def load_roi_settings(self):
        """从文件加载ROI设置"""
        # 检查文件是否存在
        if not os.path.exists("roi_settings.json"):
            return
            
        # 加载设置
        try:
            with open("roi_settings.json", "r") as f:
                settings = json.load(f)
                
            # 应用设置
            self.roi_enabled = settings.get("roi_enabled", False)
            self.roi = settings.get("roi")
            
            if self.roi_enabled and self.roi:
                print(f"已加载保存的ROI设置: {self.roi}")
        except Exception as e:
            print(f"加载ROI设置失败: {e}")
            
    def complete_roi_selection(self, roi):
        """完成ROI区域选择"""
        self.roi = roi
        self.roi_enabled = True
        self.roi_selecting = False
        print(f"ROI区域已设置: {roi}")
        
        # 保存ROI设置
        self.save_roi_settings()
        
    def clear_roi(self):
        """清除ROI区域设置"""
        self.roi = None
        self.roi_enabled = False
        print("ROI区域已清除，将对整个画面进行检测")
        
        # 删除保存的ROI设置
        if os.path.exists("roi_settings.json"):
            try:
                os.remove("roi_settings.json")
                print("已删除保存的ROI设置")
            except Exception as e:
                print(f"删除ROI设置文件失败: {e}")
                
    def _group_same_type_garbage(self, detections):
        """
        将相同类型且距离较近的垃圾分组
        :param detections: 检测到的垃圾列表
        :return: 按类别分组后的垃圾 {category: [[item1, item2, ...], [item3, item4, ...]]}
        """
        if not hasattr(self, 'same_type_garbage_enabled') or not self.same_type_garbage_enabled or not detections:
            return {}
            
        # 确保same_category_threshold和multi_garbage_distance_threshold已定义
        if not hasattr(self, 'same_category_threshold'):
            self.same_category_threshold = 2
        if not hasattr(self, 'multi_garbage_distance_threshold'):
            self.multi_garbage_distance_threshold = 100  # 降低距离阈值，避免错误分组
            
        # 按类别初始化分组
        category_items = {}
        for detection in detections:
            category = detection['category']
            if category not in category_items:
                category_items[category] = []
            category_items[category].append(detection)
            
        # 对每个类别进行分组
        grouped_items = {}
        for category, items in category_items.items():
            # 如果该类别只有一个物体，不需要分组
            if len(items) < 2:  # 修改为2，确保单个物体也能被正确处理
                if category not in grouped_items:
                    grouped_items[category] = []
                grouped_items[category].append(items)  # 每个物体单独一组
                continue
                
            # 如果有多个物体，优先考虑尺寸差异，避免将不同物体分为一组
            groups = []
            remaining = sorted(items, key=lambda x: x['box'][2] * x['box'][3], reverse=True)  # 按面积排序
            
            while remaining:
                current_group = [remaining.pop(0)]  # 从列表中取出第一个元素作为当前组的种子
                seed_item = current_group[0]
                seed_box = seed_item['box']
                seed_area = seed_box[2] * seed_box[3]  # 种子物体的面积
                center_x, center_y = current_group[0]['position']
                
                i = 0
                while i < len(remaining):
                    item = remaining[i]
                    pos_x, pos_y = item['position']
                    item_box = item['box']
                    item_area = item_box[2] * item_box[3]
                    
                    # 计算与当前组中心的距离
                    distance = np.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2)
                    
                    # 计算面积比例，避免将大小差异明显的物体分为一组
                    area_ratio = min(seed_area, item_area) / max(seed_area, item_area)
                    
                    # 如果距离足够近且面积比例合理，加入当前组
                    # 增加面积比例判断，确保大小相似的物体才会被分到一组
                    if distance < self.multi_garbage_distance_threshold and area_ratio > 0.5:
                        current_group.append(remaining.pop(i))
                        # 更新组中心
                        center_x = sum(item['position'][0] for item in current_group) / len(current_group)
                        center_y = sum(item['position'][1] for item in current_group) / len(current_group)
                    else:
                        i += 1
                
                groups.append(current_group)
            
            grouped_items[category] = groups
            
        # 打印分组结果，便于调试
        for category, groups in grouped_items.items():
            print(f"类别 {category} 分组结果: {len(groups)}个组")
            for i, group in enumerate(groups):
                positions = [item['position'] for item in group]
                print(f"  组 {i+1}: {len(group)}个物体, 位置: {positions}")
            
        return grouped_items

    def toggle_force_single_garbage(self):
        """切换强制单垃圾模式"""
        if not hasattr(self, 'force_single_garbage'):
            self.force_single_garbage = False
            
        self.force_single_garbage = not self.force_single_garbage
        status = "启用" if self.force_single_garbage else "禁用"
        print(f"强制单垃圾模式: {status}")
        
        # 如果切换到强制单垃圾模式，重置当前项目索引
        if self.force_single_garbage:
            self.current_item_indices = {cat: 0 for cat in self.transmission_order} if hasattr(self, 'transmission_order') else {}
            
        return self.force_single_garbage