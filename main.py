import cv2
import numpy as np
import time
import serial
import sys
import argparse
import threading
from detection_core import DetectionCore

# 检查是否安装了Tkinter和PIL
HAS_TKINTER = False
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import PIL.Image, PIL.ImageTk
    HAS_TKINTER = True
except ImportError:
    pass

# 全局变量用于存储当前鼠标位置
current_mouse_pos = (0, 0)

# 鼠标事件回调函数，用于ROI选择
def mouse_callback(event, x, y, flags, param):
    global current_mouse_pos
    detector = param
    
    # 更新当前鼠标位置
    current_mouse_pos = (x, y)
    
    # 如果正在选择ROI区域
    if hasattr(detector, 'roi_selecting') and detector.roi_selecting:
        # 开始选择
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.roi_start_point = (x, y)
            print(f"开始选择ROI: 起始点 ({x}, {y})")
        # 拖动中
        elif event == cv2.EVENT_MOUSEMOVE and detector.roi_start_point:
            # 这里只更新鼠标位置，绘制工作在主循环中进行
            pass
        # 结束选择
        elif event == cv2.EVENT_LBUTTONUP and detector.roi_start_point:
            # 计算ROI区域
            x1, y1 = detector.roi_start_point
            width = abs(x - x1)
            height = abs(y - y1)
            x_start = min(x1, x)
            y_start = min(y1, y)
            
            # 设置ROI区域
            if width > 10 and height > 10:  # 确保选择区域不太小
                roi = [x_start, y_start, width, height]
                detector.complete_roi_selection(roi)
                print(f"完成ROI选择: {roi}")
            else:
                detector.roi_selecting = False
                detector.roi_start_point = None
                print("ROI区域过小，已取消选择")

# Tkinter界面类
class GarbageClassificationApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("垃圾分类系统 - Tkinter界面")
        self.root.geometry("1000x700")
        
        # 创建检测核心实例
        self.detector = DetectionCore()
        
        # 设置参数
        self.detector.stability_threshold = args.stability
        self.detector.position_tolerance = args.tolerance
        self.detector.max_tracking_age = args.tracking_age
        self.detector.small_object_mode = not args.no_small_object_mode
        
        # 重新配置串口
        try:
            # 关闭旧串口连接（如果存在）
            if hasattr(self.detector, 'ser') and self.detector.ser:
                self.detector.ser.close()
                
            # 创建新串口连接
            if args.port:
                port = args.port
            elif 'win' in sys.platform:
                port = 'COM3'
            else:
                port = '/dev/ttyUSB0'
            
            self.detector.ser = serial.Serial(
                port=port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            print(f"串口配置成功: {port}, 波特率: 115200, 8N1")
            self.serial_status = True
        except Exception as e:
            self.detector.ser = None
            self.serial_status = False
            print(f"串口打开失败: {e}")
        
        # 创建UI界面
        self.create_widgets()
        
        # 视频流相关变量
        self.is_running = False
        self.thread = None
        self.current_image = None
        self.roi_selecting = False
        self.roi_start_point = None
        self.current_mouse_pos = (0, 0)
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左侧视频显示区域
        self.video_frame = ttk.LabelFrame(main_frame, text="摄像头视图")
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")
        
        # 视频画布
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 右侧控制区域
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # 控制按钮
        ttk.Button(control_frame, text="开始摄像头", command=self.start_camera).pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="停止摄像头", command=self.stop_camera).pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="选择ROI区域", command=self.start_roi_selection).pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="清除ROI区域", command=self.clear_roi).pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="全平台扫描", command=self.start_platform_scan).pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="停止平台扫描", command=self.stop_platform_scan).pack(fill="x", padx=5, pady=5)
        
        # 状态显示区域
        status_frame = ttk.LabelFrame(main_frame, text="系统状态")
        status_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # 串口状态
        self.serial_status_var = tk.StringVar(value="串口: 未连接")
        ttk.Label(status_frame, textvariable=self.serial_status_var).pack(anchor="w", padx=5, pady=2)
        
        # ROI状态
        self.roi_status_var = tk.StringVar(value="ROI: 未启用")
        ttk.Label(status_frame, textvariable=self.roi_status_var).pack(anchor="w", padx=5, pady=2)
        
        # 扫描模式状态
        self.scan_mode_var = tk.StringVar(value="扫描模式: 未启动")
        ttk.Label(status_frame, textvariable=self.scan_mode_var).pack(anchor="w", padx=5, pady=2)
        
        # 识别结果区域
        result_frame = ttk.LabelFrame(main_frame, text="识别结果")
        result_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")
        
        # 结果显示框
        self.result_text = tk.Text(result_frame, width=25, height=20)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 设置网格权重
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 键盘事件绑定
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.bind("r", lambda e: self.start_roi_selection())
        self.root.bind("R", lambda e: self.start_roi_selection())
        self.root.bind("c", lambda e: self.clear_roi())
        self.root.bind("C", lambda e: self.clear_roi())
        
        # 程序关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 更新状态显示
        self.update_status()
    
    def update_status(self):
        # 更新串口状态
        if hasattr(self.detector, 'ser') and self.detector.ser:
            self.serial_status_var.set(f"串口: 已连接 ({self.detector.ser.port}, 115200, 8N1)")
        else:
            self.serial_status_var.set("串口: 未连接")
        
        # 更新ROI状态
        if self.detector.roi_enabled:
            self.roi_status_var.set(f"ROI: 已启用 {self.detector.roi}")
        else:
            self.roi_status_var.set("ROI: 未启用")
        
        # 更新扫描模式
        if self.detector.platform_scan_mode:
            self.scan_mode_var.set("扫描模式: 全平台扫描中")
        else:
            self.scan_mode_var.set("扫描模式: 未启动")
    
    def start_camera(self):
        if self.is_running:
            return
        
        # 打开摄像头
        if not self.detector.open_camera():
            messagebox.showerror("错误", "无法打开摄像头!")
            return
        
        # 启动检测器
        self.detector.start()
        self.is_running = True
        
        # 启动视频线程
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.update_status()
        self.log_message("系统已启动，开始检测")
    
    def stop_camera(self):
        if not self.is_running:
            return
        
        # 停止检测器
        self.detector.stop()
        self.is_running = False
        
        # 等待线程结束
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # 关闭摄像头
        self.detector.close_camera()
        
        # 清空画布
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text="摄像头已停止", fill="white", font=("Arial", 20))
        
        self.update_status()
        self.log_message("系统已停止")
    
    def video_loop(self):
        try:
            while self.is_running:
                # 处理当前帧
                frame, detections = self.detector.process_frame()
                
                if frame is not None:
                    # 转换为Tkinter可显示的格式
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = PIL.Image.fromarray(image)
                    photo = PIL.ImageTk.PhotoImage(image=image)
                    
                    # 更新画布
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.current_image = photo  # 保留引用防止垃圾回收
                    
                    # 如果正在选择ROI，显示实时ROI框
                    if self.roi_selecting and self.roi_start_point:
                        x1, y1 = self.roi_start_point
                        x2, y2 = self.current_mouse_pos
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                    
                    # 更新状态显示
                    self.update_status()
                    
                    # 如果有检测结果，更新结果显示
                    if detections:
                        self.display_results(detections)
                
                # 适当延时，减轻CPU负担
                time.sleep(0.01)
        except Exception as e:
            print(f"视频线程异常: {e}")
            self.stop_camera()
    
    def start_roi_selection(self):
        if not self.is_running:
            messagebox.showinfo("提示", "请先启动摄像头")
            return
        
        self.detector.start_roi_selection()
        self.roi_selecting = True
        self.log_message("开始选择ROI区域，请用鼠标拖拽选择区域")
    
    def clear_roi(self):
        if not self.is_running:
            return
        
        self.detector.clear_roi()
        self.roi_selecting = False
        self.roi_start_point = None
        self.update_status()
        self.log_message("ROI区域已清除")
    
    def start_platform_scan(self):
        if not self.is_running:
            messagebox.showinfo("提示", "请先启动摄像头")
            return
        
        self.detector.start_platform_scan()
        self.update_status()
        self.log_message("全平台扫描模式已启动! 系统将自动扫描3秒。")
    
    def stop_platform_scan(self):
        if not self.is_running:
            return
        
        self.detector.stop_platform_scan()
        self.update_status()
        self.log_message("全平台扫描已停止")
    
    def on_mouse_down(self, event):
        if self.roi_selecting:
            self.roi_start_point = (event.x, event.y)
    
    def on_mouse_move(self, event):
        self.current_mouse_pos = (event.x, event.y)
    
    def on_mouse_up(self, event):
        if self.roi_selecting and self.roi_start_point:
            x1, y1 = self.roi_start_point
            x2, y2 = event.x, event.y
            
            # 计算ROI区域
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x_start = min(x1, x2)
            y_start = min(y1, y2)
            
            # 设置ROI区域
            self.detector.complete_roi_selection([x_start, y_start, width, height])
            self.roi_selecting = False
            self.roi_start_point = None
            self.update_status()
            self.log_message(f"ROI区域已设置: [{x_start}, {y_start}, {width}, {height}]")
    
    def display_results(self, detections):
        # 清空结果文本框
        self.result_text.delete(1.0, tk.END)
        
        # 统计各类别数量
        categories = {}
        for det in detections:
            category = det["category"]
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
        
        # 显示检测结果
        self.result_text.insert(tk.END, f"检测时间: {time.strftime('%H:%M:%S')}\n")
        self.result_text.insert(tk.END, f"检测到 {len(detections)} 个物体\n\n")
        
        for category, count in categories.items():
            self.result_text.insert(tk.END, f"{category}: {count} 个\n")
        
        self.result_text.insert(tk.END, "\n详细信息:\n")
        for i, det in enumerate(detections):
            category = det["category"]
            confidence = det["confidence"]
            self.result_text.insert(tk.END, f"{i+1}. {category} (置信度: {confidence:.2f})\n")
    
    def log_message(self, message):
        self.result_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.result_text.see(tk.END)  # 滚动到最后
    
    def on_close(self):
        # 停止摄像头
        if self.is_running:
            self.stop_camera()
        
        # 关闭窗口
        self.root.destroy()

# OpenCV界面主函数
def opencv_main(args):
    # 创建检测核心实例前先修改串口参数
    # 修改detection_core.py中的串口配置
    DetectionCore.original_init = DetectionCore.__init__
    
    def custom_init(self):
        # 调用原始初始化函数
        DetectionCore.original_init(self)
        
        # 设置稳定性阈值
        self.stability_threshold = args.stability
        print(f"设置物体稳定性阈值为: {self.stability_threshold} 帧")
        
        # 设置位置容差
        self.position_tolerance = args.tolerance
        print(f"设置位置容差为: {self.position_tolerance} 像素")
        
        # 设置跟踪帧龄
        self.max_tracking_age = args.tracking_age
        print(f"设置物体最大跟踪帧龄为: {self.max_tracking_age} 帧")
        
        # 设置小物体模式
        self.small_object_mode = not args.no_small_object_mode
        print(f"小物体检测增强模式: {'已启用' if self.small_object_mode else '已禁用'}")
        
        # 重新配置串口
        try:
            # 关闭旧串口连接（如果存在）
            if hasattr(self, 'ser') and self.ser:
                self.ser.close()
                
            # 创建新串口连接
            # 修改为与STM32匹配的串口参数
            # 波特率设置为115200，匹配STM32高速串口通信
            # COM端口需要根据实际情况修改
            if args.port:
                port = args.port
            elif 'win' in sys.platform:
                # Windows系统
                port = 'COM3'  # 请根据实际COM端口修改
            else:
                # Linux/嵌入式系统
                port = '/dev/ttyUSB0'  # 对于USB转串口，修改为实际设备
            
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,    # 将波特率修改为115200
                bytesize=serial.EIGHTBITS,   # 8数据位
                parity=serial.PARITY_NONE,   # 无校验位
                stopbits=serial.STOPBITS_ONE, # 1停止位
                timeout=0.1
            )
            print(f"Serial port configured: {port}, baudrate: 115200, 8N1")
        except Exception as e:
            self.ser = None
            print(f"Failed to open serial port: {e}")
    
    # 替换初始化方法
    DetectionCore.__init__ = custom_init
    
    # 创建检测核心实例
    detector = DetectionCore()
    
    # 打开摄像头
    if not detector.open_camera():
        print("Failed to open camera!")
        return
    
    # 开始检测
    detector.start()
    print("Garbage Classification System Started!")
    
    # 默认启动全平台扫描模式
    detector.start_platform_scan()
    print("Auto platform scanning mode activated! System will scan for 3 seconds automatically.")
    
    # 打印ROI功能说明
    print("\n区域检测功能说明:")
    print("1. 按 'R' 键开始选择检测区域 (ROI)")
    print("2. 在画面中用鼠标拖动选择要检测的区域")
    print("3. 按 'C' 键清除检测区域 (恢复全画面检测)")
    print("4. 按 ESC 键退出程序\n")
    
    # 打印性能优化信息
    print("性能优化说明:")
    print("1. 已启用性能优化模式，检测分辨率降低以提高帧率")
    print("2. 已降低稳定性阈值（从5帧降为3帧）以更快识别垃圾")
    print("3. 已增加位置容差（从50像素增加到80像素）以更好跟踪移动物体")
    print("4. 跳帧处理：每2帧进行一次完整检测，提高整体流畅度")
    print("5. 已启用小物体检测增强模式:")
    print("   - 小物体会用黄色框标出，标签标记为(S)")
    print("   - 降低NMS阈值和置信度，提高小物体检测能力")
    print("   - 降低图像缩放比例，保留更多细节")
    print("   - 给予小物体稳定性帧数加成，更快确认稳定")
    print("6. 如果仍需调整性能，可使用以下命令行参数:")
    print("   --stability=N   设置稳定性阈值，默认3，小值更快检测，大值更稳定")
    print("   --tolerance=N   设置位置容差，默认80，大值更容易跟踪移动物体")
    print("   --tracking-age=N 设置跟踪帧龄，默认15，大值跟踪更持久")
    print("   --no-small-object-mode 禁用小物体检测增强模式\n")
    
    try:
        # 创建窗口
        cv2.namedWindow("Garbage Classification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Garbage Classification", 800, 600)
        
        # 设置鼠标回调
        cv2.setMouseCallback("Garbage Classification", mouse_callback, detector)
        
        # 主循环
        while True:
            # 处理当前帧
            frame, detections = detector.process_frame()
            
            if frame is not None:
                # 如果正在选择ROI，显示实时ROI框
                if hasattr(detector, 'roi_selecting') and detector.roi_selecting and hasattr(detector, 'roi_start_point') and detector.roi_start_point:
                    # 绘制临时ROI框
                    x1, y1 = detector.roi_start_point
                    x2, y2 = current_mouse_pos
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 添加提示文字
                    cv2.putText(frame, "正在选择ROI区域", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示帧
                cv2.imshow("Garbage Classification", frame)
                
                # 显示系统状态
                status_frame = np.zeros((80, 800, 3), dtype=np.uint8)
                status_text = f"Serial: {'Connected (115200, 8N1)' if detector.ser else 'Disconnected'}"
                mode_text = f"Mode: Fully Automatic (Stability: {detector.stability_threshold} frames)"
                roi_status = f"ROI: {'Enabled' if hasattr(detector, 'roi_enabled') and detector.roi_enabled else 'Disabled'}"
                help_text = "System running automatically. R: Set ROI, C: Clear ROI, ESC: Exit"
                
                cv2.putText(status_frame, status_text, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if detector.ser else (0, 0, 255), 2)
                cv2.putText(status_frame, mode_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(status_frame, roi_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(status_frame, help_text, (300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 添加中文状态说明（在主窗口上）
                cn_roi_status = f"区域检测: {'已启用' if hasattr(detector, 'roi_enabled') and detector.roi_enabled else '未启用'}"
                cn_roi_hint = "正在选择区域，请拖动鼠标" if (hasattr(detector, 'roi_selecting') and detector.roi_selecting) else "按R键选择区域，C键清除区域"
                
                # 在主窗口底部添加中文说明
                cv2.rectangle(frame, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, cn_roi_status, (10, frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, cn_roi_hint, (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
                cv2.imshow("System Status", status_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == ord('r') or key == ord('R'):  # R键开始选择ROI
                detector.start_roi_selection()
                print("开始选择ROI区域，请用鼠标拖拽选择区域")
            elif key == ord('c') or key == ord('C'):  # C键清除ROI
                detector.clear_roi()
                
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        # 停止检测
        detector.stop()
        
        # 关闭摄像头
        detector.close_camera()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        print("System closed")

# Tkinter界面主函数
def tkinter_main(args):
    if not HAS_TKINTER:
        print("错误: 没有安装Tkinter或PIL库，无法使用Tkinter界面")
        print("请安装必要的库: pip install pillow")
        print("正在启动OpenCV界面作为替代...")
        opencv_main(args)
        return
    
    # 创建Tkinter根窗口
    root = tk.Tk()
    app = GarbageClassificationApp(root, args)
    
    # 启动主循环
    root.mainloop()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='垃圾分类系统')
    parser.add_argument('--ui', type=str, choices=['opencv', 'tkinter'], default=None, 
                      help='选择UI界面类型: opencv (原始界面) 或 tkinter (新界面)')
    parser.add_argument('--stability', type=int, default=3, 
                      help='物体稳定性阈值（帧数），默认3，较小的值可以更快检测，较大的值更稳定')
    parser.add_argument('--tolerance', type=int, default=80, 
                      help='位置容差（像素），默认80，较大的值在物体移动时更容易跟踪')
    parser.add_argument('--tracking-age', type=int, default=15, 
                      help='物体跟踪最大帧龄，默认15，较大的值能跟踪更长时间')
    parser.add_argument('--port', type=str, default=None, 
                      help='串口端口（例如：COM3或/dev/ttyUSB0）')
    parser.add_argument('--no-small-object-mode', action='store_true', 
                      help='禁用小物体检测增强模式')
    
    args = parser.parse_args()
    
    # 如果没有通过命令行指定界面类型，则显示选择对话框
    if args.ui is None:
        # 检查是否安装了Tkinter
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
            
            # 创建选择界面
            def show_ui_selector():
                root = tk.Tk()
                root.title("垃圾分类系统 - 界面选择")
                root.geometry("500x400")
                root.resizable(False, False)
                
                # 导入PIL用于处理图像
                try:
                    from PIL import Image, ImageTk, ImageDraw, ImageFilter
                    has_pil = True
                except ImportError:
                    has_pil = False
                
                # 应用深色主题
                style = ttk.Style()
                try:
                    # 尝试使用更现代的主题
                    available_themes = style.theme_names()
                    if 'clam' in available_themes:
                        style.theme_use('clam')
                    
                    # 配置现代化外观
                    style.configure('TFrame', background='#2E3B4E')
                    style.configure('TLabel', background='#2E3B4E', foreground='white')
                    style.configure('Header.TLabel', background='#2E3B4E', foreground='#64B5F6', font=('Arial', 22, 'bold'))
                    style.configure('SubHeader.TLabel', background='#2E3B4E', foreground='#90CAF9', font=('Arial', 12))
                    
                    # 定制按钮样式
                    style.configure('Modern.TButton', 
                                   background='#1976D2', 
                                   foreground='white', 
                                   font=('Arial', 11, 'bold'),
                                   padding=10,
                                   relief='flat')
                    style.map('Modern.TButton',
                            background=[('active', '#1565C0'), ('pressed', '#0D47A1')],
                            foreground=[('active', 'white'), ('pressed', 'white')])
                            
                    style.configure('Secondary.TButton', 
                                   background='#26A69A', 
                                   foreground='white', 
                                   font=('Arial', 11, 'bold'),
                                   padding=10)
                    style.map('Secondary.TButton',
                            background=[('active', '#00897B'), ('pressed', '#00796B')],
                            foreground=[('active', 'white'), ('pressed', 'white')])
                            
                    # 描述标签样式                   
                    style.configure('Desc.TLabel', background='#2E3B4E', foreground='#B0BEC5', font=('Arial', 10))
                except Exception as e:
                    print(f"应用主题时出错: {e}")
                
                # 创建渐变背景
                if has_pil:
                    try:
                        # 创建渐变背景
                        bg_image = Image.new('RGB', (500, 400), color='#2E3B4E')
                        draw = ImageDraw.Draw(bg_image)
                        
                        # 创建渐变效果
                        for i in range(400):
                            # 从顶部到底部的渐变
                            r = int(46 - (i/400) * 15)  # 从46到31
                            g = int(59 - (i/400) * 15)  # 从59到44
                            b = int(78 - (i/400) * 15)  # 从78到63
                            draw.line([(0, i), (500, i)], fill=(r, g, b))
                        
                        # 添加轻微的噪点纹理效果
                        bg_image = bg_image.filter(ImageFilter.SMOOTH)
                        
                        # 创建圆角矩形
                        def rounded_rectangle(self, xy, radius=20, fill=None, outline=None, width=0):
                            x1, y1, x2, y2 = xy
                            diameter = 2 * radius
                            
                            # 绘制四个角
                            draw.ellipse((x1, y1, x1 + diameter, y1 + diameter), fill=fill, outline=outline, width=width)
                            draw.ellipse((x2 - diameter, y1, x2, y1 + diameter), fill=fill, outline=outline, width=width)
                            draw.ellipse((x1, y2 - diameter, x1 + diameter, y2), fill=fill, outline=outline, width=width)
                            draw.ellipse((x2 - diameter, y2 - diameter, x2, y2), fill=fill, outline=outline, width=width)
                            
                            # 填充中心部分
                            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill, outline=None)
                            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill, outline=None)
                        
                        # 绘制主面板背景
                        rounded_rectangle(draw, (50, 50, 450, 350), radius=20, fill="#3A4A60")
                        
                        # 转换为Tkinter可用的格式
                        bg_photo = ImageTk.PhotoImage(bg_image)
                        
                        # 创建Canvas并放置背景图片
                        canvas = tk.Canvas(root, width=500, height=400, highlightthickness=0)
                        canvas.pack(fill="both", expand=True)
                        canvas.create_image(0, 0, image=bg_photo, anchor="nw")
                        
                        # 在Canvas上创建主框架
                        main_frame = ttk.Frame(canvas)
                        canvas.create_window(250, 200, window=main_frame, width=360, height=280)
                    except Exception as e:
                        print(f"创建背景时出错: {e}")
                        has_pil = False
                
                # 如果PIL创建背景失败，使用传统布局
                if not has_pil:
                    root.configure(bg='#2E3B4E')
                    main_frame = ttk.Frame(root, padding="20")
                    main_frame.pack(fill="both", expand=True)
                
                # 标题标签
                title_label = ttk.Label(
                    main_frame, 
                    text="垃圾分类系统", 
                    style="Header.TLabel"
                )
                title_label.pack(pady=(20, 5))
                
                # 副标题
                subtitle_label = ttk.Label(
                    main_frame, 
                    text="请选择您要使用的界面类型", 
                    style="SubHeader.TLabel"
                )
                subtitle_label.pack(pady=5)
                
                # 分隔线
                separator = ttk.Separator(main_frame, orient='horizontal')
                separator.pack(fill='x', padx=20, pady=15)
                
                # 按钮框架
                button_frame = ttk.Frame(main_frame)
                button_frame.pack(pady=10)
                
                # 选择变量
                selected_ui = [None]
                
                # OpenCV按钮
                def select_opencv():
                    selected_ui[0] = 'opencv'
                    # 添加按钮动画效果
                    opencv_btn.state(['pressed'])
                    root.after(100, root.destroy)
                
                opencv_btn = ttk.Button(
                    button_frame,
                    text="OpenCV界面",
                    command=select_opencv,
                    width=25,
                    style="Modern.TButton"
                )
                opencv_btn.pack(pady=(5, 2))
                
                # 显示OpenCV描述
                opencv_label = ttk.Label(
                    button_frame, 
                    text="原始界面，简洁高效，适合性能优先场景", 
                    style="Desc.TLabel"
                )
                opencv_label.pack(pady=(0, 10))
                
                # Tkinter按钮
                def select_tkinter():
                    selected_ui[0] = 'tkinter'
                    # 添加按钮动画效果
                    tkinter_btn.state(['pressed'])
                    root.after(100, root.destroy)
                
                tkinter_btn = ttk.Button(
                    button_frame,
                    text="Tkinter界面",
                    command=select_tkinter,
                    width=25,
                    style="Secondary.TButton"
                )
                tkinter_btn.pack(pady=(10, 2))
                
                # 显示Tkinter描述
                tkinter_label = ttk.Label(
                    button_frame, 
                    text="现代图形界面，功能完整，操作便捷", 
                    style="Desc.TLabel"
                )
                tkinter_label.pack()
                
                # 版权信息
                copyright_label = ttk.Label(
                    main_frame, 
                    text="© 2023 垃圾分类系统 v1.0", 
                    font=("Arial", 8),
                    foreground="#78909C"
                )
                copyright_label.pack(side="bottom", pady=10)
                
                # 居中窗口
                root.update_idletasks()
                width = root.winfo_width()
                height = root.winfo_height()
                x = (root.winfo_screenwidth() // 2) - (width // 2)
                y = (root.winfo_screenheight() // 2) - (height // 2)
                root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
                
                # 保存图片引用防止垃圾回收
                if has_pil:
                    root.bg_photo = bg_photo
                    
                # 添加键盘快捷键
                root.bind('<o>', lambda e: select_opencv())
                root.bind('<t>', lambda e: select_tkinter())
                root.bind('<Escape>', lambda e: root.destroy())
                
                # 运行主循环
                root.mainloop()
                return selected_ui[0]
            
            # 显示界面选择对话框
            selected_ui = show_ui_selector()
            
            # 如果用户选择了界面类型
            if selected_ui:
                args.ui = selected_ui
            else:
                # 用户关闭了窗口而没有选择，默认使用OpenCV界面
                args.ui = 'opencv'
                
        except ImportError:
            # 如果没有Tkinter，直接使用OpenCV界面
            print("未检测到Tkinter库，默认使用OpenCV界面...")
            args.ui = 'opencv'
    
    # 根据UI参数选择界面
    if args.ui == 'tkinter':
        print("正在启动Tkinter界面...")
        tkinter_main(args)
    else:
        print("正在启动OpenCV界面...")
        opencv_main(args)

if __name__ == "__main__":
    main()