# 基于RDK X5的垃圾分类系统

## 项目简介
本项目基于深度学习视觉模型，结合RDK X5硬件平台，实现高效准确的垃圾识别与分类。系统通过摄像头采集图像，利用ONNX模型进行垃圾检测，并通过串口与下位机（如STM32）通信，实现自动化分拣。

## 主要功能
- 支持多类别垃圾识别（可回收物、有害垃圾、厨余垃圾、其他垃圾）
- 支持小物体和静止物体检测增强
- 支持ROI区域选择与保存
- 支持全平台自动扫描与批量传输
- 支持与下位机串口通信，自动控制机械臂分拣
- 提供OpenCV和Tkinter两种用户界面

## 依赖环境
- Python 3.7及以上
- OpenCV (`opencv-python`)
- NumPy
- onnxruntime
- pyserial
- Pillow

安装依赖：
```bash
pip install opencv-python numpy onnxruntime pyserial pillow
```

## 使用方法
1. **准备模型文件**：将`yun1000.onnx`模型文件放在项目根目录。
2. **运行主程序**：
   - OpenCV界面：
     ```bash
     python main.py --ui opencv
     ```
   - Tkinter界面：
     ```bash
     python main.py --ui tkinter
     ```
3. **参数说明**：
   - `--stability` 物体稳定性阈值（帧数，默认3）
   - `--tolerance` 位置容差（像素，默认80）
   - `--tracking-age` 跟踪最大帧龄（默认15）
   - `--port` 串口端口（如COM3或/dev/ttyUSB0）
   - `--no-small-object-mode` 禁用小物体检测增强

## 界面说明
- **OpenCV界面**：适合性能优先场景，支持键盘快捷键（R选择区域，C清除区域，ESC退出）。
- **Tkinter界面**：现代图形界面，支持鼠标操作ROI选择、全平台扫描、结果实时显示。

## 文件说明
- `main.py`：主程序入口，包含界面与流程控制
- `detection_core.py`：核心检测与分拣逻辑
- `yun1000.onnx`：ONNX格式的垃圾分类模型（需自行准备）

## 联系方式
如有问题或建议，请通过GitHub Issue联系作者。 