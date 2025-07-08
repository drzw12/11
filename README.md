# Garbage Classification System

This repository contains a garbage classification system that uses computer vision to detect and classify different types of waste.

## Files
- `detection_core.py`: Core detection functionality that handles garbage classification
- `main.py`: User interface implementation with both OpenCV and Tkinter options

## Features
- Real-time garbage detection and classification
- Support for multiple garbage types (recyclable, hazardous, kitchen waste, other)
- ROI (Region of Interest) selection for targeted detection
- Platform scanning mode for batch processing
- Small object detection optimization
- Serial communication with external hardware

## Requirements
- Python 3.6+
- OpenCV
- NumPy
- ONNX Runtime
- PySerial
- PIL (for Tkinter UI) 
