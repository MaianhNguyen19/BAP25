import sys
import os
import re
import time
import serial
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QFileDialog, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg


class SerialReader(QThread):
    data_received = pyqtSignal(list)
    finished = pyqtSignal()
    
    def __init__(self, port, baudrate, duration, num_channels):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.duration = duration
        self.num_channels = num_channels
        self.running = False

    def run(self):
        ser = serial.Serial(self.port, self.baudrate, timeout=1)
        ser.reset_input_buffer()
        start_time = time.time()
        self.running = True

        try:
            while self.running and (time.time() - start_time) < self.duration:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line or not re.match(r'^\d+,', line):
                    continue
                parts = line.split(',')
                if len(parts) != 1 + self.num_channels:
                    continue
                try:
                    ts = int(parts[0])
                    chans = list(map(int, parts[1:]))
                    self.data_received.emit([ts] + chans)
                except ValueError:
                    continue
        finally:
            ser.close()
            self.finished.emit()

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 Multi-Channel Recorder")
        self.resize(900, 700)

        self.data = []
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # === Controls ===
        self.filename_input = QLineEdit("mcp3208_6ch.csv")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(20)

        self.channel_count_input = QSpinBox()
        self.channel_count_input.setRange(1, 6)
        self.channel_count_input.setValue(6)
        self.channel_count_input.valueChanged.connect(self.update_channel_inputs)

        self.channel_name_inputs = []
        self.channel_form = QFormLayout()

        for i in range(6):
            inp = QLineEdit(f"ch{i}")
            self.channel_form.addRow(QLabel(f"Channel {i} name:"), inp)
            self.channel_name_inputs.append(inp)

        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)

        self.status_label = QLabel("Idle")

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Filename:"))
        input_layout.addWidget(self.filename_input)
        input_layout.addWidget(QLabel("Duration (s):"))
        input_layout.addWidget(self.duration_input)
        input_layout.addWidget(QLabel("Channels:"))
        input_layout.addWidget(self.channel_count_input)
        input_layout.addWidget(self.start_button)
        self.layout.addLayout(input_layout)
        self.layout.addLayout(self.channel_form)
        self.layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(0, 4095)
        self.plot_widget.addLegend()
        self.curves = []
        self.layout.addWidget(self.plot_widget)

        self.plot_buffer = []
        self.time_buffer = []

        self.reader = None

    def update_channel_inputs(self):
        count = self.channel_count_input.value()
        for i, inp in enumerate(self.channel_name_inputs):
            inp.setVisible(i < count)
            self.channel_form.labelForField(inp).setVisible(i < count)

    def start_recording(self):
        filename = self.filename_input.text()
        duration = self.duration_input.value()
        num_channels = self.channel_count_input.value()

        self.data.clear()
        self.time_buffer.clear()

        self.plot_widget.clear()
        self.plot_buffer = [[] for _ in range(num_channels)]
        self.curves = []
        names = [self.channel_name_inputs[i].text() for i in range(num_channels)]

        for i in range(num_channels):
            curve = self.plot_widget.plot(pen=pg.mkPen(i), name=names[i])
            self.curves.append(curve)

        self.reader = SerialReader("COM3", 921600, duration, num_channels)
        self.reader.data_received.connect(self.handle_data)
        self.reader.finished.connect(self.recording_finished)
        self.reader.start()

        self.status_label.setText("Recording...")
        self.start_button.setEnabled(False)

    def handle_data(self, data_point):
        self.data.append(data_point)
        ts = data_point[0]
        chans = data_point[1:]

        if not self.time_buffer:
            self.t0 = ts
        t_s = (ts - self.t0) / 1e6
        self.time_buffer.append(t_s)

        for i, val in enumerate(chans):
            self.plot_buffer[i].append(val)
            self.curves[i].setData(self.time_buffer, self.plot_buffer[i])

    def recording_finished(self):
        self.status_label.setText("Done")
        self.start_button.setEnabled(True)

        if self.data:
            num_channels = self.channel_count_input.value()
            names = [self.channel_name_inputs[i].text() for i in range(num_channels)]
            cols = ['timestamp_us'] + names
            df = pd.DataFrame(self.data, columns=cols)
            df['time_s'] = (df['timestamp_us'] - df['timestamp_us'][0]) / 1e6
            df.to_csv(self.filename_input.text(), index=False)
            self.status_label.setText(f"Saved to {self.filename_input.text()}")
        else:
            self.status_label.setText("No data collected!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
