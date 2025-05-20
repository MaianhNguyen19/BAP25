import sys
import re
import time
import serial
import numpy as np
from collections import deque
from scipy.signal import firwin, lfilter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg


class SerialReader(QThread):
    data_received = pyqtSignal(list)

    def __init__(self, port, baudrate, num_channels):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.num_channels = num_channels
        self.running = False

    def run(self):
        ser = serial.Serial(self.port, self.baudrate, timeout=1)
        ser.reset_input_buffer()
        self.running = True

        try:
            while self.running:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line or not re.match(r'^\d+,', line):
                    continue
                parts = line.split(',')
                if len(parts) != 1 + self.num_channels:
                    continue
                try:
                    int(parts[0])  # timestamp, not used here
                    chans = list(map(int, parts[1:]))
                    self.data_received.emit(chans)
                except ValueError:
                    continue
        finally:
            ser.close()

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Channel 0 Only")
        self.resize(800, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.total_channels = 6  # Still expect 6 channels over serial
        self.window_samples = 300
        self.sampling_rate = 2000  # Hz
        self.fir_taps = 101
        self.fir_cutoff = 120
        self.fir_coeff = firwin(self.fir_taps, self.fir_cutoff, fs=self.sampling_rate)

        self.plot_widget = pg.PlotWidget(title="Channel 0 - Time Domain")
        self.plot_widget.setYRange(0, 4095)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self.curve_time = self.plot_widget.plot(pen='y', name="ch0")
        self.plot_buffer = deque(maxlen=self.window_samples)
        self.filter_buffer = deque(maxlen=self.fir_taps)

        self.reader = SerialReader("COM3", 921600, self.total_channels)
        self.reader.data_received.connect(self.update_plot)
        self.reader.start()

    def update_plot(self, chans):
        val = chans[0]  # Only use channel 0
        self.filter_buffer.append(val)
        if len(self.filter_buffer) == self.fir_taps:
            val = lfilter(self.fir_coeff, 1.0, list(self.filter_buffer))[-1]
        self.plot_buffer.append(val)

        y = list(self.plot_buffer)
        x = list(range(len(y)))
        self.curve_time.setData(x, y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
