import numpy as np

from src.ml.calibration_buffer import CalibrationBuffer


class TestDeterministicCalibration:
    def test_same_seed_same_data(self):
        buf1 = CalibrationBuffer(machine_index=0, sensor_offset=0, global_seed=42)
        data1 = buf1.generate_synthetic_data(mu=100.0, sigma=5.0, samples=1000)
        buf2 = CalibrationBuffer(machine_index=0, sensor_offset=0, global_seed=42)
        data2 = buf2.generate_synthetic_data(mu=100.0, sigma=5.0, samples=1000)
        assert np.allclose(data1, data2)

    def test_different_seed_different_data(self):
        buf1 = CalibrationBuffer(machine_index=0, sensor_offset=0, global_seed=42)
        data1 = buf1.generate_synthetic_data(mu=100.0, sigma=5.0, samples=1000)
        buf2 = CalibrationBuffer(machine_index=0, sensor_offset=0, global_seed=43)
        data2 = buf2.generate_synthetic_data(mu=100.0, sigma=5.0, samples=1000)
        assert not np.allclose(data1, data2)
