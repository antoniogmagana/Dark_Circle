import numpy as np
from inference_protos import inference_pb2


class SensorBuffer():
    def __init__(self, sensor_id):
        self.active_channels = set()
        self.sensor_id = sensor_id
        self.start_time = None
        self.window = 1.0

        self.rates = {
            'acoustic': 16000,
            'seismic': 100,
            'accel_x': 100,
            'accel_y': 100,
            'accel_z': 100
            }
        self.limits = {
            'acoustic': int(self.window * self.rates['acoustic']),
            'seismic': int(self.window * self.rates['seismic']),
            'accel_x': int(self.window * self.rates['accel_x']),
            'accel_y': int(self.window * self.rates['accel_y']),
            'accel_z': int(self.window * self.rates['accel_z'])
            }
        self.buffers = {
            'acoustic': np.zeros(self.limits['acoustic']),
            'seismic': np.zeros(self.limits['seismic']),
            'accel_x': np.zeros(self.limits['accel_x']),
            'accel_y': np.zeros(self.limits['accel_y']),
            'accel_z': np.zeros(self.limits['accel_z'])
            }
        
        self.holding_pen = {
            'acoustic': [],
            'seismic': [],
            'accel_x': [],
            'accel_y': [],
            'accel_z': []
        }
        self.adc_scale = {
            'acoustic': 2**15,
            'seismic': 2**23,
            'accel': 2**23,
        }

    
    def _package_window(self):
        ts = inference_pb2.google_dot_protobuf_dot_timestamp__pb2.Timestamp()
        ts.seconds = int(self.start_time)
        ts.nanos = int((self.start_time - int(self.start_time)) * 1e9)

        payload = inference_pb2.SensorData(
            sensor_id=self.sensor_id,
            time_stamp = ts
        )

        if 'acoustic' in self.active_channels:
            arr = self.buffers['acoustic'] / self.adc_scale['acoustic']
            arr = arr - arr.mean()
            arr = arr / (arr.std() + 1e-8)
            arr = np.clip(arr, -10.0, 10.0)
            payload.channels.append('acoustic')
            payload.acoustic_data.CopyFrom(
                inference_pb2.Tensor(
                    shape=list(arr.shape),
                    data=arr.flatten().tolist()
                )
            )

        if 'seismic' in self.active_channels:
            arr = self.buffers['seismic'] / self.adc_scale['seismic']
            arr = arr - arr.mean()
            arr = arr / (arr.std() + 1e-8)
            arr = np.clip(arr, -10.0, 10.0)
            payload.channels.append('seismic')
            payload.seismic_data.CopyFrom(
                inference_pb2.Tensor(
                    shape=list(arr.shape),
                    data=arr.flatten().tolist()
                )
            )

        if any(axis in self.active_channels for axis in ['accel_x', 'accel_y', 'accel_z']):
            payload.channels.append('accel')
            accel_matrix = np.vstack((
                self.buffers['accel_x'],
                self.buffers['accel_y'],
                self.buffers['accel_z']
            )) / self.adc_scale['accel']
            accel_matrix = accel_matrix - accel_matrix.mean(axis=1, keepdims=True)
            accel_matrix = accel_matrix / (accel_matrix.std(axis=1, keepdims=True) + 1e-8)
            accel_matrix = np.clip(accel_matrix, -10.0, 10.0)
            payload.accel_data.CopyFrom(
                inference_pb2.Tensor(
                    shape=list(accel_matrix.shape),
                    data=accel_matrix.flatten().tolist()
                )
            )
        return payload

    def _reset_buffers(self):
        # 1. Wipe the slate clean for the new second
        self.active_channels.clear()
        
        for ch in self.buffers:
            self.buffers[ch].fill(0)
            pen_len = len(self.holding_pen[ch])
            
            # 2. Check if this channel has leftovers in the pen
            if pen_len > 0:
                self.buffers[ch][0:pen_len] = self.holding_pen[ch]
                self.holding_pen[ch].clear()
                
                # 3. Re-register this channel so it gets packaged!
                self.active_channels.add(ch)
                
        # 4. Advance the anchor
        self.start_time += self.window
        return None

    
    def load_buffer(self, channel, data, timestamp):
        
        rate = self.rates[channel]
        limit = self.limits[channel]
        buffer = self.buffers[channel]
        
        if self.start_time is None:
            self.start_time = timestamp
        
        current_time = timestamp
        time_diff = current_time - self.start_time
        
        if time_diff < 0:
            return None
        
        if time_diff >= self.window:
            if channel == 'acoustic':
                ready_payload = self._package_window()
                self._reset_buffers()
                
                self.load_buffer(channel, data, timestamp)
                
                return ready_payload
            else:
                self.holding_pen[channel].extend(data)
                return None
        
        self.active_channels.add(channel)
        start_index = int(time_diff * rate)
        data_length = len(data)
        end_index = start_index + data_length
        
        if end_index < limit:
            buffer[start_index:end_index] = data
            return None
        else:
            space_left = limit - start_index
            chunk_1 = data[0:space_left]
            chunk_2 = data[space_left:]
            buffer[start_index:limit] = chunk_1
            
            if channel == 'acoustic':
                ready_payload = self._package_window()
                self.holding_pen[channel].extend(chunk_2)
                self._reset_buffers()
                return ready_payload
            else:
                self.holding_pen[channel].extend(chunk_2)
                return None