# sensor id from array producing readings
string sensor_id

# time stamp of detection window
float64 timestamp

# vehicle detection status
bool vehicle_detected

# detection confidence score
float32 detection_confidence

# vehicle type determination
string vehicle_class

# vehicle type confidence score
float32 classification_confidence

# wall-clock moment egress published this result (Unix epoch seconds)
float64 publish_time

# publish_time - original capture timestamp; includes 1.0s window-fill floor
float64 latency_seconds