// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/sensor_data_window.h"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_H_
#define PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'readings'
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.h"

/// Struct defined in msg/SensorDataWindow in the package pipeline_interfaces.
/**
  * an array of sensor data readings collected by the ingestor node
 */
typedef struct pipeline_interfaces__msg__SensorDataWindow
{
  pipeline_interfaces__msg__RawSensorReading__Sequence readings;
} pipeline_interfaces__msg__SensorDataWindow;

// Struct for a sequence of pipeline_interfaces__msg__SensorDataWindow.
typedef struct pipeline_interfaces__msg__SensorDataWindow__Sequence
{
  pipeline_interfaces__msg__SensorDataWindow * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} pipeline_interfaces__msg__SensorDataWindow__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_H_
