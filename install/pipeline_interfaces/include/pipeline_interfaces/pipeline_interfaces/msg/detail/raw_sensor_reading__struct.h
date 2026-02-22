// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/raw_sensor_reading.h"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_H_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'sensor_label'
#include "rosidl_runtime_c/string.h"
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/RawSensorReading in the package pipeline_interfaces.
/**
  * Add the sensor id to make it scalable
 */
typedef struct pipeline_interfaces__msg__RawSensorReading
{
  rosidl_runtime_c__String sensor_label;
  /// timestamp for sensor reading in seconds
  builtin_interfaces__msg__Time timestamp;
  /// sensor reading value
  double sensor_values;
  /// sensor location values
  double latitude;
  double longitude;
} pipeline_interfaces__msg__RawSensorReading;

// Struct for a sequence of pipeline_interfaces__msg__RawSensorReading.
typedef struct pipeline_interfaces__msg__RawSensorReading__Sequence
{
  pipeline_interfaces__msg__RawSensorReading * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} pipeline_interfaces__msg__RawSensorReading__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_H_
