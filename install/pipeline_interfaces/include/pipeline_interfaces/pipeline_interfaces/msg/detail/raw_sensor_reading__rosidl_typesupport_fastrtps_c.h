// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice
#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "pipeline_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
bool cdr_serialize_pipeline_interfaces__msg__RawSensorReading(
  const pipeline_interfaces__msg__RawSensorReading * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
bool cdr_deserialize_pipeline_interfaces__msg__RawSensorReading(
  eprosima::fastcdr::Cdr &,
  pipeline_interfaces__msg__RawSensorReading * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
size_t get_serialized_size_pipeline_interfaces__msg__RawSensorReading(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
size_t max_serialized_size_pipeline_interfaces__msg__RawSensorReading(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
bool cdr_serialize_key_pipeline_interfaces__msg__RawSensorReading(
  const pipeline_interfaces__msg__RawSensorReading * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
size_t get_serialized_size_key_pipeline_interfaces__msg__RawSensorReading(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
size_t max_serialized_size_key_pipeline_interfaces__msg__RawSensorReading(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_pipeline_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, pipeline_interfaces, msg, RawSensorReading)();

#ifdef __cplusplus
}
#endif

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
