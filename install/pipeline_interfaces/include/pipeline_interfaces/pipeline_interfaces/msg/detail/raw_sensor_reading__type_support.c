// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__rosidl_typesupport_introspection_c.h"
#include "pipeline_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__functions.h"
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.h"


// Include directives for member types
// Member `sensor_label`
#include "rosidl_runtime_c/string_functions.h"
// Member `timestamp`
#include "builtin_interfaces/msg/time.h"
// Member `timestamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  pipeline_interfaces__msg__RawSensorReading__init(message_memory);
}

void pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_fini_function(void * message_memory)
{
  pipeline_interfaces__msg__RawSensorReading__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_member_array[5] = {
  {
    "sensor_label",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__RawSensorReading, sensor_label),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "timestamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__RawSensorReading, timestamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sensor_values",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__RawSensorReading, sensor_values),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "latitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__RawSensorReading, latitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "longitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__RawSensorReading, longitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_members = {
  "pipeline_interfaces__msg",  // message namespace
  "RawSensorReading",  // message name
  5,  // number of fields
  sizeof(pipeline_interfaces__msg__RawSensorReading),
  false,  // has_any_key_member_
  pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_member_array,  // message members
  pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_init_function,  // function to initialize message memory (memory has to be allocated)
  pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_type_support_handle = {
  0,
  &pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_members,
  get_message_typesupport_handle_function,
  &pipeline_interfaces__msg__RawSensorReading__get_type_hash,
  &pipeline_interfaces__msg__RawSensorReading__get_type_description,
  &pipeline_interfaces__msg__RawSensorReading__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_pipeline_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, pipeline_interfaces, msg, RawSensorReading)() {
  pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_type_support_handle.typesupport_identifier) {
    pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &pipeline_interfaces__msg__RawSensorReading__rosidl_typesupport_introspection_c__RawSensorReading_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
