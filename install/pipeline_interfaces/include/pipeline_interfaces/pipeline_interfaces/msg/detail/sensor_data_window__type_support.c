// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "pipeline_interfaces/msg/detail/sensor_data_window__rosidl_typesupport_introspection_c.h"
#include "pipeline_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "pipeline_interfaces/msg/detail/sensor_data_window__functions.h"
#include "pipeline_interfaces/msg/detail/sensor_data_window__struct.h"


// Include directives for member types
// Member `readings`
#include "pipeline_interfaces/msg/raw_sensor_reading.h"
// Member `readings`
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  pipeline_interfaces__msg__SensorDataWindow__init(message_memory);
}

void pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_fini_function(void * message_memory)
{
  pipeline_interfaces__msg__SensorDataWindow__fini(message_memory);
}

size_t pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__size_function__SensorDataWindow__readings(
  const void * untyped_member)
{
  const pipeline_interfaces__msg__RawSensorReading__Sequence * member =
    (const pipeline_interfaces__msg__RawSensorReading__Sequence *)(untyped_member);
  return member->size;
}

const void * pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_const_function__SensorDataWindow__readings(
  const void * untyped_member, size_t index)
{
  const pipeline_interfaces__msg__RawSensorReading__Sequence * member =
    (const pipeline_interfaces__msg__RawSensorReading__Sequence *)(untyped_member);
  return &member->data[index];
}

void * pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_function__SensorDataWindow__readings(
  void * untyped_member, size_t index)
{
  pipeline_interfaces__msg__RawSensorReading__Sequence * member =
    (pipeline_interfaces__msg__RawSensorReading__Sequence *)(untyped_member);
  return &member->data[index];
}

void pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__fetch_function__SensorDataWindow__readings(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const pipeline_interfaces__msg__RawSensorReading * item =
    ((const pipeline_interfaces__msg__RawSensorReading *)
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_const_function__SensorDataWindow__readings(untyped_member, index));
  pipeline_interfaces__msg__RawSensorReading * value =
    (pipeline_interfaces__msg__RawSensorReading *)(untyped_value);
  *value = *item;
}

void pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__assign_function__SensorDataWindow__readings(
  void * untyped_member, size_t index, const void * untyped_value)
{
  pipeline_interfaces__msg__RawSensorReading * item =
    ((pipeline_interfaces__msg__RawSensorReading *)
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_function__SensorDataWindow__readings(untyped_member, index));
  const pipeline_interfaces__msg__RawSensorReading * value =
    (const pipeline_interfaces__msg__RawSensorReading *)(untyped_value);
  *item = *value;
}

bool pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__resize_function__SensorDataWindow__readings(
  void * untyped_member, size_t size)
{
  pipeline_interfaces__msg__RawSensorReading__Sequence * member =
    (pipeline_interfaces__msg__RawSensorReading__Sequence *)(untyped_member);
  pipeline_interfaces__msg__RawSensorReading__Sequence__fini(member);
  return pipeline_interfaces__msg__RawSensorReading__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_member_array[1] = {
  {
    "readings",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces__msg__SensorDataWindow, readings),  // bytes offset in struct
    NULL,  // default value
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__size_function__SensorDataWindow__readings,  // size() function pointer
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_const_function__SensorDataWindow__readings,  // get_const(index) function pointer
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__get_function__SensorDataWindow__readings,  // get(index) function pointer
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__fetch_function__SensorDataWindow__readings,  // fetch(index, &value) function pointer
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__assign_function__SensorDataWindow__readings,  // assign(index, value) function pointer
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__resize_function__SensorDataWindow__readings  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_members = {
  "pipeline_interfaces__msg",  // message namespace
  "SensorDataWindow",  // message name
  1,  // number of fields
  sizeof(pipeline_interfaces__msg__SensorDataWindow),
  false,  // has_any_key_member_
  pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_member_array,  // message members
  pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_init_function,  // function to initialize message memory (memory has to be allocated)
  pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_type_support_handle = {
  0,
  &pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_members,
  get_message_typesupport_handle_function,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_hash,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_description,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_pipeline_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, pipeline_interfaces, msg, SensorDataWindow)() {
  pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, pipeline_interfaces, msg, RawSensorReading)();
  if (!pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_type_support_handle.typesupport_identifier) {
    pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &pipeline_interfaces__msg__SensorDataWindow__rosidl_typesupport_introspection_c__SensorDataWindow_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
