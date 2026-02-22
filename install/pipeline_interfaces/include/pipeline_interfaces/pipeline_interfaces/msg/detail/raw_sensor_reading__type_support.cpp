// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__functions.h"
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace pipeline_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void RawSensorReading_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) pipeline_interfaces::msg::RawSensorReading(_init);
}

void RawSensorReading_fini_function(void * message_memory)
{
  auto typed_message = static_cast<pipeline_interfaces::msg::RawSensorReading *>(message_memory);
  typed_message->~RawSensorReading();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RawSensorReading_message_member_array[5] = {
  {
    "sensor_label",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::RawSensorReading, sensor_label),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "timestamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::RawSensorReading, timestamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "sensor_values",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::RawSensorReading, sensor_values),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "latitude",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::RawSensorReading, latitude),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "longitude",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::RawSensorReading, longitude),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RawSensorReading_message_members = {
  "pipeline_interfaces::msg",  // message namespace
  "RawSensorReading",  // message name
  5,  // number of fields
  sizeof(pipeline_interfaces::msg::RawSensorReading),
  false,  // has_any_key_member_
  RawSensorReading_message_member_array,  // message members
  RawSensorReading_init_function,  // function to initialize message memory (memory has to be allocated)
  RawSensorReading_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RawSensorReading_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RawSensorReading_message_members,
  get_message_typesupport_handle_function,
  &pipeline_interfaces__msg__RawSensorReading__get_type_hash,
  &pipeline_interfaces__msg__RawSensorReading__get_type_description,
  &pipeline_interfaces__msg__RawSensorReading__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace pipeline_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<pipeline_interfaces::msg::RawSensorReading>()
{
  return &::pipeline_interfaces::msg::rosidl_typesupport_introspection_cpp::RawSensorReading_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, pipeline_interfaces, msg, RawSensorReading)() {
  return &::pipeline_interfaces::msg::rosidl_typesupport_introspection_cpp::RawSensorReading_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
