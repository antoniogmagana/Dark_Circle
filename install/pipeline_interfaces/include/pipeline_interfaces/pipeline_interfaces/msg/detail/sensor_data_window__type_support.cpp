// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "pipeline_interfaces/msg/detail/sensor_data_window__functions.h"
#include "pipeline_interfaces/msg/detail/sensor_data_window__struct.hpp"
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

void SensorDataWindow_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) pipeline_interfaces::msg::SensorDataWindow(_init);
}

void SensorDataWindow_fini_function(void * message_memory)
{
  auto typed_message = static_cast<pipeline_interfaces::msg::SensorDataWindow *>(message_memory);
  typed_message->~SensorDataWindow();
}

size_t size_function__SensorDataWindow__readings(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<pipeline_interfaces::msg::RawSensorReading> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SensorDataWindow__readings(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<pipeline_interfaces::msg::RawSensorReading> *>(untyped_member);
  return &member[index];
}

void * get_function__SensorDataWindow__readings(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<pipeline_interfaces::msg::RawSensorReading> *>(untyped_member);
  return &member[index];
}

void fetch_function__SensorDataWindow__readings(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const pipeline_interfaces::msg::RawSensorReading *>(
    get_const_function__SensorDataWindow__readings(untyped_member, index));
  auto & value = *reinterpret_cast<pipeline_interfaces::msg::RawSensorReading *>(untyped_value);
  value = item;
}

void assign_function__SensorDataWindow__readings(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<pipeline_interfaces::msg::RawSensorReading *>(
    get_function__SensorDataWindow__readings(untyped_member, index));
  const auto & value = *reinterpret_cast<const pipeline_interfaces::msg::RawSensorReading *>(untyped_value);
  item = value;
}

void resize_function__SensorDataWindow__readings(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<pipeline_interfaces::msg::RawSensorReading> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SensorDataWindow_message_member_array[1] = {
  {
    "readings",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<pipeline_interfaces::msg::RawSensorReading>(),  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(pipeline_interfaces::msg::SensorDataWindow, readings),  // bytes offset in struct
    nullptr,  // default value
    size_function__SensorDataWindow__readings,  // size() function pointer
    get_const_function__SensorDataWindow__readings,  // get_const(index) function pointer
    get_function__SensorDataWindow__readings,  // get(index) function pointer
    fetch_function__SensorDataWindow__readings,  // fetch(index, &value) function pointer
    assign_function__SensorDataWindow__readings,  // assign(index, value) function pointer
    resize_function__SensorDataWindow__readings  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SensorDataWindow_message_members = {
  "pipeline_interfaces::msg",  // message namespace
  "SensorDataWindow",  // message name
  1,  // number of fields
  sizeof(pipeline_interfaces::msg::SensorDataWindow),
  false,  // has_any_key_member_
  SensorDataWindow_message_member_array,  // message members
  SensorDataWindow_init_function,  // function to initialize message memory (memory has to be allocated)
  SensorDataWindow_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SensorDataWindow_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SensorDataWindow_message_members,
  get_message_typesupport_handle_function,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_hash,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_description,
  &pipeline_interfaces__msg__SensorDataWindow__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace pipeline_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<pipeline_interfaces::msg::SensorDataWindow>()
{
  return &::pipeline_interfaces::msg::rosidl_typesupport_introspection_cpp::SensorDataWindow_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, pipeline_interfaces, msg, SensorDataWindow)() {
  return &::pipeline_interfaces::msg::rosidl_typesupport_introspection_cpp::SensorDataWindow_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
