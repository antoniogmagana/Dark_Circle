// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/raw_sensor_reading.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__TRAITS_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace pipeline_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const RawSensorReading & msg,
  std::ostream & out)
{
  out << "{";
  // member: sensor_label
  {
    out << "sensor_label: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_label, out);
    out << ", ";
  }

  // member: timestamp
  {
    out << "timestamp: ";
    to_flow_style_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: sensor_values
  {
    out << "sensor_values: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_values, out);
    out << ", ";
  }

  // member: latitude
  {
    out << "latitude: ";
    rosidl_generator_traits::value_to_yaml(msg.latitude, out);
    out << ", ";
  }

  // member: longitude
  {
    out << "longitude: ";
    rosidl_generator_traits::value_to_yaml(msg.longitude, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RawSensorReading & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: sensor_label
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sensor_label: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_label, out);
    out << "\n";
  }

  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp:\n";
    to_block_style_yaml(msg.timestamp, out, indentation + 2);
  }

  // member: sensor_values
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sensor_values: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor_values, out);
    out << "\n";
  }

  // member: latitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "latitude: ";
    rosidl_generator_traits::value_to_yaml(msg.latitude, out);
    out << "\n";
  }

  // member: longitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "longitude: ";
    rosidl_generator_traits::value_to_yaml(msg.longitude, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RawSensorReading & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace pipeline_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use pipeline_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const pipeline_interfaces::msg::RawSensorReading & msg,
  std::ostream & out, size_t indentation = 0)
{
  pipeline_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use pipeline_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const pipeline_interfaces::msg::RawSensorReading & msg)
{
  return pipeline_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<pipeline_interfaces::msg::RawSensorReading>()
{
  return "pipeline_interfaces::msg::RawSensorReading";
}

template<>
inline const char * name<pipeline_interfaces::msg::RawSensorReading>()
{
  return "pipeline_interfaces/msg/RawSensorReading";
}

template<>
struct has_fixed_size<pipeline_interfaces::msg::RawSensorReading>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<pipeline_interfaces::msg::RawSensorReading>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<pipeline_interfaces::msg::RawSensorReading>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__TRAITS_HPP_
