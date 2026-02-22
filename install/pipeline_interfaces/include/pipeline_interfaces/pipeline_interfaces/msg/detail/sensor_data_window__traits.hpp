// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/sensor_data_window.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__TRAITS_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "pipeline_interfaces/msg/detail/sensor_data_window__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'readings'
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__traits.hpp"

namespace pipeline_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const SensorDataWindow & msg,
  std::ostream & out)
{
  out << "{";
  // member: readings
  {
    if (msg.readings.size() == 0) {
      out << "readings: []";
    } else {
      out << "readings: [";
      size_t pending_items = msg.readings.size();
      for (auto item : msg.readings) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SensorDataWindow & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: readings
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.readings.size() == 0) {
      out << "readings: []\n";
    } else {
      out << "readings:\n";
      for (auto item : msg.readings) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SensorDataWindow & msg, bool use_flow_style = false)
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
  const pipeline_interfaces::msg::SensorDataWindow & msg,
  std::ostream & out, size_t indentation = 0)
{
  pipeline_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use pipeline_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const pipeline_interfaces::msg::SensorDataWindow & msg)
{
  return pipeline_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<pipeline_interfaces::msg::SensorDataWindow>()
{
  return "pipeline_interfaces::msg::SensorDataWindow";
}

template<>
inline const char * name<pipeline_interfaces::msg::SensorDataWindow>()
{
  return "pipeline_interfaces/msg/SensorDataWindow";
}

template<>
struct has_fixed_size<pipeline_interfaces::msg::SensorDataWindow>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<pipeline_interfaces::msg::SensorDataWindow>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<pipeline_interfaces::msg::SensorDataWindow>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__TRAITS_HPP_
