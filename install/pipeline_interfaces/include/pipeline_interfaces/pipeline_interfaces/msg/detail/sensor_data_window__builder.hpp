// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/sensor_data_window.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__BUILDER_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "pipeline_interfaces/msg/detail/sensor_data_window__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace pipeline_interfaces
{

namespace msg
{

namespace builder
{

class Init_SensorDataWindow_readings
{
public:
  Init_SensorDataWindow_readings()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::pipeline_interfaces::msg::SensorDataWindow readings(::pipeline_interfaces::msg::SensorDataWindow::_readings_type arg)
  {
    msg_.readings = std::move(arg);
    return std::move(msg_);
  }

private:
  ::pipeline_interfaces::msg::SensorDataWindow msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::pipeline_interfaces::msg::SensorDataWindow>()
{
  return pipeline_interfaces::msg::builder::Init_SensorDataWindow_readings();
}

}  // namespace pipeline_interfaces

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__BUILDER_HPP_
