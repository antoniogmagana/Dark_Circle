// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/raw_sensor_reading.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__BUILDER_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace pipeline_interfaces
{

namespace msg
{

namespace builder
{

class Init_RawSensorReading_longitude
{
public:
  explicit Init_RawSensorReading_longitude(::pipeline_interfaces::msg::RawSensorReading & msg)
  : msg_(msg)
  {}
  ::pipeline_interfaces::msg::RawSensorReading longitude(::pipeline_interfaces::msg::RawSensorReading::_longitude_type arg)
  {
    msg_.longitude = std::move(arg);
    return std::move(msg_);
  }

private:
  ::pipeline_interfaces::msg::RawSensorReading msg_;
};

class Init_RawSensorReading_latitude
{
public:
  explicit Init_RawSensorReading_latitude(::pipeline_interfaces::msg::RawSensorReading & msg)
  : msg_(msg)
  {}
  Init_RawSensorReading_longitude latitude(::pipeline_interfaces::msg::RawSensorReading::_latitude_type arg)
  {
    msg_.latitude = std::move(arg);
    return Init_RawSensorReading_longitude(msg_);
  }

private:
  ::pipeline_interfaces::msg::RawSensorReading msg_;
};

class Init_RawSensorReading_sensor_values
{
public:
  explicit Init_RawSensorReading_sensor_values(::pipeline_interfaces::msg::RawSensorReading & msg)
  : msg_(msg)
  {}
  Init_RawSensorReading_latitude sensor_values(::pipeline_interfaces::msg::RawSensorReading::_sensor_values_type arg)
  {
    msg_.sensor_values = std::move(arg);
    return Init_RawSensorReading_latitude(msg_);
  }

private:
  ::pipeline_interfaces::msg::RawSensorReading msg_;
};

class Init_RawSensorReading_timestamp
{
public:
  explicit Init_RawSensorReading_timestamp(::pipeline_interfaces::msg::RawSensorReading & msg)
  : msg_(msg)
  {}
  Init_RawSensorReading_sensor_values timestamp(::pipeline_interfaces::msg::RawSensorReading::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_RawSensorReading_sensor_values(msg_);
  }

private:
  ::pipeline_interfaces::msg::RawSensorReading msg_;
};

class Init_RawSensorReading_sensor_label
{
public:
  Init_RawSensorReading_sensor_label()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RawSensorReading_timestamp sensor_label(::pipeline_interfaces::msg::RawSensorReading::_sensor_label_type arg)
  {
    msg_.sensor_label = std::move(arg);
    return Init_RawSensorReading_timestamp(msg_);
  }

private:
  ::pipeline_interfaces::msg::RawSensorReading msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::pipeline_interfaces::msg::RawSensorReading>()
{
  return pipeline_interfaces::msg::builder::Init_RawSensorReading_sensor_label();
}

}  // namespace pipeline_interfaces

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__BUILDER_HPP_
