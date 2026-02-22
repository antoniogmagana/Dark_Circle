// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/raw_sensor_reading.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__pipeline_interfaces__msg__RawSensorReading __attribute__((deprecated))
#else
# define DEPRECATED__pipeline_interfaces__msg__RawSensorReading __declspec(deprecated)
#endif

namespace pipeline_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RawSensorReading_
{
  using Type = RawSensorReading_<ContainerAllocator>;

  explicit RawSensorReading_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : timestamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sensor_label = "";
      this->sensor_values = 0.0;
      this->latitude = 0.0;
      this->longitude = 0.0;
    }
  }

  explicit RawSensorReading_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : sensor_label(_alloc),
    timestamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sensor_label = "";
      this->sensor_values = 0.0;
      this->latitude = 0.0;
      this->longitude = 0.0;
    }
  }

  // field types and members
  using _sensor_label_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _sensor_label_type sensor_label;
  using _timestamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _timestamp_type timestamp;
  using _sensor_values_type =
    double;
  _sensor_values_type sensor_values;
  using _latitude_type =
    double;
  _latitude_type latitude;
  using _longitude_type =
    double;
  _longitude_type longitude;

  // setters for named parameter idiom
  Type & set__sensor_label(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->sensor_label = _arg;
    return *this;
  }
  Type & set__timestamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__sensor_values(
    const double & _arg)
  {
    this->sensor_values = _arg;
    return *this;
  }
  Type & set__latitude(
    const double & _arg)
  {
    this->latitude = _arg;
    return *this;
  }
  Type & set__longitude(
    const double & _arg)
  {
    this->longitude = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> *;
  using ConstRawPtr =
    const pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__pipeline_interfaces__msg__RawSensorReading
    std::shared_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__pipeline_interfaces__msg__RawSensorReading
    std::shared_ptr<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RawSensorReading_ & other) const
  {
    if (this->sensor_label != other.sensor_label) {
      return false;
    }
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->sensor_values != other.sensor_values) {
      return false;
    }
    if (this->latitude != other.latitude) {
      return false;
    }
    if (this->longitude != other.longitude) {
      return false;
    }
    return true;
  }
  bool operator!=(const RawSensorReading_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RawSensorReading_

// alias to use template instance with default allocator
using RawSensorReading =
  pipeline_interfaces::msg::RawSensorReading_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace pipeline_interfaces

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__STRUCT_HPP_
