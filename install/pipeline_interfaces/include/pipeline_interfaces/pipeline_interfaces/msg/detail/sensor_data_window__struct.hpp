// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/sensor_data_window.hpp"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_HPP_
#define PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'readings'
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__pipeline_interfaces__msg__SensorDataWindow __attribute__((deprecated))
#else
# define DEPRECATED__pipeline_interfaces__msg__SensorDataWindow __declspec(deprecated)
#endif

namespace pipeline_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SensorDataWindow_
{
  using Type = SensorDataWindow_<ContainerAllocator>;

  explicit SensorDataWindow_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit SensorDataWindow_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _readings_type =
    std::vector<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>>;
  _readings_type readings;

  // setters for named parameter idiom
  Type & set__readings(
    const std::vector<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<pipeline_interfaces::msg::RawSensorReading_<ContainerAllocator>>> & _arg)
  {
    this->readings = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> *;
  using ConstRawPtr =
    const pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__pipeline_interfaces__msg__SensorDataWindow
    std::shared_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__pipeline_interfaces__msg__SensorDataWindow
    std::shared_ptr<pipeline_interfaces::msg::SensorDataWindow_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SensorDataWindow_ & other) const
  {
    if (this->readings != other.readings) {
      return false;
    }
    return true;
  }
  bool operator!=(const SensorDataWindow_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SensorDataWindow_

// alias to use template instance with default allocator
using SensorDataWindow =
  pipeline_interfaces::msg::SensorDataWindow_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace pipeline_interfaces

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__STRUCT_HPP_
