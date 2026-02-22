// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `sensor_label`
#include "rosidl_runtime_c/string_functions.h"
// Member `timestamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
pipeline_interfaces__msg__RawSensorReading__init(pipeline_interfaces__msg__RawSensorReading * msg)
{
  if (!msg) {
    return false;
  }
  // sensor_label
  if (!rosidl_runtime_c__String__init(&msg->sensor_label)) {
    pipeline_interfaces__msg__RawSensorReading__fini(msg);
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__init(&msg->timestamp)) {
    pipeline_interfaces__msg__RawSensorReading__fini(msg);
    return false;
  }
  // sensor_values
  // latitude
  // longitude
  return true;
}

void
pipeline_interfaces__msg__RawSensorReading__fini(pipeline_interfaces__msg__RawSensorReading * msg)
{
  if (!msg) {
    return;
  }
  // sensor_label
  rosidl_runtime_c__String__fini(&msg->sensor_label);
  // timestamp
  builtin_interfaces__msg__Time__fini(&msg->timestamp);
  // sensor_values
  // latitude
  // longitude
}

bool
pipeline_interfaces__msg__RawSensorReading__are_equal(const pipeline_interfaces__msg__RawSensorReading * lhs, const pipeline_interfaces__msg__RawSensorReading * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // sensor_label
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->sensor_label), &(rhs->sensor_label)))
  {
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->timestamp), &(rhs->timestamp)))
  {
    return false;
  }
  // sensor_values
  if (lhs->sensor_values != rhs->sensor_values) {
    return false;
  }
  // latitude
  if (lhs->latitude != rhs->latitude) {
    return false;
  }
  // longitude
  if (lhs->longitude != rhs->longitude) {
    return false;
  }
  return true;
}

bool
pipeline_interfaces__msg__RawSensorReading__copy(
  const pipeline_interfaces__msg__RawSensorReading * input,
  pipeline_interfaces__msg__RawSensorReading * output)
{
  if (!input || !output) {
    return false;
  }
  // sensor_label
  if (!rosidl_runtime_c__String__copy(
      &(input->sensor_label), &(output->sensor_label)))
  {
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->timestamp), &(output->timestamp)))
  {
    return false;
  }
  // sensor_values
  output->sensor_values = input->sensor_values;
  // latitude
  output->latitude = input->latitude;
  // longitude
  output->longitude = input->longitude;
  return true;
}

pipeline_interfaces__msg__RawSensorReading *
pipeline_interfaces__msg__RawSensorReading__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__RawSensorReading * msg = (pipeline_interfaces__msg__RawSensorReading *)allocator.allocate(sizeof(pipeline_interfaces__msg__RawSensorReading), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(pipeline_interfaces__msg__RawSensorReading));
  bool success = pipeline_interfaces__msg__RawSensorReading__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
pipeline_interfaces__msg__RawSensorReading__destroy(pipeline_interfaces__msg__RawSensorReading * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    pipeline_interfaces__msg__RawSensorReading__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
pipeline_interfaces__msg__RawSensorReading__Sequence__init(pipeline_interfaces__msg__RawSensorReading__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__RawSensorReading * data = NULL;

  if (size) {
    data = (pipeline_interfaces__msg__RawSensorReading *)allocator.zero_allocate(size, sizeof(pipeline_interfaces__msg__RawSensorReading), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = pipeline_interfaces__msg__RawSensorReading__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        pipeline_interfaces__msg__RawSensorReading__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
pipeline_interfaces__msg__RawSensorReading__Sequence__fini(pipeline_interfaces__msg__RawSensorReading__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      pipeline_interfaces__msg__RawSensorReading__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

pipeline_interfaces__msg__RawSensorReading__Sequence *
pipeline_interfaces__msg__RawSensorReading__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__RawSensorReading__Sequence * array = (pipeline_interfaces__msg__RawSensorReading__Sequence *)allocator.allocate(sizeof(pipeline_interfaces__msg__RawSensorReading__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = pipeline_interfaces__msg__RawSensorReading__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
pipeline_interfaces__msg__RawSensorReading__Sequence__destroy(pipeline_interfaces__msg__RawSensorReading__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    pipeline_interfaces__msg__RawSensorReading__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
pipeline_interfaces__msg__RawSensorReading__Sequence__are_equal(const pipeline_interfaces__msg__RawSensorReading__Sequence * lhs, const pipeline_interfaces__msg__RawSensorReading__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!pipeline_interfaces__msg__RawSensorReading__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
pipeline_interfaces__msg__RawSensorReading__Sequence__copy(
  const pipeline_interfaces__msg__RawSensorReading__Sequence * input,
  pipeline_interfaces__msg__RawSensorReading__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(pipeline_interfaces__msg__RawSensorReading);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    pipeline_interfaces__msg__RawSensorReading * data =
      (pipeline_interfaces__msg__RawSensorReading *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!pipeline_interfaces__msg__RawSensorReading__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          pipeline_interfaces__msg__RawSensorReading__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!pipeline_interfaces__msg__RawSensorReading__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
