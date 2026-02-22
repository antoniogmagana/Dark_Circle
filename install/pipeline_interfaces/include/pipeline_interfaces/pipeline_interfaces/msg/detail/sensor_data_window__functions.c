// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice
#include "pipeline_interfaces/msg/detail/sensor_data_window__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `readings`
#include "pipeline_interfaces/msg/detail/raw_sensor_reading__functions.h"

bool
pipeline_interfaces__msg__SensorDataWindow__init(pipeline_interfaces__msg__SensorDataWindow * msg)
{
  if (!msg) {
    return false;
  }
  // readings
  if (!pipeline_interfaces__msg__RawSensorReading__Sequence__init(&msg->readings, 0)) {
    pipeline_interfaces__msg__SensorDataWindow__fini(msg);
    return false;
  }
  return true;
}

void
pipeline_interfaces__msg__SensorDataWindow__fini(pipeline_interfaces__msg__SensorDataWindow * msg)
{
  if (!msg) {
    return;
  }
  // readings
  pipeline_interfaces__msg__RawSensorReading__Sequence__fini(&msg->readings);
}

bool
pipeline_interfaces__msg__SensorDataWindow__are_equal(const pipeline_interfaces__msg__SensorDataWindow * lhs, const pipeline_interfaces__msg__SensorDataWindow * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // readings
  if (!pipeline_interfaces__msg__RawSensorReading__Sequence__are_equal(
      &(lhs->readings), &(rhs->readings)))
  {
    return false;
  }
  return true;
}

bool
pipeline_interfaces__msg__SensorDataWindow__copy(
  const pipeline_interfaces__msg__SensorDataWindow * input,
  pipeline_interfaces__msg__SensorDataWindow * output)
{
  if (!input || !output) {
    return false;
  }
  // readings
  if (!pipeline_interfaces__msg__RawSensorReading__Sequence__copy(
      &(input->readings), &(output->readings)))
  {
    return false;
  }
  return true;
}

pipeline_interfaces__msg__SensorDataWindow *
pipeline_interfaces__msg__SensorDataWindow__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__SensorDataWindow * msg = (pipeline_interfaces__msg__SensorDataWindow *)allocator.allocate(sizeof(pipeline_interfaces__msg__SensorDataWindow), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(pipeline_interfaces__msg__SensorDataWindow));
  bool success = pipeline_interfaces__msg__SensorDataWindow__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
pipeline_interfaces__msg__SensorDataWindow__destroy(pipeline_interfaces__msg__SensorDataWindow * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    pipeline_interfaces__msg__SensorDataWindow__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
pipeline_interfaces__msg__SensorDataWindow__Sequence__init(pipeline_interfaces__msg__SensorDataWindow__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__SensorDataWindow * data = NULL;

  if (size) {
    data = (pipeline_interfaces__msg__SensorDataWindow *)allocator.zero_allocate(size, sizeof(pipeline_interfaces__msg__SensorDataWindow), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = pipeline_interfaces__msg__SensorDataWindow__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        pipeline_interfaces__msg__SensorDataWindow__fini(&data[i - 1]);
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
pipeline_interfaces__msg__SensorDataWindow__Sequence__fini(pipeline_interfaces__msg__SensorDataWindow__Sequence * array)
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
      pipeline_interfaces__msg__SensorDataWindow__fini(&array->data[i]);
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

pipeline_interfaces__msg__SensorDataWindow__Sequence *
pipeline_interfaces__msg__SensorDataWindow__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  pipeline_interfaces__msg__SensorDataWindow__Sequence * array = (pipeline_interfaces__msg__SensorDataWindow__Sequence *)allocator.allocate(sizeof(pipeline_interfaces__msg__SensorDataWindow__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = pipeline_interfaces__msg__SensorDataWindow__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
pipeline_interfaces__msg__SensorDataWindow__Sequence__destroy(pipeline_interfaces__msg__SensorDataWindow__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    pipeline_interfaces__msg__SensorDataWindow__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
pipeline_interfaces__msg__SensorDataWindow__Sequence__are_equal(const pipeline_interfaces__msg__SensorDataWindow__Sequence * lhs, const pipeline_interfaces__msg__SensorDataWindow__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!pipeline_interfaces__msg__SensorDataWindow__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
pipeline_interfaces__msg__SensorDataWindow__Sequence__copy(
  const pipeline_interfaces__msg__SensorDataWindow__Sequence * input,
  pipeline_interfaces__msg__SensorDataWindow__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(pipeline_interfaces__msg__SensorDataWindow);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    pipeline_interfaces__msg__SensorDataWindow * data =
      (pipeline_interfaces__msg__SensorDataWindow *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!pipeline_interfaces__msg__SensorDataWindow__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          pipeline_interfaces__msg__SensorDataWindow__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!pipeline_interfaces__msg__SensorDataWindow__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
