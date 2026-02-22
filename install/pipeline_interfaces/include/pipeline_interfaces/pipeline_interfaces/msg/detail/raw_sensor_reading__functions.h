// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/raw_sensor_reading.h"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__FUNCTIONS_H_
#define PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/action_type_support_struct.h"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_runtime_c/type_description/type_description__struct.h"
#include "rosidl_runtime_c/type_description/type_source__struct.h"
#include "rosidl_runtime_c/type_hash.h"
#include "rosidl_runtime_c/visibility_control.h"
#include "pipeline_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "pipeline_interfaces/msg/detail/raw_sensor_reading__struct.h"

/// Initialize msg/RawSensorReading message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * pipeline_interfaces__msg__RawSensorReading
 * )) before or use
 * pipeline_interfaces__msg__RawSensorReading__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__init(pipeline_interfaces__msg__RawSensorReading * msg);

/// Finalize msg/RawSensorReading message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__RawSensorReading__fini(pipeline_interfaces__msg__RawSensorReading * msg);

/// Create msg/RawSensorReading message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * pipeline_interfaces__msg__RawSensorReading__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
pipeline_interfaces__msg__RawSensorReading *
pipeline_interfaces__msg__RawSensorReading__create(void);

/// Destroy msg/RawSensorReading message.
/**
 * It calls
 * pipeline_interfaces__msg__RawSensorReading__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__RawSensorReading__destroy(pipeline_interfaces__msg__RawSensorReading * msg);

/// Check for msg/RawSensorReading message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__are_equal(const pipeline_interfaces__msg__RawSensorReading * lhs, const pipeline_interfaces__msg__RawSensorReading * rhs);

/// Copy a msg/RawSensorReading message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__copy(
  const pipeline_interfaces__msg__RawSensorReading * input,
  pipeline_interfaces__msg__RawSensorReading * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_type_hash_t *
pipeline_interfaces__msg__RawSensorReading__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
pipeline_interfaces__msg__RawSensorReading__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeSource *
pipeline_interfaces__msg__RawSensorReading__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
pipeline_interfaces__msg__RawSensorReading__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of msg/RawSensorReading messages.
/**
 * It allocates the memory for the number of elements and calls
 * pipeline_interfaces__msg__RawSensorReading__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__Sequence__init(pipeline_interfaces__msg__RawSensorReading__Sequence * array, size_t size);

/// Finalize array of msg/RawSensorReading messages.
/**
 * It calls
 * pipeline_interfaces__msg__RawSensorReading__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__RawSensorReading__Sequence__fini(pipeline_interfaces__msg__RawSensorReading__Sequence * array);

/// Create array of msg/RawSensorReading messages.
/**
 * It allocates the memory for the array and calls
 * pipeline_interfaces__msg__RawSensorReading__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
pipeline_interfaces__msg__RawSensorReading__Sequence *
pipeline_interfaces__msg__RawSensorReading__Sequence__create(size_t size);

/// Destroy array of msg/RawSensorReading messages.
/**
 * It calls
 * pipeline_interfaces__msg__RawSensorReading__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__RawSensorReading__Sequence__destroy(pipeline_interfaces__msg__RawSensorReading__Sequence * array);

/// Check for msg/RawSensorReading message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__Sequence__are_equal(const pipeline_interfaces__msg__RawSensorReading__Sequence * lhs, const pipeline_interfaces__msg__RawSensorReading__Sequence * rhs);

/// Copy an array of msg/RawSensorReading messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__RawSensorReading__Sequence__copy(
  const pipeline_interfaces__msg__RawSensorReading__Sequence * input,
  pipeline_interfaces__msg__RawSensorReading__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__RAW_SENSOR_READING__FUNCTIONS_H_
