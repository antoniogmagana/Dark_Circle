// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from pipeline_interfaces:msg/SensorDataWindow.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "pipeline_interfaces/msg/sensor_data_window.h"


#ifndef PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__FUNCTIONS_H_
#define PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__FUNCTIONS_H_

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

#include "pipeline_interfaces/msg/detail/sensor_data_window__struct.h"

/// Initialize msg/SensorDataWindow message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * pipeline_interfaces__msg__SensorDataWindow
 * )) before or use
 * pipeline_interfaces__msg__SensorDataWindow__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__SensorDataWindow__init(pipeline_interfaces__msg__SensorDataWindow * msg);

/// Finalize msg/SensorDataWindow message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__SensorDataWindow__fini(pipeline_interfaces__msg__SensorDataWindow * msg);

/// Create msg/SensorDataWindow message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * pipeline_interfaces__msg__SensorDataWindow__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
pipeline_interfaces__msg__SensorDataWindow *
pipeline_interfaces__msg__SensorDataWindow__create(void);

/// Destroy msg/SensorDataWindow message.
/**
 * It calls
 * pipeline_interfaces__msg__SensorDataWindow__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__SensorDataWindow__destroy(pipeline_interfaces__msg__SensorDataWindow * msg);

/// Check for msg/SensorDataWindow message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__SensorDataWindow__are_equal(const pipeline_interfaces__msg__SensorDataWindow * lhs, const pipeline_interfaces__msg__SensorDataWindow * rhs);

/// Copy a msg/SensorDataWindow message.
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
pipeline_interfaces__msg__SensorDataWindow__copy(
  const pipeline_interfaces__msg__SensorDataWindow * input,
  pipeline_interfaces__msg__SensorDataWindow * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_type_hash_t *
pipeline_interfaces__msg__SensorDataWindow__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
pipeline_interfaces__msg__SensorDataWindow__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeSource *
pipeline_interfaces__msg__SensorDataWindow__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
pipeline_interfaces__msg__SensorDataWindow__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of msg/SensorDataWindow messages.
/**
 * It allocates the memory for the number of elements and calls
 * pipeline_interfaces__msg__SensorDataWindow__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__SensorDataWindow__Sequence__init(pipeline_interfaces__msg__SensorDataWindow__Sequence * array, size_t size);

/// Finalize array of msg/SensorDataWindow messages.
/**
 * It calls
 * pipeline_interfaces__msg__SensorDataWindow__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__SensorDataWindow__Sequence__fini(pipeline_interfaces__msg__SensorDataWindow__Sequence * array);

/// Create array of msg/SensorDataWindow messages.
/**
 * It allocates the memory for the array and calls
 * pipeline_interfaces__msg__SensorDataWindow__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
pipeline_interfaces__msg__SensorDataWindow__Sequence *
pipeline_interfaces__msg__SensorDataWindow__Sequence__create(size_t size);

/// Destroy array of msg/SensorDataWindow messages.
/**
 * It calls
 * pipeline_interfaces__msg__SensorDataWindow__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
void
pipeline_interfaces__msg__SensorDataWindow__Sequence__destroy(pipeline_interfaces__msg__SensorDataWindow__Sequence * array);

/// Check for msg/SensorDataWindow message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
bool
pipeline_interfaces__msg__SensorDataWindow__Sequence__are_equal(const pipeline_interfaces__msg__SensorDataWindow__Sequence * lhs, const pipeline_interfaces__msg__SensorDataWindow__Sequence * rhs);

/// Copy an array of msg/SensorDataWindow messages.
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
pipeline_interfaces__msg__SensorDataWindow__Sequence__copy(
  const pipeline_interfaces__msg__SensorDataWindow__Sequence * input,
  pipeline_interfaces__msg__SensorDataWindow__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PIPELINE_INTERFACES__MSG__DETAIL__SENSOR_DATA_WINDOW__FUNCTIONS_H_
