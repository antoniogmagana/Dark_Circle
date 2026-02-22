// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from pipeline_interfaces:msg/RawSensorReading.idl
// generated code does not contain a copyright notice

#include "pipeline_interfaces/msg/detail/raw_sensor_reading__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_pipeline_interfaces
const rosidl_type_hash_t *
pipeline_interfaces__msg__RawSensorReading__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe7, 0x88, 0xc7, 0x7d, 0x62, 0x28, 0x5a, 0x72,
      0x3b, 0xb2, 0x42, 0x7b, 0x95, 0x25, 0x21, 0x32,
      0x6c, 0x4f, 0x40, 0x2e, 0x51, 0xa7, 0x97, 0x31,
      0x87, 0x93, 0xaa, 0x78, 0x21, 0x22, 0xc9, 0x7e,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
#endif

static char pipeline_interfaces__msg__RawSensorReading__TYPE_NAME[] = "pipeline_interfaces/msg/RawSensorReading";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";

// Define type names, field names, and default values
static char pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__sensor_label[] = "sensor_label";
static char pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__timestamp[] = "timestamp";
static char pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__sensor_values[] = "sensor_values";
static char pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__latitude[] = "latitude";
static char pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__longitude[] = "longitude";

static rosidl_runtime_c__type_description__Field pipeline_interfaces__msg__RawSensorReading__FIELDS[] = {
  {
    {pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__sensor_label, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__timestamp, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    },
    {NULL, 0, 0},
  },
  {
    {pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__sensor_values, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__latitude, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {pipeline_interfaces__msg__RawSensorReading__FIELD_NAME__longitude, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription pipeline_interfaces__msg__RawSensorReading__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
pipeline_interfaces__msg__RawSensorReading__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {pipeline_interfaces__msg__RawSensorReading__TYPE_NAME, 40, 40},
      {pipeline_interfaces__msg__RawSensorReading__FIELDS, 5, 5},
    },
    {pipeline_interfaces__msg__RawSensorReading__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# Add the sensor id to make it scalable\n"
  "string sensor_label\n"
  "\n"
  "# timestamp for sensor reading in seconds\n"
  "builtin_interfaces/Time timestamp\n"
  "\n"
  "# sensor reading value\n"
  "float64 sensor_values\n"
  "\n"
  "# sensor location values\n"
  "float64 latitude\n"
  "float64 longitude";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
pipeline_interfaces__msg__RawSensorReading__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {pipeline_interfaces__msg__RawSensorReading__TYPE_NAME, 40, 40},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 244, 244},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
pipeline_interfaces__msg__RawSensorReading__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *pipeline_interfaces__msg__RawSensorReading__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
