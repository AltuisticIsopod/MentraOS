// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// See docs\c_cxx\README.md on generating the Doxygen documentation from this file

/** \mainpage ONNX Runtime
 *
 * ONNX Runtime is a high-performance inference and training graph execution engine for deep learning models.
 *
 * ONNX Runtime's C, C++ APIs offer an easy to use interface to onboard and execute onnx models.
 * - \subpage c_cpp_api "Core C, C++ APIs"
 * - \subpage training_c_cpp_api "Training C, C++ APIs for on-device training"
 *
 * \page c_cpp_api Core C, C++ APIs
 * <h1>C</h1>
 *
 * ::OrtApi - Click here to go to the structure with all C API functions.
 *
 * <h1>C++</h1>
 *
 * ::Ort - Click here to go to the namespace holding all of the C++ wrapper classes
 *
 * It is a set of header only wrapper classes around the C API. The goal is to turn the C style return value error codes into C++ exceptions, and to
 * automate memory management through standard C++ RAII principles.
 *
 * \addtogroup Global
 * ONNX Runtime C API
 * @{
 */

#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** \brief The API version defined in this header
 *
 * This value is used by some API functions to behave as this version of the header expects.
 */
#define ORT_API_VERSION 18

#ifdef __cplusplus
extern "C" {
#endif

//! @}
// SAL2 Definitions
#ifndef _WIN32
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_(X)
#define _Out_writes_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)
#define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ORT_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
#ifdef ORT_DLL_IMPORT
#define ORT_EXPORT __declspec(dllimport)
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL _stdcall
#define ORT_MUST_USE_RESULT
#define ORTCHAR_T wchar_t
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define ORT_EXPORT __attribute__((visibility("default")))
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL
#define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
#define ORTCHAR_T char
#endif

/// ORTCHAR_T, ORT_TSTR are reserved specifically for path handling.
/// All other strings are UTF-8 encoded, use char and std::string
#ifndef ORT_TSTR
#ifdef _WIN32
#define ORT_TSTR(X) L##X
// When X is a macro, L##X is not defined. In this case, we need to use ORT_TSTR_ON_MACRO.
#define ORT_TSTR_ON_MACRO(X) L"" X
#else
#define ORT_TSTR(X) X
#define ORT_TSTR_ON_MACRO(X) X
#endif
#endif

// On Windows, ORT_FILE is a wchar_t version of the __FILE__ macro.
// Otherwise, ORT_FILE is equivalent to __FILE__.
#ifndef ORT_FILE
#define ORT_FILE_INTERNAL(x) ORT_TSTR(x)
#define ORT_FILE ORT_FILE_INTERNAL(__FILE__)
#endif

// Any pointer marked with _In_ or _Out_, cannot be NULL.

// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.

#ifdef __cplusplus
// For any compiler with C++11 support, MSVC 2015 and greater, or Clang version supporting noexcept.
// Such complex condition is needed because compilers set __cplusplus value differently.
#ifndef __has_feature
#define __has_feature(x) 0
#endif
#if ((__cplusplus >= 201103L) || (_MSC_VER >= 1900) || (defined(__has_feature) && __has_feature(cxx_noexcept)))
#define NO_EXCEPTION noexcept
#else
#define NO_EXCEPTION throw()
#endif
#else
#define NO_EXCEPTION
#endif

// __VA_ARGS__ on Windows and Linux are different
#define ORT_API(RETURN_TYPE, NAME, ...) RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_API_STATUS(NAME, ...)                                                                   \
  _Success_(return == 0) _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) \
  NO_EXCEPTION ORT_MUST_USE_RESULT

// XXX: Unfortunately, SAL annotations are known to not work with function pointers
#define ORT_API2_STATUS(NAME, ...) \
  _Check_return_ _Ret_maybenull_ OrtStatusPtr(ORT_API_CALL* NAME)(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT and ORT_EXPORT
#define ORT_API_STATUS_IMPL(NAME, ...) \
  _Success_(return == 0) _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_CLASS_RELEASE(X) void(ORT_API_CALL * Release##X)(_Frees_ptr_opt_ Ort##X * input)

#ifdef __DOXYGEN__
#undef ORT_API_STATUS
#define ORT_API_STATUS(NAME, ...) OrtStatus* NAME(__VA_ARGS__)
#undef ORT_API2_STATUS
#define ORT_API2_STATUS(NAME, ...) OrtStatus* NAME(__VA_ARGS__)
#undef ORT_CLASS_RELEASE
#define ORT_CLASS_RELEASE(X) void Release##X(Ort##X* input)
#undef NO_EXCEPTION
#define NO_EXCEPTION
#endif
/** \addtogroup Global
 * ONNX Runtime C API
 * @{
 */

/** Copied from TensorProto::DataType
 * Currently, Ort doesn't support complex64, complex128
 */
typedef enum ONNXTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   // maps to c type float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,   // maps to c type uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,    // maps to c type int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,  // maps to c type uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,   // maps to c type int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,   // maps to c type int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // maps to c type int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,  // maps to c++ type std::string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,      // maps to c type double
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,      // maps to c type uint32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,      // maps to c type uint64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,    // Non-IEEE floating-point format based on IEEE754 single-precision
  // float 8 types were introduced in onnx 1.14, see https://onnx.ai/onnx/technical/float8.html
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,    // Non-IEEE floating-point format based on IEEE754 single-precision
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,  // Non-IEEE floating-point format based on IEEE754 single-precision
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,      // Non-IEEE floating-point format based on IEEE754 single-precision
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ   // Non-IEEE floating-point format based on IEEE754 single-precision
} ONNXTensorElementDataType;

// Synced with onnx TypeProto oneof
typedef enum ONNXType {
  ONNX_TYPE_UNKNOWN,
  ONNX_TYPE_TENSOR,
  ONNX_TYPE_SEQUENCE,
  ONNX_TYPE_MAP,
  ONNX_TYPE_OPAQUE,
  ONNX_TYPE_SPARSETENSOR,
  ONNX_TYPE_OPTIONAL
} ONNXType;

// These types are synced with internal
// SparseFormatFlags
typedef enum OrtSparseFormat {
  ORT_SPARSE_UNDEFINED = 0,
  ORT_SPARSE_COO = 0x1,
  ORT_SPARSE_CSRC = 0x2,
  ORT_SPARSE_BLOCK_SPARSE = 0x4
} OrtSparseFormat;

// Enum allows to query sparse tensor indices
enum OrtSparseIndicesFormat {
  ORT_SPARSE_COO_INDICES,
  ORT_SPARSE_CSR_INNER_INDICES,
  ORT_SPARSE_CSR_OUTER_INDICES,
  ORT_SPARSE_BLOCK_SPARSE_INDICES
};

/** \brief Logging severity levels
 *
 * In typical API usage, specifying a logging severity level specifies the minimum severity of log messages to show.
 */
typedef enum OrtLoggingLevel {
  ORT_LOGGING_LEVEL_VERBOSE,  ///< Verbose informational messages (least severe).
  ORT_LOGGING_LEVEL_INFO,     ///< Informational messages.
  ORT_LOGGING_LEVEL_WARNING,  ///< Warning messages.
  ORT_LOGGING_LEVEL_ERROR,    ///< Error messages.
  ORT_LOGGING_LEVEL_FATAL,    ///< Fatal error messages (most severe).
} OrtLoggingLevel;

typedef enum OrtErrorCode {
  ORT_OK,
  ORT_FAIL,
  ORT_INVALID_ARGUMENT,
  ORT_NO_SUCHFILE,
  ORT_NO_MODEL,
  ORT_ENGINE_ERROR,
  ORT_RUNTIME_EXCEPTION,
  ORT_INVALID_PROTOBUF,
  ORT_MODEL_LOADED,
  ORT_NOT_IMPLEMENTED,
  ORT_INVALID_GRAPH,
  ORT_EP_FAIL,
} OrtErrorCode;

typedef enum OrtOpAttrType {
  ORT_OP_ATTR_UNDEFINED = 0,
  ORT_OP_ATTR_INT,
  ORT_OP_ATTR_INTS,
  ORT_OP_ATTR_FLOAT,
  ORT_OP_ATTR_FLOATS,
  ORT_OP_ATTR_STRING,
  ORT_OP_ATTR_STRINGS,
} OrtOpAttrType;

//! @}
#define ORT_RUNTIME_CLASS(X) \
  struct Ort##X;             \
  typedef struct Ort##X Ort##X

/** \addtogroup Global
 * ONNX Runtime C API
 * @{
 */
// The actual types defined have an Ort prefix
ORT_RUNTIME_CLASS(Env);
ORT_RUNTIME_CLASS(Status);  // nullptr for Status* indicates success
ORT_RUNTIME_CLASS(MemoryInfo);
ORT_RUNTIME_CLASS(IoBinding);
ORT_RUNTIME_CLASS(Session);  // Don't call ReleaseSession from Dllmain (because session owns a thread pool)
ORT_RUNTIME_CLASS(Value);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(OptionalTypeInfo);
ORT_RUNTIME_CLASS(SessionOptions);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(ModelMetadata);
ORT_RUNTIME_CLASS(ThreadPoolParams);
ORT_RUNTIME_CLASS(ThreadingOptions);
ORT_RUNTIME_CLASS(ArenaCfg);
ORT_RUNTIME_CLASS(PrepackedWeightsContainer);
ORT_RUNTIME_CLASS(TensorRTProviderOptionsV2);
ORT_RUNTIME_CLASS(CUDAProviderOptionsV2);
ORT_RUNTIME_CLASS(CANNProviderOptions);
ORT_RUNTIME_CLASS(DnnlProviderOptions);
ORT_RUNTIME_CLASS(Op);
ORT_RUNTIME_CLASS(OpAttr);
ORT_RUNTIME_CLASS(Logger);
ORT_RUNTIME_CLASS(ShapeInferContext);

#ifdef _WIN32
typedef _Return_type_success_(return == 0) OrtStatus* OrtStatusPtr;
#else
typedef OrtStatus* OrtStatusPtr;
#endif

/** \brief Memory allocation interface
 *
 * Structure of function pointers that defines a memory allocator. This can be created and filled in by the user for custom allocators.
 *
 * When an allocator is passed to any function, be sure that the allocator object is not destroyed until the last allocated object using it is freed.
 */
typedef struct OrtAllocator {
  uint32_t version;                                                                   ///< Must be initialized to ORT_API_VERSION
  void*(ORT_API_CALL* Alloc)(struct OrtAllocator* this_, size_t size);                ///< Returns a pointer to an allocated block of `size` bytes
  void(ORT_API_CALL* Free)(struct OrtAllocator* this_, void* p);                      ///< Free a block of memory previously allocated with OrtAllocator::Alloc
  const struct OrtMemoryInfo*(ORT_API_CALL* Info)(const struct OrtAllocator* this_);  ///< Return a pointer to an ::OrtMemoryInfo that describes this allocator
  /**
   * @brief Optional allocation function to use for memory allocations made during session initialization.
   * Use this function if you want to separate allocations made by ORT during Run() calls from
   * those made during session initialization. This allows for separate memory management strategies for these allocations.
   */
  void*(ORT_API_CALL* Reserve)(struct OrtAllocator* this_, size_t size);  ///< Returns a pointer to an allocated block of `size` bytes
} OrtAllocator;

typedef void(ORT_API_CALL* OrtLoggingFunction)(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);

/** \brief Graph optimization level
 *
 * Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels
 * for an in-depth understanding of the Graph Optimization Levels.
 */
typedef enum GraphOptimizationLevel {
  ORT_DISABLE_ALL = 0,
  ORT_ENABLE_BASIC = 1,
  ORT_ENABLE_EXTENDED = 2,
  ORT_ENABLE_ALL = 99
} GraphOptimizationLevel;

typedef enum ExecutionMode {
  ORT_SEQUENTIAL = 0,
  ORT_PARALLEL = 1,
} ExecutionMode;

/** \brief Language projection identifiers
 * /see OrtApi::SetLanguageProjection
 */
typedef enum OrtLanguageProjection {
  ORT_PROJECTION_C = 0,
  ORT_PROJECTION_CPLUSPLUS = 1,
  ORT_PROJECTION_CSHARP = 2,
  ORT_PROJECTION_PYTHON = 3,
  ORT_PROJECTION_JAVA = 4,
  ORT_PROJECTION_WINML = 5,
  ORT_PROJECTION_NODEJS = 6,
} OrtLanguageProjection;

struct OrtKernelInfo;
typedef struct OrtKernelInfo OrtKernelInfo;
struct OrtKernelContext;
typedef struct OrtKernelContext OrtKernelContext;
struct OrtCustomOp;
typedef struct OrtCustomOp OrtCustomOp;

typedef enum OrtAllocatorType {
  OrtInvalidAllocator = -1,
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1
} OrtAllocatorType;

/** \brief Memory types for allocated memory, execution provider specific types should be extended in each provider.
 */
// Whenever this struct is updated, please also update the MakeKey function in onnxruntime / core / framework / execution_provider.cc
typedef enum OrtMemType {
  OrtMemTypeCPUInput = -2,              ///< Any CPU memory used by non-CPU execution provider
  OrtMemTypeCPUOutput = -1,             ///< CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeCPU = OrtMemTypeCPUOutput,  ///< Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeDefault = 0,                ///< The default allocator for execution provider
} OrtMemType;

/** \brief This mimics OrtDevice type constants so they can be returned in the API
 */
typedef enum OrtMemoryInfoDeviceType {
  OrtMemoryInfoDeviceType_CPU = 0,
  OrtMemoryInfoDeviceType_GPU = 1,
  OrtMemoryInfoDeviceType_FPGA = 2
} OrtMemoryInfoDeviceType;

/** \brief Algorithm to use for cuDNN Convolution Op
 */
typedef enum OrtCudnnConvAlgoSearch {
  OrtCudnnConvAlgoSearchExhaustive,  // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
  OrtCudnnConvAlgoSearchHeuristic,   // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
  OrtCudnnConvAlgoSearchDefault,     // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
} OrtCudnnConvAlgoSearch;

/** \brief CUDA Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_CUDA
 */
typedef struct OrtCUDAProviderOptions {
#ifdef __cplusplus
  OrtCUDAProviderOptions()
      : device_id{},
        cudnn_conv_algo_search{OrtCudnnConvAlgoSearchExhaustive},
        gpu_mem_limit{SIZE_MAX},
        arena_extend_strategy{},
        do_copy_in_default_stream{1},
        has_user_compute_stream{},
        user_compute_stream{},
        default_memory_arena_cfg{},
        tunable_op_enable{false},
        tunable_op_tuning_enable{false},
        tunable_op_max_tuning_duration_ms{} {}
#endif

  /** \brief CUDA device Id
   *   Defaults to 0.
   */
  int device_id;

  /** \brief CUDA Convolution algorithm search configuration.
   *   See enum OrtCudnnConvAlgoSearch for more details.
   *   Defaults to OrtCudnnConvAlgoSearchExhaustive.
   */
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search;

  /** \brief CUDA memory limit (To use all possible memory pass in maximum size_t)
   *   Defaults to SIZE_MAX.
   *   \note If a ::OrtArenaCfg has been applied, it will override this field
   */
  size_t gpu_mem_limit;

  /** \brief Strategy used to grow the memory arena
   *   0 = kNextPowerOfTwo<br>
   *   1 = kSameAsRequested<br>
   *   Defaults to 0.
   *   \note If a ::OrtArenaCfg has been applied, it will override this field
   */
  int arena_extend_strategy;

  /** \brief Flag indicating if copying needs to take place on the same stream as the compute stream in the CUDA EP
   *   0 = Use separate streams for copying and compute.
   *   1 = Use the same stream for copying and compute.
   *   Defaults to 1.
   *   WARNING: Setting this to 0 may result in data races for some models.
   *   Please see issue #4829 for more details.
   */
  int do_copy_in_default_stream;

  /** \brief Flag indicating if there is a user provided compute stream
   *   Defaults to 0.
   */
  int has_user_compute_stream;

  /** \brief User provided compute stream.
   *   If provided, please set `has_user_compute_stream` to 1.
   */
  void* user_compute_stream;

  /** \brief CUDA memory arena configuration parameters
   */
  OrtArenaCfg* default_memory_arena_cfg;

  /** \brief Enable TunableOp for using.
   *   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.
   *   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_ENABLE.
   */
  int tunable_op_enable;

  /** \brief Enable TunableOp for tuning.
   *   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.
   *   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_TUNING_ENABLE.
   */
  int tunable_op_tuning_enable;

  /** \brief Max tuning duration time limit for each instance of TunableOp.
   *   Defaults to 0 to disable the limit.
   */
  int tunable_op_max_tuning_duration_ms;

} OrtCUDAProviderOptions;

/** \brief ROCM Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_ROCM
 */
typedef struct OrtROCMProviderOptions {
#ifdef __cplusplus
  OrtROCMProviderOptions()
      : device_id{},
        miopen_conv_exhaustive_search{0},
        gpu_mem_limit{SIZE_MAX},
        arena_extend_strategy{},
        do_copy_in_default_stream{1},
        has_user_compute_stream{},
        user_compute_stream{},
        default_memory_arena_cfg{},
        enable_hip_graph{false},
        tunable_op_enable{false},
        tunable_op_tuning_enable{false},
        tunable_op_max_tuning_duration_ms{} {}
#endif

  /** \brief ROCM device Id
   *   Defaults to 0.
   */
  int device_id;

  /** \brief ROCM MIOpen Convolution algorithm exaustive search option.
   *   Defaults to 0 (false).
   */
  int miopen_conv_exhaustive_search;

  /** \brief ROCM memory limit (To use all possible memory pass in maximum size_t)
   *   Defaults to SIZE_MAX.
   *   \note If a ::OrtArenaCfg has been applied, it will override this field
   */
  size_t gpu_mem_limit;

  /** \brief Strategy used to grow the memory arena
   *   0 = kNextPowerOfTwo<br>
   *   1 = kSameAsRequested<br>
   *   Defaults to 0.
   *   \note If a ::OrtArenaCfg has been applied, it will override this field
   */
  int arena_extend_strategy;

  /** \brief Flag indicating if copying needs to take place on the same stream as the compute stream in the ROCM EP
   *   0 = Use separate streams for copying and compute.
   *   1 = Use the same stream for copying and compute.
   *   Defaults to 1.
   *   WARNING: Setting this to 0 may result in data races for some models.
   *   Please see issue #4829 for more details.
   */
  int do_copy_in_default_stream;

  /** \brief Flag indicating if there is a user provided compute stream
   *   Defaults to 0.
   */
  int has_user_compute_stream;

  /** \brief User provided compute stream.
   *   If provided, please set `has_user_compute_stream` to 1.
   */
  void* user_compute_stream;

  /** \brief ROCM memory arena configuration parameters
   */
  OrtArenaCfg* default_memory_arena_cfg;

  int enable_hip_graph;

  /** \brief Enable TunableOp for using.
   *   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.
   *   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_ENABLE.
   */
  int tunable_op_enable;

  /** \brief Enable TunableOp for tuning.
   *   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.
   *   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_TUNING_ENABLE.
   */
  int tunable_op_tuning_enable;

  /** \brief Max tuning duration time limit for each instance of TunableOp.
   *   Defaults to 0 to disable the limit.
   */
  int tunable_op_max_tuning_duration_ms;

} OrtROCMProviderOptions;

/** \brief TensorRT Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
 */
typedef struct OrtTensorRTProviderOptions {
  int device_id;                                ///< CUDA device id (0 = default device)
  int has_user_compute_stream;                  // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                    // user specified CUDA compute stream.
  int trt_max_partition_iterations;             // maximum iterations for TensorRT parser to get capability
  int trt_min_subgraph_size;                    // minimum size of TensorRT subgraphs
  size_t trt_max_workspace_size;                // maximum workspace size for TensorRT.
  int trt_fp16_enable;                          // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int trt_int8_enable;                          // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* trt_int8_calibration_table_name;  // TensorRT INT8 calibration table name.
  int trt_int8_use_native_calibration_table;    // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int trt_dla_enable;                           // enable DLA. Default 0 = false, nonzero = true
  int trt_dla_core;                             // DLA core number. Default 0
  int trt_dump_subgraphs;                       // dump TRT subgraph. Default 0 = false, nonzero = true
  int trt_engine_cache_enable;                  // enable engine caching. Default 0 = false, nonzero = true
  const char* trt_engine_cache_path;            // specify engine cache path
  int trt_engine_decryption_enable;             // enable engine decryption. Default 0 = false, nonzero = true
  const char* trt_engine_decryption_lib_path;   // specify engine decryption library path
  int trt_force_sequential_engine_build;        // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
  // This is the legacy struct and don't add new fields here.
  // For new field that can be represented by string, please add it in include/onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h
  // For non-string field, need to create a new separate api to handle it.
} OrtTensorRTProviderOptions;

/** \brief MIGraphX Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX
 */
typedef struct OrtMIGraphXProviderOptions {
  int device_id;                                     // hip device id.
  int migraphx_fp16_enable;                          // MIGraphX FP16 precision. Default 0 = false, nonzero = true
  int migraphx_int8_enable;                          // MIGraphX INT8 precision. Default 0 = false, nonzero = true
  int migraphx_use_native_calibration_table;         // MIGraphx INT8 cal table. Default 0 = false, noznero = true
  const char* migraphx_int8_calibration_table_name;  // MIGraphx INT8 calibration table name
} OrtMIGraphXProviderOptions;

/** \brief OpenVINO Provider Options
 *
 * \see OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO
 */
typedef struct OrtOpenVINOProviderOptions {
#ifdef __cplusplus
  OrtOpenVINOProviderOptions() : device_type{},
                                 enable_npu_fast_compile{},
                                 device_id{},
                                 num_of_threads{},
                                 cache_dir{},
                                 context{},
                                 enable_opencl_throttling{},
                                 enable_dynamic_shapes{} {}
#endif
  /** \brief Device type string
   *
   * Valid settings are one of: "CPU_FP32", "CPU_FP16", "GPU_FP32", "GPU_FP16"
   */
  const char* device_type;
  unsigned char enable_npu_fast_compile;  ///< 0 = disabled, nonzero = enabled
  const char* device_id;
  size_t num_of_threads;  ///< 0 = Use default number of threads
  const char* cache_dir;  // path is set to empty by default
  void* context;
  unsigned char enable_opencl_throttling;  ///< 0 = disabled, nonzero = enabled
  unsigned char enable_dynamic_shapes;     ///< 0 = disabled, nonzero = enabled
} OrtOpenVINOProviderOptions;

struct OrtApi;
typedef struct OrtApi OrtApi;

struct OrtTrainingApi;
typedef struct OrtTrainingApi OrtTrainingApi;

/** \brief The helper interface to get the right version of OrtApi
 *
 * Get a pointer to this structure through ::OrtGetApiBase
 */
struct OrtApiBase {
  /** \brief Get a pointer to the requested version of the ::OrtApi
   *
   * \param[in] version Must be ::ORT_API_VERSION
   * \return The ::OrtApi for the version requested, nullptr will be returned if this version is unsupported, for example when using a runtime
   *   older than the version created with this header file.
   *
   * One can call GetVersionString() to get the version of the Onnxruntime library for logging
   * and error reporting purposes.
   */
  const OrtApi*(ORT_API_CALL* GetApi)(uint32_t version)NO_EXCEPTION;

  /** \brief Returns a null terminated string of the version of the Onnxruntime library (eg: "1.8.1")
   *
   *  \return UTF-8 encoded version string. Do not deallocate the returned buffer.
   */
  const char*(ORT_API_CALL* GetVersionString)(void)NO_EXCEPTION;
};

typedef struct OrtApiBase OrtApiBase;

/** \brief The Onnxruntime library's entry point to access the C API
 *
 * Call this to get the a pointer to an ::OrtApiBase
 */
ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION;

/** \brief Thread work loop function
 *
 * Onnxruntime will provide the working loop on custom thread creation
 * Argument is an onnxruntime built-in type which will be provided when thread pool calls OrtCustomCreateThreadFn
 */
typedef void (*OrtThreadWorkerFn)(void* ort_worker_fn_param);

typedef const struct OrtCustomHandleType {
  char __place_holder;
}* OrtCustomThreadHandle;

/** \brief Ort custom thread creation function
 *
 * The function should return a thread handle to be used in onnxruntime thread pools
 * Onnxruntime will throw exception on return value of nullptr or 0, indicating that the function failed to create a thread
 */
typedef OrtCustomThreadHandle (*OrtCustomCreateThreadFn)(void* ort_custom_thread_creation_options, OrtThreadWorkerFn ort_thread_worker_fn, void* ort_worker_fn_param);

/** \brief Custom thread join function
 *
 * Onnxruntime thread pool destructor will call the function to join a custom thread.
 * Argument ort_custom_thread_handle is the value returned by OrtCustomCreateThreadFn
 */
typedef void (*OrtCustomJoinThreadFn)(OrtCustomThreadHandle ort_custom_thread_handle);

typedef OrtStatus*(ORT_API_CALL* RegisterCustomOpsFn)(OrtSessionOptions* options, const OrtApiBase* api);

/** \brief Callback function for RunAsync
 *
 * \param[in] user_data User specific data that passed back to the callback
 * \param[out] outputs On succeed, outputs host inference results, on error, the value will be nullptr
 * \param[out] num_outputs Number of outputs, on error, the value will be zero
 * \param[out] status On error, status will provide details
 */
typedef void (*RunAsyncCallbackFn)(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status);

/** \brief The C API
 *
 * All C API functions are defined inside this structure as pointers to functions.
 * Call OrtApiBase::GetApi to get a pointer to it
 *
 * \nosubgrouping
 */
struct OrtApi {
  /// \name OrtStatus
  /// @{

  /**
   * \brief Create an OrtStatus from a null terminated string
   *
   * \param[in] code
   * \param[in] msg A null-terminated string. Its contents will be copied.
   * \return A new OrtStatus object, must be destroyed with OrtApi::ReleaseStatus
   */
  OrtStatus*(ORT_API_CALL* CreateStatus)(OrtErrorCode code, _In_ const char* msg)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /** \brief Get OrtErrorCode from OrtStatus
   *
   * \param[in] status
   * \return OrtErrorCode that \p status was created with
   */
  OrtErrorCode(ORT_API_CALL* GetErrorCode)(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /** \brief Get error string from OrtStatus
   *
   * \param[in] status
   * \return The error message inside the `status`. Do not free the returned value.
   */
  const char*(ORT_API_CALL* GetErrorMessage)(_In_ const OrtStatus* status)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /// @}
  /// \name OrtEnv
  /// @{

  /** \brief Create an OrtEnv
   *
   * \note Invoking this function will return the same instance of the environment as that returned by a previous call
   * to another env creation function; all arguments to this function will be ignored.
   * \param[in] log_severity_level The log severity level.
   * \param[in] logid The log identifier.
   * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateEnv, OrtLoggingLevel log_severity_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  /** \brief Create an OrtEnv
   *
   * \note Invoking this function will return the same instance of the environment as that returned by a previous call
   * to another env creation function; all arguments to this function will be ignored. If you want to provide your
   * own logging function, consider setting it using the SetUserLoggingFunction API instead.
   * \param[in] logging_function A pointer to a logging function.
   * \param[in] logger_param A pointer to arbitrary data passed as the ::OrtLoggingFunction `param` parameter to
   *                         `logging_function`. This parameter is optional.
   * \param[in] log_severity_level The log severity level.
   * \param[in] logid The log identifier.
   * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateEnvWithCustomLogger, _In_ OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                  _In_ OrtLoggingLevel log_severity_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  /** \brief Enable Telemetry
   *
   * \note Telemetry events are on by default since they are lightweight
   * \param[in] env
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(EnableTelemetryEvents, _In_ const OrtEnv* env);
  /** \brief Disable Telemetry
   *
   * \see OrtApi::EnableTelemetryEvents
   * \param[in] env
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(DisableTelemetryEvents, _In_ const OrtEnv* env);

  /// @}
  /// \name OrtSession
  /// @{

  /** \brief Create an OrtSession from a model file
   *
   * \param[in] env
   * \param[in] model_path
   * \param[in] options
   * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  // TODO: document the path separator convention? '/' vs '\'
  // TODO: should specify the access characteristics of model_path. Is this read only during the
  // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
  // and continue to access throughout the OrtSession lifetime?
  //  What sort of access is needed to model_path : read or read/write?
  ORT_API2_STATUS(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  /** \brief Create an OrtSession from memory
   *
   * \param[in] env
   * \param[in] model_data
   * \param[in] model_data_length
   * \param[in] options
   * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  /** \brief Run the model in an ::OrtSession
   *
   * Will not return until the model run has completed. Multiple threads might be used to run the model based on
   * the options in the ::OrtSession and settings used when creating the ::OrtEnv
   *
   * \param[in] session
   * \param[in] run_options If nullptr, will use a default ::OrtRunOptions
   * \param[in] input_names Array of null terminated UTF8 encoded strings of the input names
   * \param[in] inputs Array of ::OrtValue%s of the input values
   * \param[in] input_len Number of elements in the input_names and inputs arrays
   * \param[in] output_names Array of null terminated UTF8 encoded strings of the output names
   * \param[in] output_names_len Number of elements in the output_names and outputs array
   * \param[out] outputs Array of ::OrtValue%s that the outputs are stored in. This can also be
   *     an array of nullptr values, in this case ::OrtValue objects will be allocated and pointers
   *     to them will be set into the `outputs` array.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(Run, _Inout_ OrtSession* session, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(input_len) const char* const* input_names,
                  _In_reads_(input_len) const OrtValue* const* inputs, size_t input_len,
                  _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                  _Inout_updates_all_(output_names_len) OrtValue** outputs);

  /// @}
  /// \name OrtSessionOptions
  /// @{

  /** \brief Create an ::OrtSessionOptions object
   *
   * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
   * functions to enable them in the session:<br>
   *   OrtSessionOptionsAppendExecutionProvider_CPU<br>
   *   OrtSessionOptionsAppendExecutionProvider_CUDA<br>
   *   OrtSessionOptionsAppendExecutionProvider_(remaining providers...)<br>
   * The order they are called indicates the preference order as well. In other words call this method
   * on your most preferred execution provider first followed by the less preferred ones.
   * If none are called Ort will use its internal CPU execution provider.
   *
   * \param[out] options The newly created OrtSessionOptions. Must be freed with OrtApi::ReleaseSessionOptions
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateSessionOptions, _Outptr_ OrtSessionOptions** options);

  /** \brief Set filepath to save optimized model after graph level transformations
   *
   * \param[in] options
   * \param[in] optimized_model_filepath
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetOptimizedModelFilePath, _Inout_ OrtSessionOptions* options,
                  _In_ const ORTCHAR_T* optimized_model_filepath);

  /** \brief Create a copy of an existing ::OrtSessionOptions
   *
   * \param[in] in_options OrtSessionOptions to copy
   * \param[out] out_options Returned newly created ::OrtSessionOptions. Must be freed with OrtApi::ReleaseSessionOptions
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CloneSessionOptions, _In_ const OrtSessionOptions* in_options,
                  _Outptr_ OrtSessionOptions** out_options);

  /** \brief Set execution mode
   *
   * Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
   *  has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
   *  See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
   *
   * \param[in] options
   * \param[in] execution_mode
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetSessionExecutionMode, _Inout_ OrtSessionOptions* options, ExecutionMode execution_mode);

  /** \brief Enable profiling for a session
   *
   * \param[in] options
   * \param[in] profile_file_prefix
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(EnableProfiling, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);

  /** \brief Disable profiling for a session
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(DisableProfiling, _Inout_ OrtSessionOptions* options);

  /** \brief Enable the memory pattern optimization
   *
   * The idea is if the input shapes are the same, we could trace the internal memory allocation
   * and generate a memory pattern for future request. So next time we could just do one allocation
   * with a big chunk for all the internal memory allocation.
   * \note Memory pattern optimization is only available when Sequential Execution mode is enabled (see OrtApi::SetSessionExecutionMode)
   *
   * \see OrtApi::DisableMemPattern
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(EnableMemPattern, _Inout_ OrtSessionOptions* options);

  /** \brief Disable the memory pattern optimization
   *
   * \see OrtApi::EnableMemPattern
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(DisableMemPattern, _Inout_ OrtSessionOptions* options);

  /** \brief Enable the memory arena on CPU
   *
   * Arena may pre-allocate memory for future usage.
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(EnableCpuMemArena, _Inout_ OrtSessionOptions* options);

  /** \brief Disable the memory arena on CPU
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(DisableCpuMemArena, _Inout_ OrtSessionOptions* options);

  /** \brief Set session log id
   *
   * \param[in] options
   * \param[in] logid The log identifier.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetSessionLogId, _Inout_ OrtSessionOptions* options, const char* logid);

  /** \brief Set session log verbosity level
   *
   * Applies to session load, initialization, etc
   *
   * \param[in] options
   * \param[in] session_log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetSessionLogVerbosityLevel, _Inout_ OrtSessionOptions* options, int session_log_verbosity_level);

  /** \brief Set session log severity level
   *
   * \param[in] options
   * \param[in] session_log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetSessionLogSeverityLevel, _Inout_ OrtSessionOptions* options, int session_log_severity_level);

  /** \brief Set the optimization level to apply when loading a graph
   *
   * Please see https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html for an in-depth explanation
   * \param[in,out] options The session options object
   * \param[in] graph_optimization_level The optimization level
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetSessionGraphOptimizationLevel, _Inout_ OrtSessionOptions* options,
                  GraphOptimizationLevel graph_optimization_level);

  /** \brief Sets the number of threads used to parallelize the execution within nodes
   *
   * When running a single node operation, ex. add, this sets the maximum number of threads to use.
   *
   * \note If built with OpenMP, this has no effect on the number of threads used. In this case
   *       use the OpenMP env variables to configure the number of intra op num threads.
   *
   * \param[in] options
   * \param[in] intra_op_num_threads Number of threads to use<br>
   *   A value of 0 will use the default number of threads<br>
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);

  /** \brief Sets the number of threads used to parallelize the execution of the graph
   *
   * If nodes can be run in parallel, this sets the maximum number of threads to use to run them in parallel.
   *
   * \note If sequential execution is enabled this value is ignored, it acts as if it was set to 1.
   *
   * \param[in] options
   * \param[in] inter_op_num_threads Number of threads to use<br>
   *   A value of 0 will use the default number of threads<br>
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

  /// @}
  /// \name OrtCustomOpDomain
  /// @{

  /** \brief Create a custom op domain
   *
   * \param[in] domain
   * \param[out] out Newly created domain. Must be freed with OrtApi::ReleaseCustomOpDomain
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

  /** \brief Add a custom op to a custom op domain
   *
   * \note The OrtCustomOp* pointer must remain valid until the ::OrtCustomOpDomain using it is released
   *
   * \param[in] custom_op_domain
   * \param[in] op
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);

  /// @}
  /// \name OrtSessionOptions
  /// @{

  /** \brief Add custom op domain to a session options
   *
   * \note The OrtCustomOpDomain* must not be deleted until all sessions using it are released
   *
   * \param[in] options
   * \param[in] custom_op_domain
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);

  /** \deprecated Use OrtApi::RegisterCustomOpsLibrary_V2.
   *
   * Registers custom ops from a shared library.
   *
   * Loads a shared library (dll on windows, so on linux, etc) named 'library_path' and looks for this entry point:
   *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
   * It then passes in the provided session options to this function along with the api base.
   * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
   * session options are destroyed, or if an error occurs and it is non null.
   *
   * \param[in] options
   * \param[in] library_path
   * \param[out] library_handle OS specific handle to the loaded library (Use FreeLibrary on Windows, dlclose on Linux, etc.. to unload)
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, _Outptr_ void** library_handle);

  /// @}
  /// \name OrtSession
  /// @{

  /** \brief Get input count for a session
   *
   * This number must also match the number of inputs passed to OrtApi::Run
   *
   * \see OrtApi::SessionGetInputTypeInfo, OrtApi::SessionGetInputName, OrtApi::Session
   *
   * \param[in] session
   * \param[out] out Number of inputs
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetInputCount, _In_ const OrtSession* session, _Out_ size_t* out);

  /** \brief Get output count for a session
   *
   * This number must also match the number of outputs returned by OrtApi::Run
   *
   * \see OrtApi::SessionGetOutputTypeInfo, OrtApi::SessionGetOutputName, OrtApi::Session
   *
   * \param[in] session
   * \param[out] out Number of outputs
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOutputCount, _In_ const OrtSession* session, _Out_ size_t* out);

  /** \brief Get overridable initializer count
   *
   * \see OrtApi::SessionGetOverridableInitializerTypeInfo, OrtApi::SessionGetOverridableInitializerName
   *
   * \param[in] session
   * \param[in] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOverridableInitializerCount, _In_ const OrtSession* session, _Out_ size_t* out);

  /** \brief Get input type information
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
   * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetInputTypeInfo, _In_ const OrtSession* session, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /** \brief Get output type information
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
   * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOutputTypeInfo, _In_ const OrtSession* session, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /** \brief Get overridable initializer type information
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOverridableInitializerCount returns (exclusive)
   * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* session, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /** \brief Get input name
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
   * \param[in] allocator
   * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetInputName, _In_ const OrtSession* session, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /** \brief Get output name
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
   * \param[in] allocator
   * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOutputName, _In_ const OrtSession* session, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /** \brief Get overridable initializer name
   *
   * \param[in] session
   * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOverridableInitializerCount returns (exclusive)
   * \param[in] allocator
   * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetOverridableInitializerName, _In_ const OrtSession* session, size_t index,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /// @}
  /// \name OrtRunOptions
  /// @{

  /** \brief Create an OrtRunOptions
   *
   * \param[out] out Returned newly created ::OrtRunOptions. Must be freed with OrtApi::ReleaseRunOptions
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateRunOptions, _Outptr_ OrtRunOptions** out);

  /** \brief Set per-run log verbosity level
   *
   * \see OrtApi::RunOptionsGetRunLogVerbosityLevel
   *
   * \param[in] options
   * \param[in] log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int log_verbosity_level);

  /** \brief Set per-run log severity level
   *
   * \see OrtApi::RunOptionsGetRunLogSeverityLevel
   *
   * \param[in] options
   * \param[in] log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
   */
  ORT_API2_STATUS(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int log_severity_level);

  /** \brief Set per-run tag
   *
   * This is used in a per-run log identifier.
   *
   * \see OrtApi::RunOptionsGetRunTag
   *
   * \param[in] options
   * \param[in] run_tag The run tag.
   */
  ORT_API2_STATUS(RunOptionsSetRunTag, _Inout_ OrtRunOptions* options, _In_ const char* run_tag);

  /** \brief Get per-run log verbosity level
   *
   * \see OrtApi::RunOptionsSetRunLogVerbosityLevel
   *
   * \param[in] options
   * \param[out] log_verbosity_level \snippet{doc} snippets.dox Log Verbosity Level
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options,
                  _Out_ int* log_verbosity_level);

  /** \brief Get per-run log severity level
   *
   * \see OrtApi::RunOptionsSetRunLogSeverityLevel
   *
   * \param[in] options
   * \param[out] log_severity_level The log severity level (refer to ::OrtLoggingLevel for possible values).
   */
  ORT_API2_STATUS(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* log_severity_level);

  /** \brief Get per-run tag
   *
   * This is used in a per-run log identifier.
   *
   * \see OrtApi::RunOptionsSetRunTag
   *
   * \param[in] options
   * \param[out] run_tag The run tag.
   *                     Do not free this value, it is owned by `options`. It will be invalidated if the run tag
   *                     changes (i.e., with OrtApi::RunOptionsSetRunTag) or `options` is freed.
   */
  ORT_API2_STATUS(RunOptionsGetRunTag, _In_ const OrtRunOptions* options, _Out_ const char** run_tag);

  /** \brief Set terminate flag
   *
   * If a currently executing session needs to be force terminated, this can be called from another thread to force it to fail with an error.
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);

  /** \brief Clears the terminate flag
   *
   * Used so the OrtRunOptions instance can be used in a new OrtApi::Run call without it instantly terminating
   *
   * \param[in] options
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

  /// @}
  /// \name OrtValue
  /// @{

  /** \brief Create a tensor
   *
   * Create a tensor using a supplied ::OrtAllocator
   *
   * \param[in] allocator
   * \param[in] shape Pointer to the tensor shape dimensions.
   * \param[in] shape_len The number of tensor shape dimensions.
   * \param[in] type
   * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* shape, size_t shape_len,
                  ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

  /** \brief Create a tensor backed by a user supplied buffer
   *
   * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
   * p_data is owned by caller. ReleaseValue won't release p_data.
   *
   * \param[in] info Memory description of where the p_data buffer resides (CPU vs GPU etc).
   * \param[in] p_data Pointer to the data buffer.
   * \param[in] p_data_len The number of bytes in the data buffer.
   * \param[in] shape Pointer to the tensor shape dimensions.
   * \param[in] shape_len The number of tensor shape dimensions.
   * \param[in] type The data type.
   * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                  size_t p_data_len, _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                  _Outptr_ OrtValue** out);

  /** \brief Return if an ::OrtValue is a tensor type
   *
   * \param[in] value A tensor type (string tensors are not supported)
   * \param[out] out Set to 1 iff ::OrtValue is a tensor, 0 otherwise
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(IsTensor, _In_ const OrtValue* value, _Out_ int* out);

  /** \brief Get a pointer to the raw data inside a tensor
   *
   * Used to read/write/modify the internal tensor data directly.
   * \note The returned pointer is valid until the \p value is destroyed.
   *
   * \param[in] value A tensor type (string tensors are not supported)
   * \param[out] out Filled in with a pointer to the internal storage
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetTensorMutableData, _In_ OrtValue* value, _Outptr_ void** out);

  /** \brief Set all strings at once in a string tensor
   *
   * \param[in,out] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   * \param[in] s An array of strings. Each string in this array must be null terminated.
   * \param[in] s_len Count of strings in s (Must match the size of \p value's tensor shape)
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);

  /** \brief Get total byte length for all strings in a string tensor
   *
   * Typically used with OrtApi::GetStringTensorContent
   *
   * \param[in] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   * \param[out] len Total byte length of all strings (does not include trailing nulls)
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);

  /** \brief Get all strings from a string tensor
   *
   * An example of the results:<br>
   * Given \p value is a string tensor with the strings { "This" "is" "a" "test" }<br>
   * \p s must have a size of 11 bytes<br>
   * \p offsets must have 4 elements<br>
   * After the call, these values will be filled in:<br>
   * \p s will contain "Thisisatest"<br>
   * \p offsets will contain { 0, 4, 6, 7 }<br>
   * The length of the last string is just s_len - offsets[last]
   *
   * \param[in] value A tensor of type ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   * \param[in] s Buffer to sequentially write all tensor strings to. Each string is NOT null-terminated.
   * \param[in] s_len Number of bytes of buffer pointed to by \p s (Get it from OrtApi::GetStringTensorDataLength)
   * \param[out] offsets Array of start offsets into the strings written to \p s
   * \param[in] offsets_len Number of elements in offsets
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                  size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);

  /// @}
  /// \name OrtTypeInfo
  /// @{

  /** \brief Get ::OrtTensorTypeAndShapeInfo from an ::OrtTypeInfo
   *
   * \param[in] type_info
   * \param[out] out Do not free this value, it will be valid until type_info is freed.
   *             If type_info does not represent tensor, this value will be set to nullptr.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);

  /** \brief Get ::ONNXType from ::OrtTypeInfo
   *
   * \param[in] type_info
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ enum ONNXType* out);

  /// @}
  /// \name OrtTensorTypeAndShapeInfo
  /// @{

  /** \brief Create an ::OrtTensorTypeAndShapeInfo object
   *
   * \param[out] out Returns newly created ::OrtTensorTypeAndShapeInfo. Must be freed with OrtApi::ReleaseTensorTypeAndShapeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  /** \brief Set element type in ::OrtTensorTypeAndShapeInfo
   *
   * \param[in] info
   * \param[in] type
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo* info, enum ONNXTensorElementDataType type);

  /** \brief Set shape information in ::OrtTensorTypeAndShapeInfo
   *
   * \param[in] info
   * \param[in] dim_values Array with `dim_count` elements. Can contain negative values.
   * \param[in] dim_count Number of elements in `dim_values`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  /** \brief Get element type in ::OrtTensorTypeAndShapeInfo
   *
   * \see OrtApi::SetTensorElementType
   *
   * \param[in] info
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo* info,
                  _Out_ enum ONNXTensorElementDataType* out);

  /** \brief Get dimension count in ::OrtTensorTypeAndShapeInfo
   *
   * \see OrtApi::GetDimensions
   *
   * \param[in] info
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

  /** \brief Get dimensions in ::OrtTensorTypeAndShapeInfo
   *
   * \param[in] info
   * \param[out] dim_values Array with `dim_values_length` elements. On return, filled with the dimensions stored in the ::OrtTensorTypeAndShapeInfo
   * \param[in] dim_values_length Number of elements in `dim_values`. Use OrtApi::GetDimensionsCount to get this value
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values,
                  size_t dim_values_length);

  /** \brief Get symbolic dimension names in ::OrtTensorTypeAndShapeInfo
   *
   * \param[in] info
   * \param[in] dim_params Array with `dim_params_length` elements. On return filled with pointers to null terminated strings of the dimension names
   * \param[in] dim_params_length Number of elements in `dim_params`. Use OrtApi::GetDimensionsCount to get this value
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                  _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);

  /** \brief Get total number of elements in a tensor shape from an ::OrtTensorTypeAndShapeInfo
   *
   * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
   * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
   *
   * Examples:<br>
   * [] = 1<br>
   * [1,3,4] = 12<br>
   * [2,0,4] = 0<br>
   * [-1,3,4] = -1<br>
   *
   * \param[in] info
   * \param[out] out Number of elements
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

  /// @}
  /// \name OrtValue
  /// @{

  /** \brief Get type and shape information from a tensor ::OrtValue
   *
   * \param[in] value Must be a tensor (not a map/sequence/etc) or will return failure
   * \param[out] out Newly created ::OrtTensorTypeAndShapeInfo. Must be freed with OrtApi::ReleaseTensorTypeAndShapeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  /** \brief Get type information of an OrtValue
   *
   * \param[in] value
   * \param[out] out Newly created ::OrtTypeInfo. Must be freed with OrtApi::ReleaseTypeInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);

  /** \brief Get ONNXType of an ::OrtValue
   *
   * \param[in] value
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);

  /// @}
  /// \name OrtMemoryInfo
  /// @{

  /** \brief Create an ::OrtMemoryInfo
   *
   * \param[in] name
   * \param[in] type
   * \param[in] id
   * \param[in] mem_type
   * \param[out] out Newly created ::OrtMemoryInfo. Must be freed with OrtAPi::ReleaseMemoryInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateMemoryInfo, _In_ const char* name, enum OrtAllocatorType type, int id,
                  enum OrtMemType mem_type, _Outptr_ OrtMemoryInfo** out);

  /** \brief Create an ::OrtMemoryInfo for CPU memory
   *
   * Special case version of OrtApi::CreateMemoryInfo for CPU based memory. Same as using OrtApi::CreateMemoryInfo with name = "Cpu" and id = 0.
   *
   * \param[in] type
   * \param[in] mem_type
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type,
                  _Outptr_ OrtMemoryInfo** out);

  /** \brief Compare ::OrtMemoryInfo objects for equality
   *
   * Compares all settings of each ::OrtMemoryInfo for equality
   *
   * \param[in] info1
   * \param[in] info2
   * \param[out] out Set to 0 if equal, -1 if not equal
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out);

  /** \brief Get name from ::OrtMemoryInfo
   *
   * \param[in] ptr
   * \param[out] out Writes null terminated string to this pointer. Do NOT free the returned pointer. It is valid for the lifetime of the ::OrtMemoryInfo
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);

  /** \brief Get the id from ::OrtMemoryInfo
   */
  ORT_API2_STATUS(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);

  /** \brief Get the ::OrtMemType from ::OrtMemoryInfo
   */
  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);

  /** \brief Get the ::OrtAllocatorType from ::OrtMemoryInfo
   */
  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

  /// @}
  /// \name OrtAllocator
  /// @{

  /// \brief Calls OrtAllocator::Alloc function
  ORT_API2_STATUS(AllocatorAlloc, _Inout_ OrtAllocator* ort_allocator, size_t size, _Outptr_ void** out);
  /// \brief Calls OrtAllocator::Free function
  ORT_API2_STATUS(AllocatorFree, _Inout_ OrtAllocator* ort_allocator, void* p);
  /// \brief Calls OrtAllocator::Info function
  ORT_API2_STATUS(AllocatorGetInfo, _In_ const OrtAllocator* ort_allocator, _Outptr_ const struct OrtMemoryInfo** out);

  /** \brief Get the default allocator
   *
   * The default allocator is a CPU based, non-arena. Always returns the same pointer to the same default allocator.
   *
   * \param[out] out Returned value should NOT be freed
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);

  /// @}
  /// \name OrtSessionOptions
  /// @{

  /** \brief Override session symbolic dimensions
   *
   * Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
   * optimizations that can take advantage of fixed values (such as memory planning, etc)
   *
   * \param[in] options
   * \param[in] dim_denotation
   * \param[in] dim_value
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* dim_denotation,
                  _In_ int64_t dim_value);

  /// @}
  /// \name OrtValue
  /// @{

  /* Internal information (not seen in Doxygen)
   *
   * APIs to support non-tensor types - map and sequence.
   * Currently only the following types are supported
   * Note: the following types should be kept in sync with data_types.h
   * Map types
   * =========
   * std::map<std::string, std::string>
   * std::map<std::string, int64_t>
   * std::map<std::string, float>
   * std::map<std::string, double>
   * std::map<int64_t, std::string>
   * std::map<int64_t, int64_t>
   * std::map<int64_t, float>
   * std::map<int64_t, double>
   *
   * Sequence types
   * ==============
   * std::vector<std::string>
   * std::vector<int64_t>
   * std::vector<float>
   * std::vector<double>
   * std::vector<std::map<std::string, float>>
   * std::vector<std::map<int64_t, float>
   */

  /** \brief Get non tensor data from an ::OrtValue
   *
   * If `value` is of type ONNX_TYPE_MAP, you need to retrieve the keys and values
   * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
   * If `value` is of type ONNX_TYPE_SEQUENCE, use index to retrieve the index'th element
   * of the sequence.
   *
   * \param[in] value
   * \param[in] index See above for usage based on `value` type
   * \param[in] allocator Allocator used to allocate ::OrtValue
   * \param[out] out Created ::OrtValue that holds the element requested. Must be freed with OrtApi::ReleaseValue
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ OrtValue** out);

  /** \brief Get non tensor value count from an ::OrtValue
   *
   * If `value` is of type ONNX_TYPE_MAP 2 will always be returned. For ONNX_TYPE_SEQUENCE
   * the number of elements in the sequence will be returned
   *
   * \param[in] value
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);

  /** \brief Create a map or sequence ::OrtValue
   *
   * To construct a map (ONNX_TYPE_MAP), use num_values = 2 and `in` should be an array of 2 ::OrtValue%s
   * representing keys and values.<br>
   *
   * To construct a sequence (ONNX_TYPE_SEQUENCE), use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an array of N ::OrtValue%s.
   *
   * \param[in] in See above for details
   * \param[in] num_values
   * \param[in] value_type Must be either ONNX_TYPE_MAP or ONNX_TYPE_SEQUENCE
   * \param[out] out Newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                  enum ONNXType value_type, _Outptr_ OrtValue** out);

  /** \brief Create an opaque (custom user defined type) ::OrtValue
   *
   * Constructs an ::OrtValue that contains a value of non-standard type created for
   * experiments or while awaiting standardization. ::OrtValue in this case would contain
   * an internal representation of the Opaque type. Opaque types are distinguished from
   * each other by two strings 1) domain and 2) type name. The combination of the two
   * must be unique, so the type representation is properly identified internally. The combination
   * must be properly registered from within ORT at both compile/run time or by another API.
   *
   * To construct the ::OrtValue pass domain and type names, also a pointer to a data container
   * the type of which must be known to both ORT and the client program. That data container may or may
   * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
   * verification purposes.
   *
   * \param[in] domain_name Null terminated string of the domain name
   * \param[in] type_name Null terminated string of the type name
   * \param[in] data_container User pointer Data to populate ::OrtValue
   * \param[in] data_container_size Size in bytes of what `data_container` points to
   * \param[out] out Newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                  _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);

  /** \brief Get internal data from an opaque (custom user defined type) ::OrtValue
   *
   * Copies internal data from an opaque value into a user provided buffer
   *
   * \see OrtApi::CreateOpaqueValue
   *
   * \param[in] domain_name Null terminated string of the domain name
   * \param[in] type_name Null terminated string of the type name
   * \param[in] in The opaque ::OrtValue
   * \param[out] data_container Buffer to copy data into
   * \param[out] data_container_size Size in bytes of the buffer pointed to by data_container. Must match the size of the internal buffer.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in,
                  _Out_ void* data_container, size_t data_container_size);

  /// @}
  /// \name OrtKernelInfo
  /// Custom operator APIs.
  /// @{

  /** \brief Get a float stored as an attribute in the graph node
   *
   * \param[in] info ::OrtKernelInfo instance
   * \param[in] name Null terminated string of the name of the attribute
   * \param[out] out Pointer to memory where the attribute will be stored
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ float* out);

  /** \brief Fetch a 64-bit int stored as an attribute in the graph node
   *
   * \param[in] info ::OrtKernelInfo instance
   * \param[in] name Null terminated string of the name of the attribute
   * \param[out] out Pointer to memory where the attribute will be stored
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ int64_t* out);

  /** \brief Fetch a string stored as an attribute in the graph node
   *
   * If `out` is nullptr, the value of `size` is set to the true size of the string
   * attribute, and a success status is returned.
   *
   * If the `size` parameter is greater than or equal to the actual string attribute's size,
   * the value of `size` is set to the true size of the string attribute, the provided memory
   * is filled with the attribute's contents, and a success status is returned.
   *
   * If the `size` parameter is less than the actual string attribute's size and `out`
   * is not nullptr, the value of `size` is set to the true size of the string attribute
   * and a failure status is returned.)
   *
   * \param[in] info ::OrtKernelInfo instance
   * \param[in] name Null terminated string of the name of the attribute
   * \param[out] out Pointer to memory where the attribute will be stored
   * \param[in,out] size See above comments for details
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out,
                  _Inout_ size_t* size);

  /// @}
  /// \name OrtKernelContext
  /// Custom operator APIs.
  /// @{

  /** \brief Used for custom operators, get the input count of a kernel
   *
   * \see ::OrtCustomOp
   */
  ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);

  /** \brief Used for custom operators, get the output count of a kernel
   *
   * \see ::OrtCustomOp
   */
  ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);

  /** \brief Used for custom operators, get an input of a kernel
   *
   * The function attempts fetches the input of the kernel. If the input is optional
   * and not present, the function returns success and out is set to nullptr.
   *
   * \param[in] context ::OrtKernelContext instance
   * \param[in] input index. See KernelContext_GetInputCount for boundaries check.
   * \param[in, out] returns a ptr to OrtValue if the input is present
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index,
                  _Out_ const OrtValue** out);

  /** \brief Used for custom operators, get an output of a kernel
   *
   * The function attempts fetches the output of the kernel. If the output is optional
   * and not present, the function returns success and out is set to nullptr.
   *
   * \param[in] context ::OrtKernelContext instance
   * \param[in] output index. See KernelContext_GetOutputCount for boundaries check.
   * \param[in, out] returns a ptr to OrtValue if the output is present
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index,
                  _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);

  /// @}
  /// \name OrtEnv
  /// @{
  ORT_CLASS_RELEASE(Env);
  /// @}
  /// \name OrtStatus
  /// @{
  ORT_CLASS_RELEASE(Status);
  /// @}
  /// \name OrtMemoryInfo
  /// @{
  ORT_CLASS_RELEASE(MemoryInfo);
  /// @}
  /// \name OrtSession
  /// @{
  ORT_CLASS_RELEASE(Session);  // Don't call ReleaseSession from Dllmain (because session owns a thread pool)
  /// @}
  /// \name OrtValue
  /// @{
  ORT_CLASS_RELEASE(Value);
  /// @}
  /// \name OrtRunOptions
  /// @{
  ORT_CLASS_RELEASE(RunOptions);
  /// @}
  /// \name OrtTypeInfo
  /// @{
  ORT_CLASS_RELEASE(TypeInfo);
  /// @}
  /// \name OrtTensorTypeAndShapeInfo
  /// @{
  ORT_CLASS_RELEASE(TensorTypeAndShapeInfo);
  /// @}
  /// \name OrtSessionOptions
  /// @{
  ORT_CLASS_RELEASE(SessionOptions);
  /// @}
  /// \name OrtCustomOpDomain
  /// @{
  ORT_CLASS_RELEASE(CustomOpDomain);

  /// @}
  /// \name OrtTypeInfo
  /// @{

  /** \brief Get denotation from type information
   *
   * Augments ::OrtTypeInfo to return denotations on the type.
   *
   * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
   *
   * \param[in] type_info
   * \param[out] denotation Pointer to the null terminated denotation string is written to this pointer. This pointer is valid until the object is destroyed or the name is changed, do not free.
   * \param[out] len Length in bytes of the string returned in `denotation`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const char** const denotation,
                  _Out_ size_t* len);

  /** \brief Get detailed map information from an ::OrtTypeInfo
   *
   * This augments ::OrtTypeInfo to return an ::OrtMapTypeInfo when the type is a map.
   * The OrtMapTypeInfo has additional information about the map's key type and value type.
   *
   * This is used by WinML to support model reflection APIs.
   *
   * \param[out] type_info
   * \param[out] out A pointer to the ::OrtMapTypeInfo. Do not free this value. If type_info
   *             does not contain a map, this value will be set to nullptr.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtMapTypeInfo** out);

  /** \brief Cast ::OrtTypeInfo to an ::OrtSequenceTypeInfo
   *
   * This api augments ::OrtTypeInfo to return an ::OrtSequenceTypeInfo when the type is a sequence.
   * The ::OrtSequenceTypeInfo has additional information about the sequence's element type.
   *
   * This is used by WinML to support model reflection APIs.
   *
   * \param[in] type_info
   * \param[out] out A pointer to the OrtSequenceTypeInfo. Do not free this value. If type_info
   *             doesn not contain a sequence, this value will be set to nullptr.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

  /// @}
  /// \name OrtMapTypeInfo
  /// @{

  /** \brief Get key type from an ::OrtMapTypeInfo
   *
   * Key types are restricted to being scalar types.
   *
   * This is used by WinML to support model reflection APIs.
   *
   * \param[in] map_type_info
   * \param[out] out
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);

  /** \brief Get the value type from an ::OrtMapTypeInfo
   *
   * \param[in] map_type_info
   * \param[out] type_info
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

  /// @}
  /// \name OrtSequenceTypeInfo
  /// @{

  /** \brief Get element type from an ::OrtSequenceTypeInfo
   *
   * This is used by WinML to support model reflection APIs.
   *
   * \param[in] sequence_type_info
   * \param[out] type_info
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info,
                  _Outptr_ OrtTypeInfo** type_info);

  /// @}
  /// \name OrtMapTypeInfo
  /// @{
  ORT_CLASS_RELEASE(MapTypeInfo);
  /// @}
  /// \name OrtSequenceTypeInfo
  /// @{
  ORT_CLASS_RELEASE(SequenceTypeInfo);

  /// @}
  /// \name OrtSession
  /// @{

  /** \brief End profiling and return filename of the profile data
   *
   * Profiling is turned on through OrtApi::EnableProfiling
   *
   * \param[in] session
   * \param[in] allocator
   * \param[out] out Null terminated string of the filename, allocated using `allocator`. Must be freed using `allocator`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionEndProfiling, _In_ OrtSession* session, _Inout_ OrtAllocator* allocator, _Outptr_ char** out);

  /** \brief Get ::OrtModelMetadata from an ::OrtSession
   *
   * \param[in] session
   * \param[out] out Newly created ::OrtModelMetadata. Must be freed using OrtApi::ReleaseModelMetadata
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(SessionGetModelMetadata, _In_ const OrtSession* session, _Outptr_ OrtModelMetadata** out);

  /// @}
  /// \name OrtModelMetadata
  /// @{

  /** \brief Get `producer name` from an ::OrtModelMetadata
   *
   * \param[in] model_metadata
   * \param[in] allocator
   * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /** \brief Get `graph name` from an ::OrtModelMetadata
   *
   * \param[in] model_metadata
   * \param[in] allocator
   * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /** \brief Get `domain` from an ::OrtModelMetadata
   *
   * \param[in] model_metadata
   * \param[in] allocator
   * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);

  /** \brief Get `description` from an ::OrtModelMetadata
   *
   * \param[in] model_metadata
   * \param[in] allocator
   * \param[out] value Set to a null terminated string allocated using `allocator`. Must be freed using `allocator`
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   */
  ORT_API2_STATUS(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /** \brief Return data for a key in the custom metadata map in an ::OrtModelMetadata
   *
   * \param[in] model_metadata
   * \param[in] allocator
   * \p