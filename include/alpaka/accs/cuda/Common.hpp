/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>              // Vec
#include <alpaka/core/IntegerSequence.hpp>  // integer_sequence

// cuda_runtime_api.h: CUDA Runtime API C-style interface that does not require compiling with nvcc.
// cuda_runtime.h: CUDA Runtime API  C++-style interface built on top of the C API.
//  It wraps some of the C API routines, using overloading, references and default arguments.
//  These wrappers can be used from C++ code and can be compiled with any C++ compiler.
//  The C++ API also has some CUDA-specific wrappers that wrap C API routines that deal with symbols, textures, and device functions.
//  These wrappers require the use of \p nvcc because they depend on code being generated by the compiler.
//  For example, the execution configuration syntax to invoke kernels is only available in source code compiled with nvcc.
#include <cuda_runtime.h>
// cuda.h: CUDA Driver API
//#include <cuda.h>

#include <array>                            // std::array
#include <type_traits>                      // std::enable_if
#include <utility>                          // std::forward
#include <iostream>                         // std::cerr
#include <string>                           // std::string, std::to_string
#include <stdexcept>                        // std::runtime_error
#include <cstddef>                          // std::size_t

#if (!defined(CUDART_VERSION) || (CUDART_VERSION < 7000))
    #error "CUDA version 7.0 or greater required!"
#endif

/*#if (!defined(CUDA_VERSION) || (CUDA_VERSION < 7000))
    #error "CUDA version 7.0 or greater required!"
#endif*/

namespace alpaka
{
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Applies the trait to all types and combines the result with &&.
        //! Multiple argument version.
        //-----------------------------------------------------------------------------
        template<
            template<typename, typename...> class TTrait,
            typename THead,
            typename... TTail>
        struct ApplyAllCombineAndInternal
        {
            enum
            {
                value = TTrait<THead>::value
                    && ApplyAllCombineAndInternal<TTrait, TTail...>::value
            };
        };
        //-----------------------------------------------------------------------------
        //! Applies the trait to all types and combines the result with &&.
        //! Single argument version.
        //-----------------------------------------------------------------------------
        template<
            template<typename, typename...> class TTrait,
            typename THead>
        struct ApplyAllCombineAndInternal<
            TTrait,
            THead>
        {
            enum
            {
                value = TTrait<THead>::value
            };
        };
        //-----------------------------------------------------------------------------
        //! Applies the trait to all types and combines the result with &&.
        //-----------------------------------------------------------------------------
        template<
            template<typename, typename...> class TTrait,
            typename... TApplicants>
        struct ApplyAllCombineAnd
        {
            enum
            {
                value = ApplyAllCombineAndInternal<
                    TTrait,
                    TApplicants...>::value
            };
        };
        //-----------------------------------------------------------------------------
        //! Applies the trait to all types and combines the result with &&.
        //! Zero argument version always returns true.
        //-----------------------------------------------------------------------------
        template<
            template<typename, typename...> class TTrait>
        struct ApplyAllCombineAnd<
            TTrait>
        {
            enum
            {
                value = true
            };
        };
    }
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                template <typename T>
                using IsConvertibleCudaError = std::is_convertible<T, cudaError_t>;

                //-----------------------------------------------------------------------------
                //! CUDA runtime error checking with log and exception, ignoring specific error values
                //-----------------------------------------------------------------------------
                template<
                    typename... TErrors,
                    typename = typename std::enable_if<alpaka::detail::ApplyAllCombineAnd<IsConvertibleCudaError, TErrors...>::value>::type>
                ALPAKA_FCT_HOST auto cudaRtCheckIgnore(
                    cudaError_t const & error,
                    char const * cmd,
                    char const * file,
                    int const & line,
                    TErrors && ... ignoredErrorCodes)
                -> void
                {
                    // Even if we get the error directly from the command, we have to reset the global error state by getting it.
                    cudaGetLastError();
                    if(error != cudaSuccess)
                    {
                        // If the error code is not one of the ignored ones.
                        std::array<cudaError_t, sizeof...(ignoredErrorCodes)> const aIgnoredErrorCodes{std::forward<TErrors>(ignoredErrorCodes)...};
                        if(std::find(aIgnoredErrorCodes.cbegin(), aIgnoredErrorCodes.cend(), error) == aIgnoredErrorCodes.cend())
                        {
                            std::string const sError(std::string(file) + "(" + std::to_string(line) + ") '" + std::string(cmd) + "' returned error: '" + std::string(cudaGetErrorString(error)) + "' (possibly from a previous CUDA call)!");
                            std::cerr << sError << std::endl;
                            ALPAKA_DEBUG_BREAK;
                            throw std::runtime_error(sError);
                        }
                    }
                }
            }
        }
    }
}

#if BOOST_COMP_MSVC
    //-----------------------------------------------------------------------------
    //! CUDA runtime error checking with log and exception, ignoring specific error values
    //-----------------------------------------------------------------------------
    #define ALPAKA_CUDA_RT_CHECK_IGNORE(cmd, ...)\
        ::alpaka::accs::cuda::detail::cudaRtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#else
    //-----------------------------------------------------------------------------
    //! CUDA runtime error checking with log and exception, ignoring specific error values
    //-----------------------------------------------------------------------------
    #define ALPAKA_CUDA_RT_CHECK_IGNORE(cmd, ...)\
        ::alpaka::accs::cuda::detail::cudaRtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

//-----------------------------------------------------------------------------
//! CUDA runtime error checking with log and exception.
//-----------------------------------------------------------------------------
#define ALPAKA_CUDA_RT_CHECK(cmd)\
    ALPAKA_CUDA_RT_CHECK_IGNORE(cmd)

/*namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! CUDA runtime error checking with log and exception, ignoring specific error values
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto cudaDrvCheck(
                    cudaError_t const & error,
                    char const * cmd,
                    char const * file,
                    int const & line)
                -> void
                {
                    // Even if we get the error directly from the command, we have to reset the global error state by getting it.
                    if(error != CUDA_SUCCESS)
                    {
                        std::string const sError(std::to_string(file) + "(" + std::to_string(line) + ") '" + std::to_string(cmd) + "' returned error: '" + std::to_string(error) + "' (possibly from a previous CUDA call)!");
                        std::cerr << sError << std::endl;
                        ALPAKA_DEBUG_BREAK;
                        throw std::runtime_error(sError);
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
//! CUDA driver error checking with log and exception.
//-----------------------------------------------------------------------------
#define ALPAKA_CUDA_DRV_CHECK(cmd)\
    ::alpaka::accs::cuda::detail::cudaDrvCheck(cmd, #cmd, __FILE__, __LINE__)*/


//-----------------------------------------------------------------------------
//! CUDA vector_types.h trait specializations.
//-----------------------------------------------------------------------------
namespace alpaka
{
    namespace traits
    {
        namespace cuda
        {
            //#############################################################################
            //! The CUDA vectors 1D dimension get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct IsCudaBuiltinType :
                std::integral_constant<
                    bool,
                    std::is_same<T, char1>::value
                    || std::is_same<T, double1>::value
                    || std::is_same<T, float1>::value
                    || std::is_same<T, int1>::value
                    || std::is_same<T, long1>::value
                    || std::is_same<T, longlong1>::value
                    || std::is_same<T, short1>::value
                    || std::is_same<T, uchar1>::value
                    || std::is_same<T, uint1>::value
                    || std::is_same<T, ulong1>::value
                    || std::is_same<T, ulonglong1>::value
                    || std::is_same<T, ushort1>::value
                    || std::is_same<T, char2>::value
                    || std::is_same<T, double2>::value
                    || std::is_same<T, float2>::value
                    || std::is_same<T, int2>::value
                    || std::is_same<T, long2>::value
                    || std::is_same<T, longlong2>::value
                    || std::is_same<T, short2>::value
                    || std::is_same<T, uchar2>::value
                    || std::is_same<T, uint2>::value
                    || std::is_same<T, ulong2>::value
                    || std::is_same<T, ulonglong2>::value
                    || std::is_same<T, ushort2>::value
                    || std::is_same<T, char3>::value
                    || std::is_same<T, dim3>::value
                    || std::is_same<T, double3>::value
                    || std::is_same<T, float3>::value
                    || std::is_same<T, int3>::value
                    || std::is_same<T, long3>::value
                    || std::is_same<T, longlong3>::value
                    || std::is_same<T, short3>::value
                    || std::is_same<T, uchar3>::value
                    || std::is_same<T, uint3>::value
                    || std::is_same<T, ulong3>::value
                    || std::is_same<T, ulonglong3>::value
                    || std::is_same<T, ushort3>::value
                    || std::is_same<T, char4>::value
                    || std::is_same<T, double4>::value
                    || std::is_same<T, float4>::value
                    || std::is_same<T, int4>::value
                    || std::is_same<T, long4>::value
                    || std::is_same<T, longlong4>::value
                    || std::is_same<T, short4>::value
                    || std::is_same<T, uchar4>::value
                    || std::is_same<T, uint4>::value
                    || std::is_same<T, ulong4>::value
                    || std::is_same<T, ulonglong4>::value
                    || std::is_same<T, ushort4>::value>
            {};
        }
        namespace dim
        {
            //#############################################################################
            //! The CUDA vectors 1D dimension get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char1>::value
                    || std::is_same<T, double1>::value
                    || std::is_same<T, float1>::value
                    || std::is_same<T, int1>::value
                    || std::is_same<T, long1>::value
                    || std::is_same<T, longlong1>::value
                    || std::is_same<T, short1>::value
                    || std::is_same<T, uchar1>::value
                    || std::is_same<T, uint1>::value
                    || std::is_same<T, ulong1>::value
                    || std::is_same<T, ulonglong1>::value
                    || std::is_same<T, ushort1>::value>::type>
            {
                using type = alpaka::dim::Dim1;
            };
            //#############################################################################
            //! The CUDA vectors 2D dimension get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char2>::value
                    || std::is_same<T, double2>::value
                    || std::is_same<T, float2>::value
                    || std::is_same<T, int2>::value
                    || std::is_same<T, long2>::value
                    || std::is_same<T, longlong2>::value
                    || std::is_same<T, short2>::value
                    || std::is_same<T, uchar2>::value
                    || std::is_same<T, uint2>::value
                    || std::is_same<T, ulong2>::value
                    || std::is_same<T, ulonglong2>::value
                    || std::is_same<T, ushort2>::value>::type>
            {
                using type = alpaka::dim::Dim2;
            };
            //#############################################################################
            //! The CUDA vectors 3D dimension get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char3>::value
                    || std::is_same<T, dim3>::value
                    || std::is_same<T, double3>::value
                    || std::is_same<T, float3>::value
                    || std::is_same<T, int3>::value
                    || std::is_same<T, long3>::value
                    || std::is_same<T, longlong3>::value
                    || std::is_same<T, short3>::value
                    || std::is_same<T, uchar3>::value
                    || std::is_same<T, uint3>::value
                    || std::is_same<T, ulong3>::value
                    || std::is_same<T, ulonglong3>::value
                    || std::is_same<T, ushort3>::value>::type>
            {
                using type = alpaka::dim::Dim3;
            };
            //#############################################################################
            //! The CUDA vectors 4D dimension get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char4>::value
                    || std::is_same<T, double4>::value
                    || std::is_same<T, float4>::value
                    || std::is_same<T, int4>::value
                    || std::is_same<T, long4>::value
                    || std::is_same<T, longlong4>::value
                    || std::is_same<T, short4>::value
                    || std::is_same<T, uchar4>::value
                    || std::is_same<T, uint4>::value
                    || std::is_same<T, ulong4>::value
                    || std::is_same<T, ulonglong4>::value
                    || std::is_same<T, ushort4>::value>::type>
            {
                using type = alpaka::dim::Dim4;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The CUDA vectors extent get trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct GetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value - 0u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 1)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    TExtents const & extents)
                -> decltype(extents.x)
                {
                    return extents.x;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent get trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct GetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value-1u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 2)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    TExtents const & extents)
                -> decltype(extents.y)
                {
                    return extents.y;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent get trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct GetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value-2u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 3)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    TExtents const & extents)
                -> decltype(extents.z)
                {
                    return extents.z;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent get trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct GetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value-3u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 4)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    TExtents const & extents)
                -> decltype(extents.w)
                {
                    return extents.w;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent set trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct SetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value - 0u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 1)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    TExtents const & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents.x = extent;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent set trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct SetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value - 1u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 2)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    TExtents const & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents.y = extent;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent set trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct SetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value - 2u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 3)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    TExtents const & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents.z = extent;
                }
            };
            //#############################################################################
            //! The CUDA vectors extent set trait specialization.
            //#############################################################################
            template<
                typename TExtents>
            struct SetExtent<
                alpaka::dim::Dim<alpaka::dim::DimT<TExtents>::value - 3u>,
                TExtents,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TExtents>::value
                    && (alpaka::dim::DimT<TExtents>::value >= 4)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setExtent(
                    TExtents const & extents,
                    TVal2 const & extent)
                -> void
                {
                    extents.w = extent;
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The CUDA vectors offset get trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct GetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 0u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 1)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.x)
                {
                    return offsets.x;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset get trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct GetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 1u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 2)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.y)
                {
                    return offsets.y;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset get trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct GetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 2u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 3)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.z)
                {
                    return offsets.z;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset get trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct GetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 3u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 4)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.w)
                {
                    return offsets.w;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset set trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct SetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 0u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 1)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets.x = offset;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset set trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct SetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 1u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 2)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets.y = offset;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset set trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct SetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 2u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 3)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets.z = offset;
                }
            };
            //#############################################################################
            //! The CUDA vectors offset set trait specialization.
            //#############################################################################
            template<
                typename TOffsets>
            struct SetOffset<
                alpaka::dim::Dim<alpaka::dim::DimT<TOffsets>::value - 3u>,
                TOffsets,
                typename std::enable_if<
                    cuda::IsCudaBuiltinType<TOffsets>::value
                    && (alpaka::dim::DimT<TOffsets>::value >= 4)>::type>
            {
                template<
                    typename TVal2>
                ALPAKA_FCT_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TVal2 const & offset)
                -> void
                {
                    offsets.w = offset;
                }
            };
        }
    }
}
