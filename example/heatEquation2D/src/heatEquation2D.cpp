/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei
 * Bastrakov, Bernhard Manfred Gruber, Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "InitializeBufferKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"

#include <alpaka/mem/view/Traits.hpp>


#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif
#ifdef OPENPMD_ENABLED
#    include <openPMD/openPMD.hpp>
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef OPENPMD_ENABLED
struct OpenPMDOutput
{
private:
    openPMD::Series m_series;

    template<typename Buffer>
    static auto paddedMemoryExtent(Buffer const& buffer) -> alpaka::Vec<alpaka::Dim<Buffer>, alpaka::Idx<Buffer>>
    {
        // Initialize with logical extent.
        // It has the right number of entries, and also the correct entry
        // for the first (the slowest) dimension as that is not padded.
        auto result = alpaka::getExtents(buffer);
        auto const pitches = alpaka::getPitchesInBytes(buffer);

        auto extent_it = result.begin();
        auto extent_end = result.end();
        auto pitch_it = pitches.begin();
        auto pitch_end = pitches.end();

        auto previous_pitch = *pitch_it++;
        ++extent_it;

        for(; pitch_it != pitch_end; ++extent_it, ++pitch_it)
        {
            if(previous_pitch % *pitch_it != 0)
            {
                throw std::runtime_error("No specification of memory selection possible.");
            }
            *extent_it = previous_pitch / *pitch_it;
            previous_pitch = *pitch_it;
        }
        return result;
    }

    template<typename Vec>
    static auto asOpenPMDExtent(Vec const& vec) -> openPMD::Extent
    {
        return openPMD::Extent{vec.begin(), vec.end()};
    }

    template<typename Buffer>
    using AsUniquePtr = openPMD::UniquePtrWithLambda<
        std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<Buffer>().data())>>>;

public:
    void init()
    {
        m_series = openPMD::Series(
            "heat.%E",
            openPMD::Access::CREATE,
            "@../example/heatEquation2D/src/openpmd_config.toml");
        m_series.setMeshesPath("images");
    }

    template<typename HostBuffer, typename AccBuffer, typename DumpQueue, typename DevAcc>
    void writeIteration(
        openPMD::Iteration::IterationIndex_t step,
        HostBuffer& hostBuffer,
        AccBuffer& accBuffer,
        DumpQueue& dumpQueue,
        DevAcc& devAcc)
    {
        using value_t = double;

        openPMD::Iteration current_iteration = m_series.writeIterations()[step];
        current_iteration.setTime(1.0);
        openPMD::Mesh image = current_iteration.meshes["heat"];

        image.setAxisLabels({"x", "y"});
        image.setGridGlobalOffset({0., 0.});
        image.setGridSpacing(std::vector<double>{1., 1.});
        image.setGridUnitSI(1.0);
        image.setPosition(std::vector<double>{0.5, 0.5});
        image.setUnitDimension({{openPMD::UnitDimension::theta, 1.0}});
        image.setUnitSI(1.0);

        auto logical_extents = alpaka::getExtents(accBuffer);
        image.resetDataset({openPMD::determineDatatype<value_t>(), asOpenPMDExtent(logical_extents)});

        constexpr bool direct_gpu = false;
        if constexpr(direct_gpu)
        {
            auto physicalExtent = paddedMemoryExtent(accBuffer);
            image.prepareLoadStore()
                .withRawPtr(accBuffer.data())
                // device extent with padding
                .memorySelection({{0, 0}, asOpenPMDExtent(physicalExtent)})
                .store(openPMD::EnqueuePolicy::Defer);
        }
        else
        {
            alpaka::memcpy(dumpQueue, hostBuffer, accBuffer);
            alpaka::wait(dumpQueue);
            image.storeChunkRaw(hostBuffer.data(), {0, 0}, asOpenPMDExtent(logical_extents));
        }

        current_iteration.close();
    }

    void close()
    {
        m_series.close();
    }
};
#else
struct OpenPMDOutput
{
    void init()
    {
    }

    template<typename... Args>
    void writeIteration(Args&&...)
    {
    }

    void close()
    {
    }
};
#endif

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any
//! available accelerator. Instead, a single accelerator is selected once from
//! the active accelerators and the kernels are executed with the selected
//! accelerator only. If you use the example as the starting point for your
//! project, you can rename the example() function to main() and move the
//! accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Set Dim and Idx type
    using Dim = alpaka::DimInt<2u>;
    using Idx = uint32_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Select specific devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    // get suitable device for this Acc
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // simulation defines
    // {Y, X}
    constexpr alpaka::Vec<Dim, Idx> numNodes{64, 64};
    constexpr alpaka::Vec<Dim, Idx> haloSize{2, 2};
    constexpr alpaka::Vec<Dim, Idx> extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000;
    constexpr double tMax = 0.1;

    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    double r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
    if(r > 1.)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    auto uBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    // Accelerator buffers
    auto uCurrBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent);
    auto uNextBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent);

    // Create queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc dumpQueue{devAcc};
    QueueAcc computeQueue{devAcc};

    // Set buffer to initial conditions
    InitializeBufferKernel initBufferKernel;
    // Define a workdiv for the given problem
    constexpr alpaka::Vec<Dim, Idx> elemPerThread{1, 1};

    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elemPerThread};

    auto workDivExtent = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        initBufferKernel,
        alpaka::experimental::getMdSpan(uCurrBufAcc),
        dx,
        dy);

    alpaka::exec<Acc>(
        computeQueue,
        workDivExtent,
        initBufferKernel,
        alpaka::experimental::getMdSpan(uCurrBufAcc),
        dx,
        dy);


    // Appropriate chunk size to split your problem for your Acc
    constexpr Idx xSize = 16u;
    constexpr Idx ySize = 16u;
    constexpr alpaka::Vec<Dim, Idx> chunkSize{ySize, xSize};
    constexpr auto sharedMemSize = (ySize + haloSize[0]) * (xSize + haloSize[1]);

    constexpr alpaka::Vec<Dim, Idx> numChunks{
        alpaka::core::divCeil(numNodes[0], chunkSize[0]),
        alpaka::core::divCeil(numNodes[1], chunkSize[1]),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");

    StencilKernel<sharedMemSize> stencilKernel;

    // Get max threads that can be run in a block for this kernel
    auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(
        devAcc,
        stencilKernel,
        alpaka::experimental::getMdSpan(uCurrBufAcc),
        alpaka::experimental::getMdSpan(uNextBufAcc),
        chunkSize,
        dx,
        dy,
        dt);
    auto const maxThreadsPerBlock = kernelFunctionAttributes.maxThreadsPerBlock;

    auto const threadsPerBlock
        = maxThreadsPerBlock < chunkSize.prod() ? alpaka::Vec<Dim, Idx>{maxThreadsPerBlock, 1} : chunkSize;

    alpaka::WorkDivMembers<Dim, Idx> workDiv{numChunks, threadsPerBlock, elemPerThread};

    OpenPMDOutput openPMDOutput;
    openPMDOutput.init();

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        alpaka::exec<Acc>(
            computeQueue,
            workDiv,
            stencilKernel,
            alpaka::experimental::getMdSpan(uCurrBufAcc),
            alpaka::experimental::getMdSpan(uNextBufAcc),
            chunkSize,
            dx,
            dy,
            dt);

        // Apply boundaries
        applyBoundaries<Acc>(
            workDivExtent,
            computeQueue,
            alpaka::experimental::getMdSpan(uNextBufAcc),
            step,
            dx,
            dy,
            dt);
        if((step - 1) % 10 == 0)
        {
            openPMDOutput.writeIteration(step, uBufHost, uCurrBufAcc, dumpQueue, devAcc);
        }

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            alpaka::wait(computeQueue);
            alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            alpaka::wait(dumpQueue);
            writeImage(step - 1, uBufHost);
        }
#endif

        // So we just swap next and curr (shallow copy)
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    openPMDOutput.close();

    // Copy device -> host
    alpaka::wait(computeQueue);
    alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost, dx, dy, tMax);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use
    // the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks,
    //   TagCpuTbbBlocks, TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks,
    //   TagCpuThreads, TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
