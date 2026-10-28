/***
Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.

Copyright 2016. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***/

/// Initial Author <2026>: Arthur Glowacki



#ifndef GPU_NNLS_Fit_Routine_H
#define GPU_NNLS_Fit_Routine_H

#include <vector>

#include "fitting/routines/nnls_fit_routine.h"

namespace fitting
{
namespace routines
{

namespace gpu
{
class Metal_NNLS_Backend;
}

using namespace data_struct;

/**
 * @brief The GPU_NNLS_Fit_Routine class : non negative least squares fitting on the GPU.
 *
 * Same fit as NNLS_Fit_Routine, but solving a whole batch of spectra per GPU dispatch
 * instead of one spectrum at a time on the CPU. A single XRF spectrum is a tiny least
 * squares problem (a few thousand channels against a few tens of elements), far too
 * small to pay back a GPU dispatch on its own, so the parallelism has to come from
 * solving many pixels at once. The batched entry point is fit_spectra_batch; callers
 * that go through Base_Fit_Routine::fit_spectra one spectrum at a time get the
 * inherited CPU solve, which is the faster choice at that granularity.
 *
 * Two things differ numerically from the CPU routine, both documented at their
 * implementation:
 *
 *  - The device arithmetic is single precision, because the Metal Shading Language
 *    has no double type. T_real = double instances narrow on the way in and widen on
 *    the way out, so results agree with the CPU solver only to single precision.
 *
 *  - The solve runs on the normal equations with the fit matrix column scaled to unit
 *    norm, rather than on the fit matrix directly. Same minimiser, different path to
 *    it, so iteration counts and the last digits of the counts will not match the CPU
 *    routine exactly.
 *
 * When the GPU is unavailable, the build has no Metal support, or the problem is wider
 * than the device solver accepts, the routine reports supports_batch() == false and
 * every path falls through to the inherited CPU implementation. It is always safe to
 * select; it just stops being faster.
 */
template<typename T_real>
class DLL_EXPORT GPU_NNLS_Fit_Routine : public NNLS_Fit_Routine<T_real>
{
public:

    GPU_NNLS_Fit_Routine();

    GPU_NNLS_Fit_Routine(size_t max_iter);

    virtual ~GPU_NNLS_Fit_Routine();

    virtual std::string get_name() { return STR_FIT_GPU_NNLS; }

    virtual void initialize(models::Base_Model<T_real>* const model,
                            const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                            const struct Range energy_range,
                            ArrayTr<T_real>* custom_background);

    void initialize_mp(models::Base_Model<T_real>* const model,
                       const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                       const struct Range energy_range);

    /**
     * @brief supports_batch : True once the device solver is ready for the current
     *                         fit matrix. False means callers should use fit_spectra.
     */
    virtual bool supports_batch() const { return _gpu_ready; }

    virtual int preferred_batch_size() const;

    /**
     * @brief fit_spectra_batch : Solve a batch of spectra in one GPU dispatch.
     *
     * out_counts is resized to match spectra_batch. Entries whose spectrum pointer is
     * null, or whose spectrum is empty, come back as an empty map so the caller can
     * zero that pixel. Thread safe with respect to the integrated spectra accumulators,
     * but not safe to call concurrently on the same instance: the device backend owns
     * a single command queue.
     */
    virtual void fit_spectra_batch(const models::Base_Model<T_real>* const model,
                                   const std::vector<const Spectra<T_real>*>& spectra_batch,
                                   const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                   std::vector<std::unordered_map<std::string, T_real>>& out_counts);

    /**
     * @brief set_max_batch_size : Cap the spectra per dispatch. Mostly a memory knob;
     *                             the staging buffers are batch * channels floats.
     */
    void set_max_batch_size(int n) { if (n > 0) { _max_batch_size = n; } }

    /**
     * @brief gpu_device_name : Name of the device in use, or an empty string when the
     *                          routine is running on the CPU fallback.
     */
    std::string gpu_device_name() const;

protected:

    /**
     * @brief _setup_gpu : Bring up the device and upload the fit matrix. Returns false,
     *                     having logged why, whenever the CPU fallback should be used.
     */
    bool _setup_gpu();

    gpu::Metal_NNLS_Backend* _backend;

    bool _gpu_ready;

    int _max_batch_size;

};

} //namespace routines

} //namespace fitting

#endif // GPU_NNLS_Fit_Routine_H
