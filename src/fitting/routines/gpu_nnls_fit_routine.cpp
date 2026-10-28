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


#include "gpu_nnls_fit_routine.h"

#include <cmath>

#ifdef _BUILD_WITH_METAL
#include "fitting/routines/gpu/metal_nnls_backend.h"
#endif

namespace fitting
{
namespace routines
{

// ----------------------------------------------------------------------------

template<typename T_real>
GPU_NNLS_Fit_Routine<T_real>::GPU_NNLS_Fit_Routine() : NNLS_Fit_Routine<T_real>()
{
    _backend = nullptr;
    _gpu_ready = false;
    _max_batch_size = 4096;
}

// ----------------------------------------------------------------------------

template<typename T_real>
GPU_NNLS_Fit_Routine<T_real>::GPU_NNLS_Fit_Routine(size_t max_iter) : NNLS_Fit_Routine<T_real>(max_iter)
{
    _backend = nullptr;
    _gpu_ready = false;
    _max_batch_size = 4096;
}

// ----------------------------------------------------------------------------

template<typename T_real>
GPU_NNLS_Fit_Routine<T_real>::~GPU_NNLS_Fit_Routine()
{
#ifdef _BUILD_WITH_METAL
    delete _backend;
#endif
    _backend = nullptr;
    _gpu_ready = false;
}

// ----------------------------------------------------------------------------

template<typename T_real>
std::string GPU_NNLS_Fit_Routine<T_real>::gpu_device_name() const
{
#ifdef _BUILD_WITH_METAL
    if (_backend != nullptr)
    {
        return _backend->device_name();
    }
#endif
    return "";
}

// ----------------------------------------------------------------------------

template<typename T_real>
int GPU_NNLS_Fit_Routine<T_real>::preferred_batch_size() const
{
#ifdef _BUILD_WITH_METAL
    if (_gpu_ready && _backend != nullptr)
    {
        int rec = _backend->recommended_batch_size();
        return (rec < _max_batch_size) ? rec : _max_batch_size;
    }
#endif
    return 1;
}

// ----------------------------------------------------------------------------

template<typename T_real>
bool GPU_NNLS_Fit_Routine<T_real>::_setup_gpu()
{
    _gpu_ready = false;

#ifdef _BUILD_WITH_METAL

    const int m = (int)this->_fitmatrix.rows();
    const int n = (int)this->_fitmatrix.cols();

    if (m <= 0 || n <= 0)
    {
        logW << "GPU NNLS: fit matrix is empty, using CPU solver\n";
        return false;
    }

    if (n > gpu::METAL_NNLS_MAX_COMPONENTS)
    {
        logW << "GPU NNLS: " << n << " components exceeds the device limit of "
             << gpu::METAL_NNLS_MAX_COMPONENTS << ", using CPU solver\n";
        return false;
    }

    if (false == gpu::Metal_NNLS_Backend::is_available())
    {
        logW << "GPU NNLS: no Metal device available, using CPU solver\n";
        return false;
    }

    if (_backend == nullptr)
    {
        _backend = new gpu::Metal_NNLS_Backend();
    }

    if (false == _backend->initialize())
    {
        logW << "GPU NNLS: " << _backend->last_error() << ", using CPU solver\n";
        return false;
    }

    // Hand the backend the fit matrix in double, column major, whatever T_real is.
    // The scaling and the Gram matrix are accumulated in double on the way in; see
    // Metal_NNLS_Backend::set_fit_matrix.
    std::vector<double> host_a((size_t)m * (size_t)n);
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            host_a[(size_t)j * (size_t)m + (size_t)i] = (double)this->_fitmatrix(i, j);
        }
    }

    if (false == _backend->set_fit_matrix(host_a.data(), m, n))
    {
        logW << "GPU NNLS: " << _backend->last_error() << ", using CPU solver\n";
        return false;
    }

    logI << "GPU NNLS: using " << _backend->device_name() << " for " << m
         << " channels x " << n << " components\n";

    _gpu_ready = true;

#else

    logI << "GPU NNLS: built without Metal support, using CPU solver\n";

#endif

    return _gpu_ready;
}

// ----------------------------------------------------------------------------

template<typename T_real>
void GPU_NNLS_Fit_Routine<T_real>::initialize(models::Base_Model<T_real>* const model,
                                              const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                              const struct Range energy_range,
                                              ArrayTr<T_real>* custom_background)
{
    NNLS_Fit_Routine<T_real>::initialize(model, elements_to_fit, energy_range, custom_background);
    _setup_gpu();
}

// ----------------------------------------------------------------------------

template<typename T_real>
void GPU_NNLS_Fit_Routine<T_real>::initialize_mp(models::Base_Model<T_real>* const model,
                                                 const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                                 const struct Range energy_range)
{
    NNLS_Fit_Routine<T_real>::initialize_mp(model, elements_to_fit, energy_range);
    _setup_gpu();
}

// ----------------------------------------------------------------------------

template<typename T_real>
void GPU_NNLS_Fit_Routine<T_real>::fit_spectra_batch(const models::Base_Model<T_real>* const model,
                                                     const std::vector<const Spectra<T_real>*>& spectra_batch,
                                                     const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                                     std::vector<std::unordered_map<std::string, T_real>>& out_counts)
{
    const size_t n_in = spectra_batch.size();

    out_counts.clear();
    out_counts.resize(n_in);

    if (n_in == 0)
    {
        return;
    }

    if (false == _gpu_ready)
    {
        // Nothing to batch onto, so fall back to the inherited CPU solve. Identical
        // results to selecting the plain NNLS routine.
        for (size_t s = 0; s < n_in; s++)
        {
            if (spectra_batch[s] == nullptr || spectra_batch[s]->size() == 0)
            {
                continue;
            }
            NNLS_Fit_Routine<T_real>::fit_spectra(model, spectra_batch[s], elements_to_fit, out_counts[s]);
        }
        return;
    }

#ifdef _BUILD_WITH_METAL

    const int m = this->_energy_range.count();
    const int n = (int)this->_fitmatrix.cols();

    Fit_Parameters<T_real> fit_params = model->fit_parameters();

    // A custom background does not depend on the spectrum, so resolve it once.
    const bool has_custom_bkg = (this->_custom_background != nullptr);
    const bool has_snip = (false == has_custom_bkg) && fit_params.contains(STR_SNIP_WIDTH);

    ArrayTr<T_real> custom_bkg;
    if (has_custom_bkg)
    {
        custom_bkg = this->_custom_background->segment(this->_energy_range.min, this->_energy_range.count());
    }

    // Device input: one background subtracted, clamped row per usable spectrum.
    // Rows are packed, so keep a map back to the caller's indices.
    std::vector<size_t> row_to_src;
    row_to_src.reserve(n_in);
    std::vector<float> host_b;
    host_b.reserve(n_in * (size_t)m);

    ArrayTr<T_real> sum_background = ArrayTr<T_real>::Zero(m);

    for (size_t s = 0; s < n_in; s++)
    {
        const Spectra<T_real>* const spectra = spectra_batch[s];
        if (spectra == nullptr || spectra->size() == 0)
        {
            continue;
        }

        ArrayTr<T_real> background;
        if (has_custom_bkg)
        {
            background = custom_bkg;
        }
        else if (has_snip)
        {
            ArrayTr<T_real> bkg = snip_background<T_real>(spectra,
                                                          fit_params.value(STR_ENERGY_OFFSET),
                                                          fit_params.value(STR_ENERGY_SLOPE),
                                                          fit_params.value(STR_ENERGY_QUADRATIC),
                                                          fit_params.value(STR_SNIP_WIDTH),
                                                          this->_energy_range.min,
                                                          this->_energy_range.max);
            background = bkg.segment(this->_energy_range.min, this->_energy_range.count());
        }
        else
        {
            background.setZero(m);
        }

        ArrayTr<T_real> sub = spectra->segment(this->_energy_range.min, this->_energy_range.count());
        sub -= background;
        sub = sub.unaryExpr([](T_real v) { return v > 0.0 ? v : (T_real)0.0; });

        sum_background += background;

        for (int i = 0; i < m; i++)
        {
            host_b.push_back((float)sub[i]);
        }
        row_to_src.push_back(s);
    }

    const int n_rows = (int)row_to_src.size();
    if (n_rows == 0)
    {
        return;
    }

    std::vector<float> host_x((size_t)n_rows * (size_t)n, 0.f);
    std::vector<int>   host_iters((size_t)n_rows, 0);
    std::vector<float> host_npg((size_t)n_rows, 0.f);

    if (false == _backend->solve_batch(host_b.data(),
                                       n_rows,
                                       host_x.data(),
                                       host_iters.data(),
                                       host_npg.data(),
                                       (int)this->_max_iter,
                                       (float)1e-3))
    {
        logE << "GPU NNLS: " << _backend->last_error() << ", falling back to CPU for this batch\n";
        for (size_t s = 0; s < n_in; s++)
        {
            if (spectra_batch[s] == nullptr || spectra_batch[s]->size() == 0)
            {
                continue;
            }
            NNLS_Fit_Routine<T_real>::fit_spectra(model, spectra_batch[s], elements_to_fit, out_counts[s]);
        }
        return;
    }

    // The integrated fitted spectrum is a sum over the batch of (background + A x),
    // and escape_peak is linear in the spectrum it is given, so the whole batch needs
    // one matrix-vector product and one escape peak rather than a reconstruction per
    // spectrum. Summing the solution vectors first is what makes that possible.
    Eigen::Matrix<T_real, Eigen::Dynamic, 1> sum_x = Eigen::Matrix<T_real, Eigen::Dynamic, 1>::Zero(n);

    for (int r = 0; r < n_rows; r++)
    {
        const size_t src = row_to_src[(size_t)r];
        std::unordered_map<std::string, T_real>& counts = out_counts[src];
        const float* const xr = host_x.data() + (size_t)r * (size_t)n;

        for (const auto& itr : this->_element_models)
        {
            const int idx = this->_element_row_index[itr.first];
            const T_real v = (T_real)xr[idx];
            if (std::isfinite(v))
            {
                counts[itr.first] = v;
                sum_x(idx) += v;
            }
            else
            {
                // Same contract as the CPU routine: a non finite amplitude is reported
                // as zero and contributes nothing to the fitted model.
                counts[itr.first] = (T_real)0.0;
            }
        }

        counts[STR_NUM_ITR] = (T_real)host_iters[(size_t)r];
        counts[STR_RESIDUAL] = (T_real)host_npg[(size_t)r];
    }

    Spectra<T_real> batch_model(m);
    batch_model.setZero(m);
    {
        Eigen::Matrix<T_real, Eigen::Dynamic, 1> fitted = this->_fitmatrix * sum_x;
        for (int i = 0; i < m; i++)
        {
            batch_model[i] = sum_background[i] + fitted(i);
        }
    }

    if (model != nullptr
        && fit_params.contains(STR_SI_ESCAPE)
        && fit_params.at(STR_SI_ESCAPE).value > 0.0)
    {
        ArrayTr<T_real> energy = ArrayTr<T_real>::LinSpaced(this->_energy_range.count(),
                                                            this->_energy_range.min,
                                                            this->_energy_range.max);
        ArrayTr<T_real> ev = fit_params.value(STR_ENERGY_OFFSET)
                             + (energy * fit_params.value(STR_ENERGY_SLOPE))
                             + (pow(energy, (T_real)2.0) * fit_params.value(STR_ENERGY_QUADRATIC));

        batch_model += model->escape_peak(batch_model,
                                          ev,
                                          fit_params.value(STR_SI_ESCAPE),
                                          fit_params.value_or(STR_ESCAPE_LINEAR, (T_real)0.0));
    }

    {
        std::lock_guard<std::mutex> lock(this->_int_spec_mutex);

        if (false == this->_integrated_fitted_spectra.add(batch_model))
        {
            logW << "Could not add model spectra to _integrated_fitted_spectra\n";
        }
        if (false == this->_integrated_background.add(sum_background))
        {
            logW << "Could not add background spectra to _integrated_background\n";
        }
    }

#endif

}

// ----------------------------------------------------------------------------

TEMPLATE_CLASS_DLL_EXPORT GPU_NNLS_Fit_Routine<float>;
TEMPLATE_CLASS_DLL_EXPORT GPU_NNLS_Fit_Routine<double>;

} //namespace routines
} //namespace fitting
