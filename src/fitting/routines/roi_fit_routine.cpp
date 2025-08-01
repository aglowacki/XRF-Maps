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

/// Initial Author <2016>: Arthur Glowacki



#include "roi_fit_routine.h"

namespace fitting
{
namespace routines
{

// --------------------------------------------------------------------------------------------------------------------

template<typename T_real>
ROI_Fit_Routine<T_real>::ROI_Fit_Routine() : Base_Fit_Routine<T_real>()
{

}

// --------------------------------------------------------------------------------------------------------------------

template<typename T_real>
ROI_Fit_Routine<T_real>::~ROI_Fit_Routine()
{

}

// --------------------------------------------------------------------------------------------------------------------

template<typename T_real>
optimizers::OPTIMIZER_OUTCOME ROI_Fit_Routine<T_real>::fit_spectra(const models::Base_Model<T_real>* const model,
                                                            const Spectra<T_real>* const spectra,
                                                            const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                                            std::unordered_map<std::string, T_real>& out_counts)
 {    
    Fit_Parameters<T_real> fitp = model->fit_parameters();
    unsigned int n_mca_channels = spectra->size();

    T_real energy_offset = fitp.value(STR_ENERGY_OFFSET);
    T_real energy_slope = fitp.value(STR_ENERGY_SLOPE);
    for(const auto& e_itr : *elements_to_fit)
    {
        unsigned int left_roi = 0;
        unsigned int right_roi = 0;
        Fit_Element_Map<T_real>* element = e_itr.second;
        if (element != nullptr)
        {
            left_roi = static_cast<unsigned int>(std::round(((element->center() - element->width()) - energy_offset) / energy_slope));
            right_roi = static_cast<unsigned int>(std::round(((element->center() + element->width()) - energy_offset) / energy_slope));

            if (right_roi >= n_mca_channels)
            {
                right_roi = n_mca_channels - 2;
            }
            if (left_roi > right_roi)
            {
                left_roi = right_roi - 1;
            }

            size_t spec_size = (right_roi - left_roi) + 1;
            out_counts[e_itr.first] = spectra->segment(left_roi, spec_size).sum();
        }
    }
    return optimizers::OPTIMIZER_OUTCOME::CONVERGED;
}

// --------------------------------------------------------------------------------------------------------------------

template<typename T_real>
void ROI_Fit_Routine<T_real>::initialize(models::Base_Model<T_real>* const model,
                                 const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                 const struct Range energy_range,
                                 ArrayTr<T_real>* custom_background)
{
    //N/A
}

// --------------------------------------------------------------------------------------------------------------------

TEMPLATE_CLASS_DLL_EXPORT ROI_Fit_Routine<float>;
TEMPLATE_CLASS_DLL_EXPORT ROI_Fit_Routine<double>;

} //namespace routines
} //namespace fitting
