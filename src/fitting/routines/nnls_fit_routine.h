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



#ifndef NNLS_Fit_Routine_H
#define NNLS_Fit_Routine_H

#include "fitting/routines/matrix_optimized_fit_routine.h"

#include "support/nnls/nnls.hpp"

namespace fitting
{
namespace routines
{

using namespace data_struct;

template<typename T_real>
class DLL_EXPORT NNLS_Fit_Routine : public Matrix_Optimized_Fit_Routine <T_real>
{
public:

    NNLS_Fit_Routine();

    NNLS_Fit_Routine(size_t max_iter);

	virtual ~NNLS_Fit_Routine();

    virtual OPTIMIZER_OUTCOME fit_spectra(const models::Base_Model<T_real>* const model,
                                        const Spectra<T_real>* const spectra,
                                        const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                        std::unordered_map<std::string, T_real>& out_counts);

    // similar to fit_spectra but want to return model instead of counts
    void fit_spectrum_model(const Spectra<T_real>* const spectra,
                            const ArrayTr<T_real>* const background,
                            const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                            Spectra<T_real>* spectra_model);

    virtual std::string get_name() { return STR_FIT_NNLS; }

    virtual void initialize(models::Base_Model<T_real>* const model,
                            const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                            const struct Range energy_range,
                            ArrayTr<T_real>* custom_background);

    void initialize_mp(models::Base_Model<T_real>* const model,
                        const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                        const struct Range energy_range);

protected:

    void _generate_fitmatrix();

    size_t _max_iter;

private:

    Eigen::Matrix<T_real, Eigen::Dynamic, Eigen::Dynamic> _fitmatrix;

    std::unordered_map<std::string, int> _element_row_index;

};

} //namespace routines

} //namespace fitting

#endif // NNLS_Fit_Routine_H
