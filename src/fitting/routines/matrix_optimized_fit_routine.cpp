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


#include "matrix_optimized_fit_routine.h"

//debug
#include <iostream>

namespace fitting
{
namespace routines
{

// ----------------------------------------------------------------------------

Matrix_Optimized_Fit_Routine::Matrix_Optimized_Fit_Routine() : Param_Optimized_Fit_Routine()
{

}

// ----------------------------------------------------------------------------

Matrix_Optimized_Fit_Routine::~Matrix_Optimized_Fit_Routine()
{

    std::cout<<"******** destroy element models *******"<<std::endl;
    _element_models.clear();

}

// --------------------------------------------------------------------------------------------------------------------

Spectra Matrix_Optimized_Fit_Routine::model_spectrum(const Fit_Parameters * const fit_params,
                                                     const Spectra * const spectra,
                                                     const Detector * const detector,
                                                     const Fit_Element_Map_Dict * const elements_to_fit,
                                                     const struct Range energy_range)
{
    Spectra spectra_model(energy_range.count());

//    valarray<real_t> energy((real_t)0.0, energy_range.count());
//    real_t e_val = energy_range.min;
//    for(int i=0; i < (energy_range.max - energy_range.min )+1; i++)
//    {
//        energy[i] = e_val;
//        e_val += 1.0;
//    }

//    real_t gain = detector->energy_slope();
//    valarray<real_t> ev = detector->energy_offset() + energy * detector->energy_energy_slope() + pow(energy, (real_t)2.0) * detector->energy_quad();
/*
    if( _snip_background )
    {
        real_t spectral_binning = 0.0;
        spectra->snip_background(_background_counts, detector->energy_offset(), detector->energy_slope(), detector->energy_quad(), spectral_binning, fit_params->at(STR_SNIP_WIDTH).value, energy_range.min, energy_range.max);
    }
*/
/*
    if (keywords.spectral_binning > 0)
    {
        ind = energy/keywords.spectral_binning;
        counts_background = keywords.background[ind.astype(int)];
    }
    else
    {
        counts_background = keywords.background[energy];
    }
*/

    //if(_element_models != nullptr)
    //{
        for(const auto& itr : _element_models)
        {
            if(fit_params->contains(itr.first))
            {
                Fit_Param param = fit_params->at(itr.first);
                real_t va = pow(10.0, param.value);
                spectra_model += pow((real_t)10.0, param.value) * itr.second;
            }
        }
    //}

    /*
    if (np.sum(this->add_matrixfit_pars[3:6]) >= 0.)
    {
        ev = this->add_matrixfit_pars[keywords.energy_pos[0]] + energy * this->add_matrixfit_pars[keywords.energy_pos[1]] + (energy)**2 * this->add_matrixfit_pars[keywords.energy_pos[2]];
        counts_escape = counts.copy();
        counts_escape[:] = 0.0;
        if (this->add_matrixfit_pars[3] > 0.0)
        {
            real_t escape_E = 1.73998;
            wo = np.where(ev > escape_E+ev[0]);

            escape_factor = np.abs(p[len(p)-3] + p[len(p)-1] * ev);
            if (len(wo[0]) > 0)
            {
                for (size_t ii=0; ii<(len(wo[0]); ii++)
                {
                    counts_escape[ii] = counts[wo[0][ii]]*np.amax(np.append(escape_factor[wo[0][ii]],0.0));
                }
            }
            counts = counts + counts_escape;
        }
    }


*/
//   *counts += _background_counts;

    return spectra_model;
}

// ----------------------------------------------------------------------------

unordered_map<string, Spectra> Matrix_Optimized_Fit_Routine::_generate_element_models(const models::Base_Model * const model,
                                                                                      const Detector * const detector,
                                                                                      const Fit_Element_Map_Dict * const elements_to_fit,
                                                                                      struct Range energy_range)
{
    //Eigen::MatrixXd fitmatrix(energy_range.count(), elements_to_fit->size()+2); //+2 for compton and elastic //n_pileup)
    unordered_map<string, Spectra> element_spectra;

    //n_pileup = 9
    //valarray<real_t> value(0.0, energy_range.count());
    valarray<real_t> counts(0.0, energy_range.count());

    Fit_Parameters fit_parameters = model->fit_parameters();
    //set all fit parameters to be fixed. We only want to fit element counts
    fit_parameters.set_all(E_Bound_Type::FIXED);

    valarray<real_t> energy((real_t)0.0, energy_range.count());
    real_t e_val = energy_range.min;
    for(int i=0; i < (energy_range.max - energy_range.min )+1; i++)
    {
        energy[i] = e_val;
        e_val += 1.0;
    }

    real_t gain = detector->energy_slope();
    valarray<real_t> ev = detector->energy_offset() + energy * detector->energy_slope() + pow(energy, (real_t)2.0) * detector->energy_quadratic();

    int i = 0;
    for(const auto& itr : (*elements_to_fit))
    {
        Fit_Element_Map* element = itr.second;
        // Set value to 0.0 . This is the pre_faktor in gauss_tails_model. we do 10.0 ^ pre_faktor = 1.0
        if( false == fit_parameters.contains(itr.first) )
        {
            data_struct::xrf::Fit_Param fp(itr.first, (real_t)-100.0, std::numeric_limits<real_t>::max(), 0.0, (real_t)0.00001, data_struct::xrf::E_Bound_Type::FIT);
            fit_parameters[itr.first] = fp;
        }
        else
        {
            fit_parameters[itr.first].value = 0.0;
        }
        element_spectra[itr.first] = model->model_spectrum_element(&fit_parameters, element, detector, energy);
    }

    //i = elements_to_fit->size();
    // scattering:
    // elastic peak

    Spectra elastic_model(energy_range.count());
    // Set value to 0 because log10(0) = 1.0
    fit_parameters[STR_COHERENT_SCT_AMPLITUDE].value = 0.0;
    elastic_model += model->elastic_peak(&fit_parameters, ev, gain);
    element_spectra[STR_COHERENT_SCT_AMPLITUDE] = elastic_model;
    //Set it so we fit coherent amp in fit params
    ///(*fit_params)[STR_COHERENT_SCT_AMPLITUDE].bound_type = data_struct::xrf::E_Bound_Type::FIT;


    // compton peak
    Spectra compton_model(energy_range.count());
    // Set value to 0 because log10(0) = 1.0
    fit_parameters[STR_COMPTON_AMPLITUDE].value = 0.0;
    compton_model += model->compton_peak(&fit_parameters, ev, gain);
    element_spectra[STR_COMPTON_AMPLITUDE] = compton_model;
    //Set it so we fit STR_COMPTON_AMPLITUDE  in fit params
    ///(*fit_params)[STR_COMPTON_AMPLITUDE].bound_type = data_struct::xrf::FIT;

    /*
    //int this_i = i + 2;
        i = np.amax(keywords.mele_pos)-np.amin(keywords.kele_pos)+1+ii;
        if (add_pars[i, j].energy <= 0.0)
        {
            continue;
        }
        delta_energy = ev.copy() - (add_pars[i, j].energy);
        faktor = add_pars[i, j].ratio;
        counts = faktor * this->model_gauss_peak(gain, sigma[i, j], delta_energy);

        //fitmatrix[:, this_i+ii] = fitmatrix[:, this_i+ii]+counts[:];
        fitmatrix.row(this_i + ii) = fitmatrix.row(this_i + ii) + counts;
        counts = 0.0;
    }
    */
    //return fitmatrix;
    return element_spectra;

}

// ----------------------------------------------------------------------------

void Matrix_Optimized_Fit_Routine::initialize(const models::Base_Model * const model,
                                              const Detector * const detector,
                                              const Fit_Element_Map_Dict * const elements_to_fit,
                                              const struct Range energy_range)
{

    _element_models.clear();
    std::cout<<"-------- Generating element models ---------"<<std::endl;
    _element_models = _generate_element_models(model, detector, elements_to_fit, energy_range);

}

// ----------------------------------------------------------------------------

void Matrix_Optimized_Fit_Routine:: fit_spectra(const models::Base_Model * const model,
                                                const Spectra * const spectra,
                                                const Detector * const detector,
                                                const Fit_Element_Map_Dict * const elements_to_fit,
                                                Fit_Count_Dict *out_counts_dic,
                                                size_t row_idx,
                                                size_t col_idx)
{

    Fit_Parameters fit_params = model->fit_parameters();
    _add_elements_to_fit_parameters(&fit_params, nullptr, detector, elements_to_fit);
    _calc_and_update_coherent_amplitude(&fit_params, spectra, detector);


    //fit

    _save_counts(&fit_params, spectra, detector, elements_to_fit, out_counts_dic, row_idx, col_idx);
}

// ----------------------------------------------------------------------------

} //namespace routines
} //namespace fitting
