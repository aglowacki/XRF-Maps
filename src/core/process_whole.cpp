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

/// Initial Author <2017>: Arthur Glowacki

#include "core/process_whole.h"

using namespace std::placeholders; //for _1, _2,

// ----------------------------------------------------------------------------

const static std::unordered_map<int, std::string> save_loc_map = {
    {data_struct::xrf::ROI, "ROI"},
    {data_struct::xrf::GAUSS_TAILS, "Params"},
    {data_struct::xrf::GAUSS_MATRIX, "Fitted"},
    {data_struct::xrf::SVD, "SVD"},
    {data_struct::xrf::NNLS, "NNLS"}
};

// ----------------------------------------------------------------------------

fitting::routines::Base_Fit_Routine * generate_fit_routine(data_struct::xrf::Fitting_Routines proc_type)
{
    //Fitting routines
    fitting::routines::Base_Fit_Routine *fit_routine = nullptr;
    switch(proc_type)
    {
        case data_struct::xrf::GAUSS_TAILS:
            fit_routine = new fitting::routines::Param_Optimized_Fit_Routine();
            ((fitting::routines::Param_Optimized_Fit_Routine*)fit_routine)->set_optimizer(optimizer);
            break;
        case data_struct::xrf::GAUSS_MATRIX:
            fit_routine = new fitting::routines::Matrix_Optimized_Fit_Routine();
            ((fitting::routines::Matrix_Optimized_Fit_Routine*)fit_routine)->set_optimizer(optimizer);
            break;
        case data_struct::xrf::ROI:
            fit_routine = new fitting::routines::ROI_Fit_Routine();
            break;
        case data_struct::xrf::SVD:
            fit_routine = new fitting::routines::SVD_Fit_Routine();
            break;
        case data_struct::xrf::NNLS:
            fit_routine = new fitting::routines::NNLS_Fit_Routine();
            break;
        default:
            break;
    }
    return fit_routine;
}

// ----------------------------------------------------------------------------

template<typename T>
data_struct::xrf::Fit_Count_Dict* generate_fit_count_dict(std::unordered_map<std::string, T> *elements_to_fit, size_t width, size_t height )
{
    data_struct::xrf::Fit_Count_Dict* element_fit_counts_dict = new data_struct::xrf::Fit_Count_Dict();
    for(auto& e_itr : *elements_to_fit)
    {
        element_fit_counts_dict->emplace(std::pair<std::string, Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(e_itr.first, Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ()) );
        element_fit_counts_dict->at(e_itr.first).resize(width, height);
    }
    return element_fit_counts_dict;
}

// ----------------------------------------------------------------------------

bool fit_single_spectra(fitting::routines::Base_Fit_Routine * fit_routine,
                        const fitting::models::Base_Model * const model,
                        const data_struct::xrf::Spectra * const spectra,
                        const data_struct::xrf::Fit_Element_Map_Dict * const elements_to_fit,
                        data_struct::xrf::Fit_Count_Dict * out_fit_counts,
                        size_t i,
                        size_t j)
{
    std::unordered_map<std::string, real_t> counts_dict = fit_routine->fit_spectra(model, spectra, elements_to_fit);
    //save count / sec
    for (auto& el_itr : *elements_to_fit)
    {
        (*out_fit_counts)[el_itr.first](i,j) = counts_dict[el_itr.first] / spectra->elapsed_lifetime();
    }
    (*out_fit_counts)[data_struct::xrf::STR_NUM_ITR](i,j) = counts_dict[data_struct::xrf::STR_NUM_ITR];
    return true;
}

// ----------------------------------------------------------------------------

 struct io::file_name_fit_params optimize_integrated_fit_params(std::string dataset_directory,
                                                                std::string  dataset_filename,
                                                                size_t detector_num)
{

    //return structure
    struct io::file_name_fit_params ret_struct;

    ret_struct.dataset_dir = dataset_directory;
    ret_struct.dataset_filename = dataset_filename;
    ret_struct.detector_num = detector_num;

    fitting::models::Gaussian_Model model;

    //Range of energy in spectra to fit
    fitting::models::Range energy_range;
    energy_range.min = 0;


    data_struct::xrf::Params_Override params_override;

    //load override parameters
    io::load_override_params(dataset_directory, -1, &params_override);

    ret_struct.elements_to_fit = params_override.elements_to_fit;

    //load the quantification standard dataset
    if(false == io::load_and_integrate_spectra_volume(dataset_directory, dataset_filename, &ret_struct.spectra, detector_num, &params_override) )
    {
        logit<<"Error in optimize_integrated_dataset loading dataset"<<dataset_filename<<" for detector"<<detector_num<<std::endl;
        ret_struct.success = false;
        return ret_struct;
    }

    energy_range.max = ret_struct.spectra.size() -1;


    //Fitting routines
    fitting::routines::Param_Optimized_Fit_Routine fit_routine;
    fit_routine.set_optimizer(optimizer);

    //reset model fit parameters to defaults
    model.reset_to_default_fit_params();
    //Update fit parameters by override values
    model.update_fit_params_values(params_override.fit_params);
    model.set_fit_params_preset(optimize_fit_params_preset);
    //Initialize the fit routine
    fit_routine.initialize(&model, &ret_struct.elements_to_fit, energy_range);
    //Fit the spectra saving the element counts in element_fit_count_dict
    ret_struct.fit_params = fit_routine.fit_spectra_parameters(&model, &ret_struct.spectra, &ret_struct.elements_to_fit);

    ret_struct.success = true;

    return ret_struct;

}

// ----------------------------------------------------------------------------

void generate_optimal_params(std::string dataset_directory,
                             std::vector<std::string> dataset_files,
                             ThreadPool* tp,
                             size_t detector_num_start,
                             size_t detector_num_end)
{
    bool first = true;
    std::queue<std::future<struct io::file_name_fit_params> > job_queue;

    std::vector<data_struct::xrf::Fit_Parameters> fit_params_avgs;
    fit_params_avgs.resize(detector_num_end - detector_num_start + 1);

    for(auto &itr : dataset_files)
    {
        for(size_t detector_num = detector_num_start; detector_num <= detector_num_end; detector_num++)
        {
            //data_struct::xrf::Fit_Parameters out_fitp;
            //out_fitp = optimize_integrated_fit_params(dataset_directory, itr, detector_num);
            job_queue.emplace( tp->enqueue(optimize_integrated_fit_params, dataset_directory, itr, detector_num) );
        }
    }


    while(!job_queue.empty())
    {
        auto ret = std::move(job_queue.front());
        job_queue.pop();
        struct io::file_name_fit_params f_struct = ret.get();
        if(f_struct.success)
        {
            if(first)
            {
                fit_params_avgs[f_struct.detector_num] = f_struct.fit_params;
                first = false;
            }
            else
            {
                fit_params_avgs[f_struct.detector_num].moving_average_with(f_struct.fit_params);
            }
            io::save_optimized_fit_params(f_struct);
        }
    }
    io::save_averaged_fit_params(dataset_directory, fit_params_avgs, detector_num_start, detector_num_end);

}

// ----------------------------------------------------------------------------

void proc_spectra(data_struct::xrf::Spectra_Volume* spectra_volume,
                  std::vector<data_struct::xrf::Fitting_Routines> proc_types,
                  data_struct::xrf::Params_Override * override_params,
                  data_struct::xrf::Quantification_Standard * quantification_standard,
                  ThreadPool* tp,
                  bool save_spectra_vol)
{
    //Model
    fitting::models::Gaussian_Model model;

    //Range of energy in spectra to fit
    fitting::models::Range energy_range;
    energy_range.min = 0;
    energy_range.max = spectra_volume->samples_size() -1;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    for(auto proc_type : proc_types)
    {
        logit << "Processing  "<< save_loc_map.at(proc_type)<<std::endl;

        start = std::chrono::system_clock::now();
        if (override_params->elements_to_fit.size() < 1)
        {
            logit<<"Error, no elements to fit. Check  maps_fit_parameters_override.txt0 - 3 exist"<<std::endl;
            continue;
        }

		//Fit job queue
		std::queue<std::future<bool> >* fit_job_queue = new std::queue<std::future<bool> >();

        //Allocate memeory to save fit counts
        data_struct::xrf::Fit_Count_Dict  *element_fit_count_dict = generate_fit_count_dict(&override_params->elements_to_fit, spectra_volume->rows(), spectra_volume->cols());

        //Fitting models
        fitting::routines::Base_Fit_Routine *fit_routine = generate_fit_routine(proc_type);

        //for now we default to true to save iter count, in the future if we change the hdf5 layout we can store it per analysis.
        bool alloc_iter_count = true;

        //reset model fit parameters to defaults
        model.reset_to_default_fit_params();
        //Update fit parameters by override values
        model.update_fit_params_values(override_params->fit_params);

        if (alloc_iter_count)
        {
            //Allocate memeory to save number of fit iterations
            element_fit_count_dict->emplace(std::pair<std::string, Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(data_struct::xrf::STR_NUM_ITR, Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>() ));
            element_fit_count_dict->at(data_struct::xrf::STR_NUM_ITR).resize(spectra_volume->rows(), spectra_volume->cols());
        }

        //Initialize model
        fit_routine->initialize(&model, &override_params->elements_to_fit, energy_range);

        for(size_t i=0; i<spectra_volume->rows(); i++)
        {
            for(size_t j=0; j<spectra_volume->cols(); j++)
            {
                //logit<< i<<" "<<j<<std::endl;
                fit_job_queue->emplace( tp->enqueue(fit_single_spectra, fit_routine, &model, &(*spectra_volume)[i][j], &override_params->elements_to_fit, element_fit_count_dict, i, j) );
            }
        }

        io::save_results( save_loc_map.at(proc_type), element_fit_count_dict, fit_routine, fit_job_queue, start );
    }

    real_t energy_offset = 0.0;
    real_t energy_slope = 0.0;
    real_t energy_quad = 0.0;
    data_struct::xrf::Fit_Parameters fit_params = model.fit_parameters();
    if(fit_params.contains(fitting::models::STR_ENERGY_OFFSET))
    {
        energy_offset = fit_params[fitting::models::STR_ENERGY_OFFSET].value;
    }
    if(fit_params.contains(fitting::models::STR_ENERGY_SLOPE))
    {
        energy_slope = fit_params[fitting::models::STR_ENERGY_SLOPE].value;
    }
    if(fit_params.contains(fitting::models::STR_ENERGY_QUADRATIC))
    {
        energy_quad = fit_params[fitting::models::STR_ENERGY_QUADRATIC].value;
    }

    if(save_spectra_vol)
    {
        io::save_volume( quantification_standard, spectra_volume, energy_offset, energy_slope, energy_quad);
    }
    else
    {
        io::file::HDF5_IO::inst()->end_save_seq();
    }

}

// ----------------------------------------------------------------------------

void process_dataset_file_quick_n_dirty(std::string dataset_directory,
                                        std::string dataset_file,
                                        std::vector<data_struct::xrf::Fitting_Routines> proc_types,
                                        ThreadPool* tp,
                                        std::vector<data_struct::xrf::Quantification_Standard>* quant_stand_list,
                                        std::unordered_map<int, data_struct::xrf::Params_Override> *fit_params_override_dict,
                                        size_t detector_num_start,
                                        size_t detector_num_end)
{

    std::chrono::time_point<std::chrono::system_clock> start, end;

    data_struct::xrf::Params_Override * override_params = nullptr;
    if(override_params == nullptr && fit_params_override_dict->count(-1) > 0)
    {
       override_params = &fit_params_override_dict->at(-1);
    }
    if(override_params == nullptr)
    {
        logit<<"Quick and Dirty Skipping file "<<dataset_directory<<" dset "<< dataset_file << " because could not find maps_fit_params_override.txt"<<std::endl;
        return;
    }

    //data_struct::xrf::Quantification_Standard * quantification_standard = &(*quant_stand_list)[detector_num];

    std::string full_save_path = dataset_directory+"/img.dat/"+dataset_file+".h5";

    //Spectra volume data
    data_struct::xrf::Spectra_Volume* spectra_volume = new data_struct::xrf::Spectra_Volume();
    data_struct::xrf::Spectra_Volume* tmp_spectra_volume = new data_struct::xrf::Spectra_Volume();

    io::file::HDF5_IO::inst()->set_filename(full_save_path);

    //load the first one
    size_t detector_num = detector_num_start;
    bool is_loaded_from_analyzed_h5;
    if (false == io::load_spectra_volume(dataset_directory, dataset_file, spectra_volume, detector_num, override_params, nullptr, &is_loaded_from_analyzed_h5, true) )
    {
        logit<<"Error loading all detectors for "<<dataset_directory<<"/"<<dataset_file<<std::endl;
        delete spectra_volume;
        delete tmp_spectra_volume;
        return;
    }

    //load spectra volume
    for(detector_num = detector_num_start+1; detector_num <= detector_num_end; detector_num++)
    {

        if (false == io::load_spectra_volume(dataset_directory, dataset_file, tmp_spectra_volume, detector_num, override_params, nullptr, &is_loaded_from_analyzed_h5, false) )
        {
            logit<<"Error loading all detectors for "<<dataset_directory<<"/"<<dataset_file<<std::endl;
            delete spectra_volume;
            delete tmp_spectra_volume;
            return;
        }
        //add all detectors up
        for(int j=0; j<spectra_volume->rows(); j++)
        {
            for(int k=0; k<spectra_volume->cols(); k++)
            {
                real_t elapsed_lifetime = (*spectra_volume)[j][k].elapsed_lifetime();
                real_t elapsed_realtime = (*spectra_volume)[j][k].elapsed_realtime();
                real_t input_counts = (*spectra_volume)[j][k].input_counts();
                real_t output_counts = (*spectra_volume)[j][k].output_counts();


                (*spectra_volume)[j][k] += (*tmp_spectra_volume)[j][k];

                elapsed_lifetime += (*tmp_spectra_volume)[j][k].elapsed_lifetime();
                elapsed_realtime += (*tmp_spectra_volume)[j][k].elapsed_realtime();
                input_counts += (*tmp_spectra_volume)[j][k].input_counts();
                output_counts += (*tmp_spectra_volume)[j][k].output_counts();

                (*spectra_volume)[j][k].elapsed_lifetime(elapsed_lifetime);
                (*spectra_volume)[j][k].elapsed_realtime(elapsed_realtime);
                (*spectra_volume)[j][k].input_counts(input_counts);
                (*spectra_volume)[j][k].output_counts(output_counts);
            }
        }
    }
    delete tmp_spectra_volume;

    proc_spectra(spectra_volume, proc_types, override_params, nullptr, tp, !is_loaded_from_analyzed_h5);

}

// ----------------------------------------------------------------------------

void process_dataset_file(std::string dataset_directory,
                          std::string dataset_file,
                          std::vector<data_struct::xrf::Fitting_Routines> proc_types,
                          ThreadPool* tp,
                          std::vector<data_struct::xrf::Quantification_Standard>* quant_stand_list,
                          std::unordered_map<int, data_struct::xrf::Params_Override> *fit_params_override_dict,
                          size_t detector_num_start,
                          size_t detector_num_end)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;

    for(size_t detector_num = detector_num_start; detector_num <= detector_num_end; detector_num++)
    {
        data_struct::xrf::Params_Override * override_params = nullptr;
        if(fit_params_override_dict->count(detector_num) > 0)
        {
           override_params = &fit_params_override_dict->at(detector_num);
        }
        if(override_params == nullptr && fit_params_override_dict->count(-1) > 0)
        {
           override_params = &fit_params_override_dict->at(-1);
        }
        if(override_params == nullptr)
        {
            logit<<"Skipping file "<<dataset_directory<<" dset "<< dataset_file << " detector "<<detector_num<<" because could not find maps_fit_params_override.txt"<<std::endl;
            continue;
        }

        data_struct::xrf::Quantification_Standard * quantification_standard = &(*quant_stand_list)[detector_num];


        //Spectra volume data
        data_struct::xrf::Spectra_Volume* spectra_volume = new data_struct::xrf::Spectra_Volume();

        std::string str_detector_num = std::to_string(detector_num);
        std::string full_save_path = dataset_directory+"/img.dat/"+dataset_file+".h5"+str_detector_num;
        io::file::HDF5_IO::inst()->set_filename(full_save_path);

        bool loaded_from_analyzed_hdf5 = false;
        //load spectra volume
        if (false == io::load_spectra_volume(dataset_directory, dataset_file, spectra_volume, detector_num, override_params, nullptr, &loaded_from_analyzed_hdf5, true) )
        {
            logit<<"Skipping detector "<<detector_num<<std::endl;
			delete spectra_volume;
            continue;
        }

        proc_spectra(spectra_volume, proc_types, override_params, quantification_standard, tp, !loaded_from_analyzed_hdf5);
    }

}


// ----------------------------------------------------------------------------

bool perform_quantification(std::string dataset_directory,
                            std::string quantification_info_file,
                            std::vector<data_struct::xrf::Fitting_Routines> proc_types,
                            std::vector<data_struct::xrf::Quantification_Standard>* quant_stand_list,
                            std::unordered_map<int, data_struct::xrf::Params_Override> *fit_params_override_dict,
                            size_t detector_num_start,
                            size_t detector_num_end)
{

    bool air_path = false;
    real_t detector_chip_thickness = 0.0;
    real_t beryllium_window_thickness = 0.0;
    real_t germanium_dead_layer = 0.0;
    real_t incident_energy = 10.0;

    fitting::models::Gaussian_Model model;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    logit << "Perform_quantification()"<<std::endl;

    data_struct::xrf::Element_Info* detector_element = data_struct::xrf::Element_Info_Map::inst()->get_element("Si");

    //Range of energy in spectra to fit
    fitting::models::Range energy_range;
    energy_range.min = 0;

    std::string standard_file_name;
    std::unordered_map<std::string, real_t> element_standard_weights;
    //data_struct::xrf::Quantification_Standard* quantification_standard = &(*quant_stand_list)[0];

    if( io::load_quantification_standard(dataset_directory, quantification_info_file, &standard_file_name, &element_standard_weights) )
    {
        for(size_t detector_num = detector_num_start; detector_num <= detector_num_end; detector_num++)
        {
            data_struct::xrf::Params_Override * override_params = nullptr;
            if(fit_params_override_dict->count(detector_num) > 0)
            {
               override_params = &fit_params_override_dict->at(detector_num);
            }
            if(override_params == nullptr && fit_params_override_dict->count(-1) > 0)
            {
               override_params = &fit_params_override_dict->at(-1);
            }
            if(override_params == nullptr)
            {
                logit<<"Skipping file "<<dataset_directory << " detector "<<detector_num<<" because could not find maps_fit_params_override.txt"<<std::endl;
                continue;
            }


            data_struct::xrf::Quantification_Standard* quantification_standard = &(*quant_stand_list)[detector_num];
            quantification_standard->standard_filename(standard_file_name);
            for(auto& itr : element_standard_weights)
            {
                quantification_standard->append_element(itr.first, itr.second);
            }

            //Parameters for calibration curve

            if (override_params->detector_element.length() > 0)
            {
                // Get the element info class                                           // detector element as string "Si" or "Ge" usually
                detector_element = data_struct::xrf::Element_Info_Map::inst()->get_element(override_params->detector_element);
            }
            if (override_params->be_window_thickness.length() > 0)
            {
                beryllium_window_thickness = std::stof(override_params->be_window_thickness);
            }
            if (override_params->ge_dead_layer.length() > 0)
            {
                germanium_dead_layer = std::stof(override_params->ge_dead_layer);
            }
            if (override_params->det_chip_thickness.length() > 0)
            {
                detector_chip_thickness = std::stof(override_params->det_chip_thickness);
            }

            if(override_params->fit_params.contains(data_struct::xrf::STR_COHERENT_SCT_ENERGY))
            {
                incident_energy = override_params->fit_params.at(data_struct::xrf::STR_COHERENT_SCT_ENERGY).value;
            }

            //Output of fits for elements specified
            std::unordered_map<std::string, data_struct::xrf::Fit_Element_Map*> elements_to_fit;
            for(auto& itr : element_standard_weights)
            {
                data_struct::xrf::Element_Info* e_info = data_struct::xrf::Element_Info_Map::inst()->get_element(itr.first);
                elements_to_fit[itr.first] = new data_struct::xrf::Fit_Element_Map(itr.first, e_info);
                elements_to_fit[itr.first]->init_energy_ratio_for_detector_element( detector_element );
            }

            data_struct::xrf::Spectra_Volume spectra_volume;
            bool is_loaded_from_analyzed_h5;
            //load the quantification standard dataset
            if(false == io::load_spectra_volume(dataset_directory, quantification_standard->standard_filename(), &spectra_volume, detector_num, override_params, quantification_standard, &is_loaded_from_analyzed_h5, false) )
            {
                //legacy code would load mca files, check for mca and replace with mda
                int std_str_len = standard_file_name.length();
                if(standard_file_name[std_str_len - 4] == '.' && standard_file_name[std_str_len - 3] == 'm' && standard_file_name[std_str_len - 2] == 'c' && standard_file_name[std_str_len - 1] == 'a')
                {
                    standard_file_name[std_str_len - 2] = 'd';
                    quantification_standard->standard_filename(standard_file_name);
                    if(false == io::load_spectra_volume(dataset_directory, quantification_standard->standard_filename(), &spectra_volume, detector_num, override_params, quantification_standard, &is_loaded_from_analyzed_h5, false) )
                    {
                        logit<<"Error perform_quantification() : could not load file "<< standard_file_name <<" for detector"<<detector_num<<std::endl;
                        return false;
                    }
                }
                else
                {
                    logit<<"Error perform_quantification() : could not load file "<< standard_file_name <<" for detector"<<detector_num<<std::endl;
                    return false;
                }
            }

            //First we integrate the spectra and get the elemental counts
            data_struct::xrf::Spectra integrated_spectra = spectra_volume.integrate();
            energy_range.max = integrated_spectra.size() -1;


            for(auto proc_type : proc_types)
            {

                //Fitting routines
                fitting::routines::Base_Fit_Routine *fit_routine = generate_fit_routine(proc_type);

                //reset model fit parameters to defaults
                model.reset_to_default_fit_params();
                //Update fit parameters by override values
                model.update_fit_params_values(override_params->fit_params);
                //Initialize the fit routine
                fit_routine->initialize(&model, &elements_to_fit, energy_range);
                //Fit the spectra
                std::unordered_map<std::string, real_t>counts_dict = fit_routine->fit_spectra(&model,
                                                                                              &integrated_spectra,
                                                                                              &elements_to_fit);

                if (fit_routine != nullptr)
                {
                    delete fit_routine;
                    fit_routine = nullptr;
                }

                for (auto& itr : elements_to_fit)
                {
                    counts_dict[itr.first] /= integrated_spectra.elapsed_lifetime();
                }
                quantification_standard->integrated_spectra(integrated_spectra);

                //save for each proc
                quantification_standard->quantifiy(optimizer,
                                                   save_loc_map.at(proc_type),
                                                  &counts_dict,
                                                  incident_energy,
                                                  detector_element,
                                                  air_path,
                                                  detector_chip_thickness,
                                                  beryllium_window_thickness,
                                                  germanium_dead_layer);

            }

            //cleanup
            for(auto &itr : elements_to_fit)
            {
                delete itr.second;
            }

        }
    }
    else
    {
        logit<<"Error loading quantification standard "<<quantification_info_file<<std::endl;
        return false;
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    logit << "quantification elapsed time: " << elapsed_seconds.count() << "s"<<std::endl;

    return true;

}

// ----------------------------------------------------------------------------

void average_quantification(std::vector<data_struct::xrf::Quantification_Standard>* quant_stand_list,
                            size_t detector_num_start,
                            size_t detector_num_end)
{
    /*
    data_struct::xrf::Quantification_Standard q_standard_0;
    q_standard_0.


    for(size_t detector_num = detector_num_start; detector_num <= detector_num_end; detector_num++)
    {
        data_struct::xrf::Quantification_Standard * quantification_standard = &(*quant_stand_list)[detector_num];
        quantification_standard->
    }
    */
}

// ----------------------------------------------------------------------------