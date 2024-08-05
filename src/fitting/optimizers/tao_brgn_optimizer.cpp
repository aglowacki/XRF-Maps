/***
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

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

/// Initial Author <2024>: Arthur Glowacki


#include "tao_brgn_optimizer.h"
#include "fitting/models/gaussian_model.h"

using namespace data_struct;


namespace fitting
{
namespace optimizers
{

// ------------------------------------------------------------ 
/*
PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec res, void *ptr)
{
    PetscFunctionBegin;
    bool first = true;
    User_Data<double>* ud = (User_Data<double>*)(ptr);
    // Debug to find which param changed last
    Fit_Parameters<double> prev_fit_p;
    prev_fit_p.update_and_add_values(ud->fit_parameters);
    const PetscReal *x;
    PetscReal *f;

    VecGetArrayRead(X, &x);
    VecGetArray(res, &f);

    // Update fit parameters from optimizer
    ud->fit_parameters->from_array(x, ud->fit_parameters->size());
    // Model spectra based on new fit parameters
    update_background_user_data(ud);
    ud->spectra_model = ud->fit_model->model_spectrum(ud->fit_parameters, ud->elements, nullptr, ud->energy_range);
    // Add background
    ud->spectra_model += ud->spectra_background;
    // Remove nan's and inf's
    ud->spectra_model = (ArrayTr<double>)ud->spectra_model.unaryExpr([ud](double v) { return std::isfinite(v) ? v : ud->normalizer; });

    ArrayTr<double> diff = ud->spectra - ud->spectra_model;
    diff = diff.pow(2.0);
    diff *= ud->weights;
    for(int i=0; i < diff.size(); i++)
    {
        f[i] = diff(i);
    }
    VecRestoreArrayRead(X, &x);
    VecRestoreArray(res, &f);
    /*
    ud->cur_itr++;
    if (ud->status_callback != nullptr)
    {
        try
        {
            (*ud->status_callback)(ud->cur_itr, ud->total_itr);
        }
        catch (...)
        {
            logI << "Cancel fitting" << std::endl;
            *userbreak = 1;
        }
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}
*/
PetscErrorCode EvaluateResidual2(Tao tao, Vec X, PetscReal *f, void *ptr)
{
    PetscFunctionBegin;
    bool first = true;
    User_Data<double>* ud = (User_Data<double>*)(ptr);
    // Debug to find which param changed last
    Fit_Parameters<double> prev_fit_p;
    prev_fit_p.update_and_add_values(ud->fit_parameters);
    const PetscReal *x;

    VecGetArrayRead(X, &x);

    // Update fit parameters from optimizer
    ud->fit_parameters->from_array(x, ud->fit_parameters->size());

    VecRestoreArrayRead(X, &x);
/*
    for(auto itr = ud->fit_parameters->begin(); itr != ud->fit_parameters->end(); itr++)
    {
        double pval = prev_fit_p.at(itr->first).value;
        if(pval != itr->second.value)
        {   
            double df = itr->second.value - pval;
            //logI<<itr->first<<" : diff : "<<df<< " : old val = "<< pval << " : new val = "<< itr->second.value<<"\n";
            //(*ud->fit_parameters)[itr->first].value = pval + (df * 10.0);
            //logI<<itr->first<<" val = "<< (*ud->fit_parameters)[itr->first].value << "\n";
        } 
    }
*/
    // Model spectra based on new fit parameters
    update_background_user_data(ud);
    ud->spectra_model = ud->fit_model->model_spectrum(ud->fit_parameters, ud->elements, nullptr, ud->energy_range);
    // Add background
    ud->spectra_model += ud->spectra_background;

    ud->spectra_model = (ArrayTr<double>)(ud->spectra_model.log10());
    // Remove nan's and inf's
    ud->spectra_model = (ArrayTr<double>)ud->spectra_model.unaryExpr([ud](double v) { return std::isfinite(v) ? v : ud->normalizer; });

    ArrayTr<double> diff = ud->spectra - ud->spectra_model;
    //diff = diff / ud->normalizer;
    //diff = Eigen::abs(diff);
    diff = diff.pow(2.0);
    diff *= ud->weights;
    //VecTDot(diff, diff, f);
    *f = diff.sum() / diff.size(); // do dot product of diff
    //*f = diff.sum();
    //logit<<"f = "<<*f<<"\n";
    /*
    ud->cur_itr++;
    if (ud->status_callback != nullptr)
    {
        try
        {
            (*ud->status_callback)(ud->cur_itr, ud->total_itr);
        }
        catch (...)
        {
            logI << "Cancel fitting" << std::endl;
            *userbreak = 1;
        }
    }
    */
    PetscFunctionReturn(PETSC_SUCCESS);
}



// ------------------------------------------------------------ 
/*
PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
    // Jacobian is not changing here, so use a empty dummy function here.  J[m][n] = df[m]/dx[n] = A[m][n] for linear least square 
    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}

 // ------------------------------------------------------------ 

PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao tao, Vec X, PetscReal *f_reg, Vec G_reg, void *ptr)
{
    PetscFunctionBegin;
    // compute regularizer objective = 0.5*x'*x 
    PetscCall(VecDot(X, X, f_reg));
    *f_reg *= 0.5;
    // compute regularizer gradient = x 
    PetscCall(VecCopy(X, G_reg));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------ 

PetscErrorCode EvaluateRegularizerHessianProd(Mat Hreg, Vec in, Vec out)
{
    PetscFunctionBegin;
    PetscCall(VecCopy(in, out));
    PetscFunctionReturn(PETSC_SUCCESS);
}

 // ------------------------------------------------------------

PetscErrorCode EvaluateRegularizerHessian(Tao tao, Vec X, Mat Hreg, void *ptr)
{
    // Hessian for regularizer objective = 0.5*x'*x is identity matrix, and is not changing
    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}
*/
// ------------------------------------------------------------ 
// ------------------------------------------------------------ 

template<typename T_real>
TAO_BRGN_Optimizer<T_real>::TAO_BRGN_Optimizer() : Optimizer<T_real>()
{

}

// ------------------------------------------------------------ 

template<typename T_real>
OPTIMIZER_OUTCOME TAO_BRGN_Optimizer<T_real>::minimize(Fit_Parameters<T_real>*fit_params,
                                    const Spectra<T_real>* const spectra,
                                    const Fit_Element_Map_Dict<T_real>* const elements_to_fit,
                                    const Base_Model<T_real>* const model,
                                    const Range energy_range,
                                    bool use_weights,
                                    Callback_Func_Status_Def* status_callback)
{
    PetscFunctionBeginUser;
    User_Data<double> ud;

    Vec  rx, x, xlb, xub; //b, xGT,
    PetscReal   hist[100], resid[100], v1, v2;
    Mat  A, Hreg;
    PetscInt    lits[100], M, N;


    Fit_Parameters<double> dfit_params;
    Fit_Element_Map_Dict<double> delements_to_fit;
    fitting::models::Gaussian_Model<double> dmodel;
    //dmodel.
    Spectra<double> dspectra = spectra->to_double();
    for(const auto itr : *elements_to_fit)
    {
        delements_to_fit[itr.first] = itr.second->to_double();
    }
// set peaks to 0
/*
    for(auto itr = fit_params->begin(); itr != fit_params->end(); itr++)
    {
        if(itr->second.name == STR_ENERGY_OFFSET || itr->second.name == STR_ENERGY_SLOPE)
        {
            dfit_params.add_parameter(Fit_Param<double>(itr->second.name, static_cast<double>(itr->second.min_val), static_cast<double>(itr->second.max_val), static_cast<double>(itr->second.value), static_cast<double>(itr->second.step_size), itr->second.bound_type));
        }
        else
        {
            dfit_params.add_parameter(Fit_Param<double>(itr->second.name, static_cast<double>(itr->second.min_val), static_cast<double>(itr->second.max_val), 1.0, static_cast<double>(itr->second.step_size), itr->second.bound_type));
        }
        
    }
*/
    for(auto itr = fit_params->begin(); itr != fit_params->end(); itr++)
    {
        dfit_params.add_parameter(Fit_Param<double>(itr->second.name, static_cast<double>(itr->second.min_val), static_cast<double>(itr->second.max_val), static_cast<double>(itr->second.value), static_cast<double>(itr->second.step_size), itr->second.bound_type));    
    }

    //logI<<"\n-=-=-=-=-==-=-=-=-=-==-\n\n";
    dfit_params.print_non_fixed();
    
    std::vector<double> fitp_arr = dfit_params.to_array();
    if (fitp_arr.size() == 0)
    {
        return OPTIMIZER_OUTCOME::STOPPED;
    }

    int total_itr = 100000;
    fill_user_data(ud, &dfit_params, &dspectra, &delements_to_fit, &dmodel, energy_range, status_callback, total_itr, use_weights);
/*
    std::vector<double> fitp_arr = fit_params->to_array();
    if (fitp_arr.size() == 0)
    {
        return OPTIMIZER_OUTCOME::STOPPED;
    }

    int total_itr = 2000;
    fill_user_data(ud, fit_params, spectra, elements_to_fit, model, energy_range, status_callback, total_itr, use_weights);
*/

    std::vector<double> lower_bound = dfit_params.lower_to_array();
    std::vector<double> upper_bound = dfit_params.upper_to_array();
    int len = lower_bound.size();
    N = len;
    int* indices = new int[len];
    for (int i = 0; i < len; ++i)
    {
        indices[i] = i;
    }
/*
    int slen = ud.spectra.size();
    M = slen;
    int* sindices = new int[slen];
    for (int i = 0; i < slen; ++i) 
    {
        sindices[i] = i;
    }
*/
    Tao tao;
    TaoConvergedReason reason;
    int argc = 1;
    char **argv;
    argv = new char*[1];
    argv[0] = new char[17];
    strcpy(argv[0], "-tao_converged_reason");
    argv[0][16] = '\0';

    PetscInitialize(&argc, &argv, (char *)0, (char *)0);
    // Create TAO solver and set desired solution method 
    TaoCreate(PETSC_COMM_SELF, &tao);
    //TaoSetType(tao, TAOBRGN);
    //TaoSetType(tao, TAOBLMVM); // do not use, will be depricated
    TaoSetType(tao, TAOBNCG); // baseline
    //TaoSetType(tao, TAOBQNLS); 
    //TaoSetType(tao, TAOBQNKLS); // does not work
    //TaoSetType(tao, TAOBQNKTR); // does not work
    //TaoSetType(tao, TAOBQNKTL); // does not work

    //TaoSetTolerances(tao, 1.0e-11, 1.0e-11, 1.8e-11);
// setup tao command line
// go to bncg.c and look at params
// up num iter
    // User set application context: A, D matrice, and b vector. 
    //InitializeUserData(&user));

    // Allocate solution vector x,  and function vectors Ax-b, 
    VecCreateSeq(PETSC_COMM_SELF, N, &x);
    VecCreateSeq(PETSC_COMM_SELF, N, &xlb);
    VecCreateSeq(PETSC_COMM_SELF, N, &xub);
   // VecCreateSeq(PETSC_COMM_SELF, M, &res);



    VecSetValues(xlb, len, indices, &lower_bound[0], INSERT_VALUES);
    VecSetValues(xub, len, indices, &upper_bound[0], INSERT_VALUES);
    VecSetValues(x, len, indices, &fitp_arr[0], INSERT_VALUES);
 //   VecSetValues(res, slen, sindices, ud.spectra.data(), INSERT_VALUES);


    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetType(A, MATSEQAIJ);

    // Set initial guess 
    //FormStartingPoint(x, &user);

    // Bind x to tao->solution. 
    //VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    TaoSetSolution(tao, x);
    // Sets the upper and lower bounds of x 
    TaoSetVariableBounds(tao, xlb, xub);

    TaoSetObjective(tao, EvaluateResidual2, (void *)&ud);
    TaoSetGradient(tao, nullptr, TaoDefaultComputeGradient, nullptr);
   // TaoSetHessian(tao, nullptr, nullptr, TaoDefaultComputeHessian, nullptr); // only for bqnkls

//  "-tao_monitor_solution"

    // Bind user.D to tao->data->D 
    //TaoBRGNSetDictionaryMatrix(tao, user.D);
    //TaoBRGNSetDictionaryMatrix(tao, nullptr);

    // Set the residual function and Jacobian routines for least squares. 
    //TaoSetResidualRoutine(tao, res, EvaluateResidual, (void *)&ud);
    // Jacobian matrix fixed as user.A for Linear least square problem. 
    //TaoSetJacobianResidualRoutine(tao, A, A, EvaluateJacobian, (void *)&ud);

    // User set the regularizer objective, gradient, and hessian. Set it the same as using l2prox choice, for testing purpose.  
    //TaoBRGNSetRegularizerObjectiveAndGradientRoutine(tao, EvaluateRegularizerObjectiveAndGradient, (void *)&ud);
    /*
    // User defined regularizer Hessian setup, here is identity shell matrix 
    MatCreate(PETSC_COMM_SELF, &Hreg);
    MatSetSizes(Hreg, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetType(Hreg, MATSHELL);
    MatSetUp(Hreg);
    MatShellSetOperation(Hreg, MATOP_MULT, (void (*)(void))EvaluateRegularizerHessianProd);
    TaoBRGNSetRegularizerHessianRoutine(tao, Hreg, EvaluateRegularizerHessian, (void *)&ud);
    */
    // Check for any TAO command line arguments 
    //TaoSetFromOptions(tao);

    ////TaoSetConvergenceHistory(tao, hist, resid, 0, lits, 100, PETSC_TRUE);
    TaoSetMaximumFunctionEvaluations(tao, total_itr);
    TaoSetMaximumIterations(tao, total_itr);
    // Perform the Solve 
    TaoSolve(tao);

    TaoView(tao, PETSC_VIEWER_STDOUT_WORLD);
    TaoGetConvergedReason(tao, &reason);
    logI<<"reason = "<<reason<<"\n\n";
    // Save x (reconstruction of object) vector to a binary file, which maybe read from MATLAB and convert to a 2D image for comparison. 
    //PetscViewerBinaryOpen(PETSC_COMM_SELF, resultFile, FILE_MODE_WRITE, &fd);
    //VecView(x, fd);
    //PetscViewerDestroy(&fd);

    TaoGetSolution(tao, &rx);
    VecView(rx, PETSC_VIEWER_STDOUT_WORLD);
    VecGetValues(rx, len, indices, &fitp_arr[0]);
    dfit_params.from_array(fitp_arr);
    dfit_params.print_non_fixed();
    
    for(auto itr = fit_params->begin(); itr != fit_params->end(); itr++)
    {
        if(dfit_params.contains(itr->first))
        {
            if(dfit_params.at(itr->first).bound_type != E_Bound_Type::FIXED)
            {
                logI<<itr->first<<" : diff : "<<(*fit_params)[itr->first].value - dfit_params.at(itr->first).value<< "\n";
            }
            (*fit_params)[itr->first].value = static_cast<T_real>(dfit_params.at(itr->first).value);
        } 
    }
    //fit_params->print();
    // compute the error 
    //VecAXPY(x, -1, xGT);
    //VecNorm(x, NORM_2, &v1);
    //VecNorm(xGT, NORM_2, &v2);
    //logI<< "relative reconstruction error: ||x-xGT||/||xGT|| = "<< (double)(v1 / v2)<<"\n";

    // Free TAO data structures 
    TaoDestroy(&tao);

    // Free PETSc data structures 
    VecDestroy(&x);
    //VecDestroy(&res);
    //MatDestroy(&Hreg);
    // Free user data structures 
    //MatDestroy(&A);
    //MatDestroy(&D);
    //VecDestroy(&b);
    //VecDestroy(&xGT);
    VecDestroy(&xlb);
    VecDestroy(&xub);
    PetscFinalize();

    return OPTIMIZER_OUTCOME::CONVERGED;
}

// ------------------------------------------------------------ 

template<typename T_real>
OPTIMIZER_OUTCOME TAO_BRGN_Optimizer<T_real>::minimize_func(Fit_Parameters<T_real>*fit_params,
                                        const Spectra<T_real>* const spectra,
                                        const Range energy_range,
                                        const ArrayTr<T_real>* background,
                                        Gen_Func_Def<T_real> gen_func,
                                        bool use_weights)
{
    return OPTIMIZER_OUTCOME::FAILED;
}

// ------------------------------------------------------------ 

template<typename T_real>
OPTIMIZER_OUTCOME TAO_BRGN_Optimizer<T_real>::minimize_quantification(Fit_Parameters<T_real>*fit_params,
                                                std::unordered_map<std::string, Element_Quant<T_real>*> * quant_map,
                                                quantification::models::Quantification_Model<T_real>* quantification_model)
{
    return OPTIMIZER_OUTCOME::FAILED;
}

// ------------------------------------------------------------ 

template<typename T_real>
std::unordered_map<std::string, T_real> TAO_BRGN_Optimizer<T_real>::get_options()
{
    std::unordered_map<std::string, T_real> options;
    return options;
}

// ------------------------------------------------------------ 

template<typename T_real>
void TAO_BRGN_Optimizer<T_real>::set_options(std::unordered_map<std::string, T_real> opt)
{

}

// ------------------------------------------------------------ 

template<typename T_real>
std::string TAO_BRGN_Optimizer<T_real>::detailed_outcome(int outcome)
{
    return "N/A";
}

// ------------------------------------------------------------ 


TEMPLATE_CLASS_DLL_EXPORT TAO_BRGN_Optimizer<float>;
TEMPLATE_CLASS_DLL_EXPORT TAO_BRGN_Optimizer<double>;

 }//namespace optimizers
 }//namespace fitting