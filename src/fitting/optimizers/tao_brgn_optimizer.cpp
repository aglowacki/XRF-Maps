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

using namespace data_struct;


namespace fitting
{
namespace optimizers
{

// ------------------------------------------------------------ 

PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec res, void *ptr)
{
    PetscFunctionBegin;
    bool first = true;
    User_Data<double>* ud = (User_Data<double>*)(ptr);
    // Debug to find which param changed last
    Fit_Parameters<double> prev_fit_p;
    prev_fit_p.update_and_add_values(ud->fit_parameters);

    int ps = ud->fit_parameters->size();
    int* pin = new int[ps];
    for (int i = 0; i < ps; ++i) {
        pin[i] = i;
    }
    double *new_params = new double[ps];
    VecGetValues(res, ps, pin, new_params);
    delete new_params;

    // Update fit parameters from optimizer
    ud->fit_parameters->from_array(new_params, ps);
    // Model spectra based on new fit parameters
    update_background_user_data(ud);
    ud->spectra_model = ud->fit_model->model_spectrum_mp(ud->fit_parameters, ud->elements, ud->energy_range);
    // Add background
    ud->spectra_model += ud->spectra_background;
    // Remove nan's and inf's
    ud->spectra_model = (ArrayTr<double>)ud->spectra_model.unaryExpr([ud](double v) { return std::isfinite(v) ? v : ud->normalizer; });

    ArrayTr<double> diff = ud->spectra - ud->spectra_model;
    diff = diff.pow(2.0);
    diff *= ud->weights;
    int len = diff.size();

    int* indices = new int[len];
    for (int i = 0; i < len; ++i) {
        indices[i] = i;
    }

    VecSetValues(res, len, indices, diff.data(), INSERT_VALUES);
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

PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
    /* Jacobian is not changing here, so use a empty dummy function here.  J[m][n] = df[m]/dx[n] = A[m][n] for linear least square */
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
    /* Hessian for regularizer objective = 0.5*x'*x is identity matrix, and is not changing*/
    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}

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

    Vec  res, x, xlb, xub; //b, xGT,
    PetscReal   hist[100], resid[100], v1, v2;
    Mat  A, Hreg;
    PetscInt    lits[100], M, N;

    std::vector<double> fitp_arr = fit_params->to_array();
    if (fitp_arr.size() == 0)
    {
        return OPTIMIZER_OUTCOME::STOPPED;
    }
    int total_itr = 2000;
    fill_user_data(ud, fit_params, spectra, elements_to_fit, model, energy_range, status_callback, total_itr, use_weights);

    std::vector<T_real> lower_bound = fit_params->lower_to_array();
    std::vector<T_real> upper_bound = fit_params->upper_to_array();
    int len = fit_params->size();
    N = len;
    int* indices = new int[len];
    for (int i = 0; i < len; ++i)
    {
        indices[i] = i;
    }

    int slen = ud.spectra.size();
    M = slen;
    int* sindices = new int[slen];
    for (int i = 0; i < len; ++i) 
    {
        indices[i] = i;
    }

    Tao tao;
 ///   PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
    // Create TAO solver and set desired solution method 
    PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
    PetscCall(TaoSetType(tao, TAOBRGN));


    // User set application context: A, D matrice, and b vector. 
    //PetscCall(InitializeUserData(&user));

    // Allocate solution vector x,  and function vectors Ax-b, 
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &xlb));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &xub));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, M, &res));

    VecSetValues(xlb, len, indices, &lower_bound[0], INSERT_VALUES);
    VecSetValues(xub, len, indices, &upper_bound[0], INSERT_VALUES);
    VecSetValues(x, len, indices, &fitp_arr[0], INSERT_VALUES);
    VecSetValues(res, slen, sindices, ud.spectra.data(), INSERT_VALUES);


    // Set initial guess 
    //PetscCall(FormStartingPoint(x, &user));

    // Bind x to tao->solution. 
    PetscCall(TaoSetSolution(tao, x));
    // Sets the upper and lower bounds of x 
    PetscCall(TaoSetVariableBounds(tao, xlb, xub));

    // Bind user.D to tao->data->D 
    //PetscCall(TaoBRGNSetDictionaryMatrix(tao, user.D));
    PetscCall(TaoBRGNSetDictionaryMatrix(tao, nullptr));

    // Set the residual function and Jacobian routines for least squares. 
    PetscCall(TaoSetResidualRoutine(tao, res, EvaluateResidual, (void *)&ud));
    // Jacobian matrix fixed as user.A for Linear least square problem. 
    PetscCall(TaoSetJacobianResidualRoutine(tao, A, A, EvaluateJacobian, (void *)&ud));

    // User set the regularizer objective, gradient, and hessian. Set it the same as using l2prox choice, for testing purpose.  
    PetscCall(TaoBRGNSetRegularizerObjectiveAndGradientRoutine(tao, EvaluateRegularizerObjectiveAndGradient, (void *)&ud));
    // User defined regularizer Hessian setup, here is identity shell matrix 
    PetscCall(MatCreate(PETSC_COMM_SELF, &Hreg));
    PetscCall(MatSetSizes(Hreg, PETSC_DECIDE, PETSC_DECIDE, N, N));
    PetscCall(MatSetType(Hreg, MATSHELL));
    PetscCall(MatSetUp(Hreg));
    PetscCall(MatShellSetOperation(Hreg, MATOP_MULT, (void (*)(void))EvaluateRegularizerHessianProd));
    PetscCall(TaoBRGNSetRegularizerHessianRoutine(tao, Hreg, EvaluateRegularizerHessian, (void *)&ud));

    // Check for any TAO command line arguments 
    PetscCall(TaoSetFromOptions(tao));

    PetscCall(TaoSetConvergenceHistory(tao, hist, resid, 0, lits, 100, PETSC_TRUE));

    // Perform the Solve 
    PetscCall(TaoSolve(tao));

    // Save x (reconstruction of object) vector to a binary file, which maybe read from MATLAB and convert to a 2D image for comparison. 
    //PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, resultFile, FILE_MODE_WRITE, &fd));
    //PetscCall(VecView(x, fd));
    //PetscCall(PetscViewerDestroy(&fd));

    // compute the error 
    //PetscCall(VecAXPY(x, -1, xGT));
    //PetscCall(VecNorm(x, NORM_2, &v1));
    //PetscCall(VecNorm(xGT, NORM_2, &v2));
    //logI<< "relative reconstruction error: ||x-xGT||/||xGT|| = "<< (double)(v1 / v2)<<"\n";

    // Free TAO data structures 
    PetscCall(TaoDestroy(&tao));

    // Free PETSc data structures 
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&res));
    PetscCall(MatDestroy(&Hreg));
    // Free user data structures 
    //PetscCall(MatDestroy(&A));
    //PetscCall(MatDestroy(&D));
    //PetscCall(VecDestroy(&b));
    //PetscCall(VecDestroy(&xGT));
    PetscCall(VecDestroy(&xlb));
    PetscCall(VecDestroy(&xub));
    PetscCall(PetscFinalize());

    return OPTIMIZER_OUTCOME::FAILED;
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

 }//namespace optimizers
 }//namespace fitting