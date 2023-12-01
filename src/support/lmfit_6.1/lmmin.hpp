/*
 * Library:   lmfit (Levenberg-Marquardt least squares fitting)
 *
 * File:      lmmin.h
 *
 * Contents:  Declarations for Levenberg-Marquardt minimization.
 *
 * Copyright: Joachim Wuttke, Forschungszentrum Juelich GmbH (2004-2013)
 *
 * License:   see ../COPYING (FreeBSD)
 *
 * Homepage:  apps.jcns.fz-juelich.de/lmfit
 */

#ifndef LMMIN_H
#define LMMIN_H

#include "lmstruct.hpp"
#include <assert.h>
#include <float.h>

/******************************************************************************/
/*  Numeric constants                                                         */
/******************************************************************************/

/* Set machine-dependent constants to values from float.h. */

/* If the above values do not work, the following seem good for an x86:
 LM_MACHEP     .555e-16
 LM_DWARF      9.9e-324
 LM_SQRT_DWARF 1.e-160
 LM_SQRT_GIANT 1.e150
 LM_USER_TOL   1.e-14
   The following values should work on any machine:
 LM_MACHEP     1.2e-16
 LM_DWARF      1.0e-38
 LM_SQRT_DWARF 3.834e-20
 LM_SQRT_GIANT 1.304e19
 LM_USER_TOL   1.e-14
*/

#define DP_LM_MACHEP DBL_EPSILON       /* resolution of arithmetic */
#define DP_LM_DWARF DBL_MIN            /* smallest nonzero number */
#define DP_LM_SQRT_DWARF sqrt(DBL_MIN) /* square should not underflow */
#define DP_LM_SQRT_GIANT sqrt(DBL_MAX) /* square should not overflow */
#define DP_LM_USERTOL 30 * DP_LM_MACHEP   /* users are recommended to require this */
#define DP_LM_EPSILON DBL_EPSILON  

#define FP_LM_MACHEP 1.19209e-10f       // resolution of arithmetic 
#define FP_LM_DWARF 1.17549e-38f           // smallest nonzero number 
#define FP_LM_GIANT   3.40282e+38f
#define FP_LM_SQRT_DWARF (sqrt(FP_LM_DWARF*1.5f)*10.0f) // square should not underflow 
#define FP_LM_SQRT_GIANT (sqrt(FP_LM_GIANT)*0.1f) // square should not overflow 
#define FP_LM_USERTOL 1.19209e-10f  // users are recommended to require this 
#define FP_LM_EPSILON 1.19209e-7f  // MPFit's epsilon


#define MIN(a, b) (((a) <= (b)) ? (a) : (b))
#define MAX(a, b) (((a) >= (b)) ? (a) : (b))
#define SQR(x) (x) * (x)

/******************************************************************************/
/*  Monitoring auxiliaries.                                                   */
/******************************************************************************/
template <typename _T>
void lm_print_pars(int nout, const _T* par, FILE* fout)
{
    int i;
    for (i = 0; i < nout; ++i)
        fprintf(fout, " %16.9g", par[i]);
    fprintf(fout, "\n");
}

/******************************************************************************/
/*  lm_enorm (Euclidean norm)                                                 */
/******************************************************************************/
/* Refined calculation of Eucledian norm. */
template <typename _T>
_T lm_enorm(int n, const _T* x)
/*     This function calculates the Euclidean norm of an n-vector x.
 *
 *     The Euclidean norm is computed by accumulating the sum of squares
 *     in three different sums. The sums of squares for the small and large
 *     components are scaled so that no overflows occur. Non-destructive
 *     underflows are permitted. Underflows and overflows do not occur in
 *     the computation of the unscaled sum of squares for the intermediate
 *     components. The definitions of small, intermediate and large components
 *     depend on two constants, LM_SQRT_DWARF and LM_SQRT_GIANT. The main
 *     restrictions on these constants are that LM_SQRT_DWARF**2 not underflow
 *     and LM_SQRT_GIANT**2 not overflow.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable.
 *
 *      x is an INPUT array of length n.
 */
{
    int i;
    _T agiant, s1, s2, s3, xabs, x1max, x3max;

    _T LM_SQRT_DWARF;
    _T LM_SQRT_GIANT;
    if (std::is_same<_T, float>::value)
    {
        LM_SQRT_DWARF = FP_LM_SQRT_DWARF;
        LM_SQRT_GIANT = FP_LM_SQRT_GIANT;
    }
    else if (std::is_same<_T, double>::value)
    {
        LM_SQRT_DWARF = DP_LM_SQRT_DWARF;
        LM_SQRT_GIANT = DP_LM_SQRT_GIANT;
    }

    s1 = 0;
    s2 = 0;
    s3 = 0;
    x1max = 0;
    x3max = 0;
    agiant = LM_SQRT_GIANT / n;

    /** Sum squares. **/
    for (i = 0; i < n; i++) {
        xabs = std::fabs(x[i]);
        if (xabs > LM_SQRT_DWARF) {
            if (xabs < agiant) {
                s2 += SQR(xabs);
            } else if (xabs > x1max) {
                s1 = 1 + s1 * SQR(x1max / xabs);
                x1max = xabs;
            } else {
                s1 += SQR(xabs / x1max);
            }
        } else if (xabs > x3max) {
            s3 = 1 + s3 * SQR(x3max / xabs);
            x3max = xabs;
        } else if (xabs != 0) {
            s3 += SQR(xabs / x3max);
        }
    }

    /** Calculate the norm. **/
    if (s1 != 0)
        return x1max * sqrt(s1 + (s2 / x1max) / x1max);
    else if (s2 != 0)
        if (s2 >= x3max)
            return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
        else
            return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
    else
        return x3max * sqrt(s3);

} /*** lm_enorm. ***/

/******************************************************************************/
/*  lm_qrsolv (linear least-squares)                                          */
/******************************************************************************/
template <typename _T>
void lm_qrsolv(const int n, _T* r, const int ldr, const int* Pivot,
               const _T* diag, const _T* qtb, _T* x,
               _T* Sdiag, _T* W)
/*
 *     Given an m by n matrix A, an n by n diagonal matrix D, and an
 *     m-vector b, the problem is to determine an x which solves the
 *     system
 *
 *          A*x = b  and  D*x = 0
 *
 *     in the least squares sense.
 *
 *     This subroutine completes the solution of the problem if it is
 *     provided with the necessary information from the QR factorization,
 *     with column pivoting, of A. That is, if A*P = Q*R, where P is a
 *     permutation matrix, Q has orthogonal columns, and R is an upper
 *     triangular matrix with diagonal elements of nonincreasing magnitude,
 *     then qrsolv expects the full upper triangle of R, the permutation
 *     matrix P, and the first n components of Q^T*b. The system
 *     A*x = b, D*x = 0, is then equivalent to
 *
 *          R*z = Q^T*b,  P^T*D*P*z = 0,
 *
 *     where x = P*z. If this system does not have full rank, then a least
 *     squares solution is obtained. On output qrsolv also provides an upper
 *     triangular matrix S such that
 *
 *          P^T*(A^T*A + D*D)*P = S^T*S.
 *
 *     S is computed within qrsolv and may be of separate interest.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable set to the order of R.
 *
 *      r is an n by n array. On INPUT the full upper triangle must contain
 *        the full upper triangle of the matrix R. On OUTPUT the full upper
 *        triangle is unaltered, and the strict lower triangle contains the
 *        strict upper triangle (transposed) of the upper triangular matrix S.
 *
 *      ldr is a positive integer INPUT variable not less than n which
 *        specifies the leading dimension of the array R.
 *
 *      Pivot is an integer INPUT array of length n which defines the
 *        permutation matrix P such that A*P = Q*R. Column j of P is column
 *        Pivot(j) of the identity matrix.
 *
 *      diag is an INPUT array of length n which must contain the diagonal
 *        elements of the matrix D.
 *
 *      qtb is an INPUT array of length n which must contain the first
 *        n elements of the vector Q^T*b.
 *
 *      x is an OUTPUT array of length n which contains the least-squares
 *        solution of the system A*x = b, D*x = 0.
 *
 *      Sdiag is an OUTPUT array of length n which contains the diagonal
 *        elements of the upper triangular matrix S.
 *
 *      W is a work array of length n.
 *
 */
{
    int i, kk, j, k, nsing;
    _T qtbpj, sum, temp;
    _T _sin, _cos, _tan, _cot; /* local variables, not functions */

    /*** Copy R and Q^T*b to preserve input and initialize S.
         In particular, save the diagonal elements of R in x. ***/

    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++)
            r[j*ldr+i] = r[i*ldr+j];
        x[j] = r[j*ldr+j];
        W[j] = qtb[j];
    }

    /*** Eliminate the diagonal matrix D using a Givens rotation. ***/

    for (j = 0; j < n; j++) {

        /*** Prepare the row of D to be eliminated, locating the diagonal
             element using P from the QR factorization. ***/

        if (diag[Pivot[j]] != 0) {
            for (k = j; k < n; k++)
                Sdiag[k] = 0;
            Sdiag[j] = diag[Pivot[j]];

            /*** The transformations to eliminate the row of D modify only
                 a single element of Q^T*b beyond the first n, which is
                 initially 0. ***/

            qtbpj = 0;
            for (k = j; k < n; k++) {

                /** Determine a Givens rotation which eliminates the
                    appropriate element in the current row of D. **/
                if (Sdiag[k] == 0)
                    continue;
                kk = k + ldr * k;
                if (std::fabs(r[kk]) < std::fabs(Sdiag[k])) {
                    _cot = r[kk] / Sdiag[k];
                    _sin = 1 / std::hypot((_T)1, _cot);
                    _cos = _sin * _cot;
                } else {
                    _tan = Sdiag[k] / r[kk];
                    _cos = 1 / std::hypot((_T)1, _tan);
                    _sin = _cos * _tan;
                }

                /** Compute the modified diagonal element of R and
                    the modified element of (Q^T*b,0). **/
                r[kk] = _cos * r[kk] + _sin * Sdiag[k];
                temp = _cos * W[k] + _sin * qtbpj;
                qtbpj = -_sin * W[k] + _cos * qtbpj;
                W[k] = temp;

                /** Accumulate the tranformation in the row of S. **/
                for (i = k+1; i < n; i++) {
                    temp = _cos * r[k*ldr+i] + _sin * Sdiag[i];
                    Sdiag[i] = -_sin * r[k*ldr+i] + _cos * Sdiag[i];
                    r[k*ldr+i] = temp;
                }
            }
        }

        /** Store the diagonal element of S and restore
            the corresponding diagonal element of R. **/
        Sdiag[j] = r[j*ldr+j];
        r[j*ldr+j] = x[j];
    }

    /*** Solve the triangular system for z. If the system is singular, then
        obtain a least-squares solution. ***/

    nsing = n;
    for (j = 0; j < n; j++) {
        if (Sdiag[j] == 0 && nsing == n)
            nsing = j;
        if (nsing < n)
            W[j] = 0;
    }

    for (j = nsing-1; j >= 0; j--) {
        sum = 0;
        for (i = j+1; i < nsing; i++)
            sum += r[j*ldr+i] * W[i];
        W[j] = (W[j] - sum) / Sdiag[j];
    }

    /*** Permute the components of z back to components of x. ***/

    for (j = 0; j < n; j++)
        x[Pivot[j]] = W[j];

} /*** lm_qrsolv. ***/

/******************************************************************************/
/*  lm_lmpar (determine Levenberg-Marquardt parameter)                        */
/******************************************************************************/
/* Declare functions that do the heavy numerics.
   Implementions are in this source file, below lmmin.
   Dependences: lmmin calls lmpar, which calls qrfac and qrsolv. */
template <typename _T>
void lm_lmpar(const int n, _T* r, const int ldr, const int* Pivot,
              const _T* diag, const _T* qtb, const _T delta,
              _T* par, _T* x, _T* Sdiag, _T* aux, _T* xdi)
/*     Given an m by n matrix A, an n by n nonsingular diagonal matrix D,
 *     an m-vector b, and a positive number delta, the problem is to
 *     determine a parameter value par such that if x solves the system
 *
 *          A*x = b  and  sqrt(par)*D*x = 0
 *
 *     in the least squares sense, and dxnorm is the Euclidean norm of D*x,
 *     then either par=0 and (dxnorm-delta) < 0.1*delta, or par>0 and
 *     abs(dxnorm-delta) < 0.1*delta.
 *
 *     Using lm_qrsolv, this subroutine completes the solution of the
 *     problem if it is provided with the necessary information from the
 *     QR factorization, with column pivoting, of A. That is, if A*P = Q*R,
 *     where P is a permutation matrix, Q has orthogonal columns, and R is
 *     an upper triangular matrix with diagonal elements of nonincreasing
 *     magnitude, then lmpar expects the full upper triangle of R, the
 *     permutation matrix P, and the first n components of Q^T*b. On output
 *     lmpar also provides an upper triangular matrix S such that
 *
 *          P^T*(A^T*A + par*D*D)*P = S^T*S.
 *
 *     S is employed within lmpar and may be of separate interest.
 *
 *     Only a few iterations are generally needed for convergence of the
 *     algorithm. If, however, the limit of 10 iterations is reached, then
 *     the output par will contain the best value obtained so far.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable set to the order of r.
 *
 *      r is an n by n array. On INPUT the full upper triangle must contain
 *        the full upper triangle of the matrix R. On OUTPUT the full upper
 *        triangle is unaltered, and the strict lower triangle contains the
 *        strict upper triangle (transposed) of the upper triangular matrix S.
 *
 *      ldr is a positive integer INPUT variable not less than n which
 *        specifies the leading dimension of the array R.
 *
 *      Pivot is an integer INPUT array of length n which defines the
 *        permutation matrix P such that A*P = Q*R. Column j of P is column
 *        Pivot(j) of the identity matrix.
 *
 *      diag is an INPUT array of length n which must contain the diagonal
 *        elements of the matrix D.
 *
 *      qtb is an INPUT array of length n which must contain the first
 *        n elements of the vector Q^T*b.
 *
 *      delta is a positive INPUT variable which specifies an upper bound
 *        on the Euclidean norm of D*x.
 *
 *      par is a nonnegative variable. On INPUT par contains an initial
 *        estimate of the Levenberg-Marquardt parameter. On OUTPUT par
 *        contains the final estimate.
 *
 *      x is an OUTPUT array of length n which contains the least-squares
 *        solution of the system A*x = b, sqrt(par)*D*x = 0, for the output par.
 *
 *      Sdiag is an array of length n needed as workspace; on OUTPUT it
 *        contains the diagonal elements of the upper triangular matrix S.
 *
 *      aux is a multi-purpose work array of length n.
 *
 *      xdi is a work array of length n. On OUTPUT: diag[j] * x[j].
 *
 */
{
    int i, iter, j, nsing;
    _T dxnorm, fp, fp_old, gnorm, parc, parl, paru;
    _T sum, temp;
    static _T p1 = (_T)0.1;

    _T LM_DWARF;
    if (std::is_same<_T, float>::value)
    {
        LM_DWARF = (_T)FP_LM_DWARF;
    }
    else if (std::is_same<_T, double>::value)
    {
        LM_DWARF = (_T)DP_LM_DWARF;
    }

    /*** Compute and store in x the Gauss-Newton direction. If the Jacobian
         is rank-deficient, obtain a least-squares solution. ***/

    nsing = n;
    for (j = 0; j < n; j++) {
        aux[j] = qtb[j];
        if (r[j*ldr+j] == 0 && nsing == n)
            nsing = j;
        if (nsing < n)
            aux[j] = 0;
    }
    for (j = nsing-1; j >= 0; j--) {
        aux[j] = aux[j] / r[j+ldr*j];
        temp = aux[j];
        for (i = 0; i < j; i++)
            aux[i] -= r[j*ldr+i] * temp;
    }

    for (j = 0; j < n; j++)
        x[Pivot[j]] = aux[j];

    /*** Initialize the iteration counter, evaluate the function at the origin,
         and test for acceptance of the Gauss-Newton direction. ***/

    for (j = 0; j < n; j++)
        xdi[j] = diag[j] * x[j];
    dxnorm = lm_enorm(n, xdi);
    fp = dxnorm - delta;
    if (fp <= p1 * delta) {
#ifdef LMFIT_DEBUG_MESSAGES
        printf("debug lmpar nsing=%d, n=%d, terminate[fp<=p1*del]\n", nsing, n);
#endif
        *par = 0;
        return;
    }

    /*** If the Jacobian is not rank deficient, the Newton step provides a
         lower bound, parl, for the zero of the function. Otherwise set this
         bound to zero. ***/

    parl = 0;
    if (nsing >= n) {
        for (j = 0; j < n; j++)
            aux[j] = diag[Pivot[j]] * xdi[Pivot[j]] / dxnorm;

        for (j = 0; j < n; j++) {
            sum = 0;
            for (i = 0; i < j; i++)
                sum += r[j*ldr+i] * aux[i];
            aux[j] = (aux[j] - sum) / r[j+ldr*j];
        }
        temp = lm_enorm(n, aux);
        parl = fp / delta / temp / temp;
    }

    /*** Calculate an upper bound, paru, for the zero of the function. ***/

    for (j = 0; j < n; j++) {
        sum = 0;
        for (i = 0; i <= j; i++)
            sum += r[j*ldr+i] * qtb[i];
        aux[j] = sum / diag[Pivot[j]];
    }
    gnorm = lm_enorm(n, aux);
    paru = gnorm / delta;
    if (paru == 0)
        paru = LM_DWARF / MIN(delta, p1);

    /*** If the input par lies outside of the interval (parl,paru),
         set par to the closer endpoint. ***/

    *par = MAX(*par, parl);
    *par = MIN(*par, paru);
    if (*par == 0)
        *par = gnorm / dxnorm;

    /*** Iterate. ***/

    for (iter = 0;; iter++) {

        /** Evaluate the function at the current value of par. **/
        if (*par == 0)
            *par = MAX(LM_DWARF, (_T)0.001 * paru);
        temp = sqrt(*par);
        for (j = 0; j < n; j++)
            aux[j] = temp * diag[j];

        lm_qrsolv(n, r, ldr, Pivot, aux, qtb, x, Sdiag, xdi);
        /* return values are r, x, Sdiag */

        for (j = 0; j < n; j++)
            xdi[j] = diag[j] * x[j]; /* used as output */
        dxnorm = lm_enorm(n, xdi);
        fp_old = fp;
        fp = dxnorm - delta;

        /** If the function is small enough, accept the current value
            of par. Also test for the exceptional cases where parl
            is zero or the number of iterations has reached 10. **/
        if (std::fabs(fp) <= p1 * delta ||
            (parl == 0 && fp <= fp_old && fp_old < 0) || iter == 10) {
#ifdef LMFIT_DEBUG_MESSAGES
            printf("debug lmpar nsing=%d, iter=%d, "
                   "par=%.4e [%.4e %.4e], delta=%.4e, fp=%.4e\n",
                   nsing, iter, *par, parl, paru, delta, fp);
#endif
            break; /* the only exit from the iteration. */
        }

        /** Compute the Newton correction. **/
        for (j = 0; j < n; j++)
            aux[j] = diag[Pivot[j]] * xdi[Pivot[j]] / dxnorm;

        for (j = 0; j < n; j++) {
            aux[j] = aux[j] / Sdiag[j];
            for (i = j+1; i < n; i++)
                aux[i] -= r[j*ldr+i] * aux[j];
        }
        temp = lm_enorm(n, aux);
        parc = fp / delta / temp / temp;

        /** Depending on the sign of the function, update parl or paru. **/
        if (fp > 0)
            parl = MAX(parl, *par);
        else /* fp < 0 [the case fp==0 is precluded by the break condition] */
            paru = MIN(paru, *par);

        /** Compute an improved estimate for par. **/
        *par = MAX(parl, *par + parc);
    }

} /*** lm_lmpar. ***/


/******************************************************************************/
/*  lm_qrfac (QR factorization, from lapack)                                  */
/******************************************************************************/
template <typename _T>
void lm_qrfac(const int m, const int n, _T* A, int* Pivot, _T* Rdiag,
              _T* Acnorm, _T* W)
/*
 *     This subroutine uses Householder transformations with column pivoting
 *     to compute a QR factorization of the m by n matrix A. That is, qrfac
 *     determines an orthogonal matrix Q, a permutation matrix P, and an
 *     upper trapezoidal matrix R with diagonal elements of nonincreasing
 *     magnitude, such that A*P = Q*R. The Householder transformation for
 *     column k, k = 1,2,...,n, is of the form
 *
 *          I - 2*w*wT/|w|^2
 *
 *     where w has zeroes in the first k-1 positions.
 *
 *     Parameters:
 *
 *      m is an INPUT parameter set to the number of rows of A.
 *
 *      n is an INPUT parameter set to the number of columns of A.
 *
 *      A is an m by n array. On INPUT, A contains the matrix for which the
 *        QR factorization is to be computed. On OUTPUT the strict upper
 *        trapezoidal part of A contains the strict upper trapezoidal part
 *        of R, and the lower trapezoidal part of A contains a factored form
 *        of Q (the non-trivial elements of the vectors w described above).
 *
 *      Pivot is an integer OUTPUT array of length n that describes the
 *        permutation matrix P. Column j of P is column Pivot(j) of the
 *        identity matrix.
 *
 *      Rdiag is an OUTPUT array of length n which contains the diagonal
 *        elements of R.
 *
 *      Acnorm is an OUTPUT array of length n which contains the norms of
 *        the corresponding columns of the input matrix A. If this information
 *        is not needed, then Acnorm can share storage with Rdiag.
 *
 *      W is a work array of length n.
 *
 */
{
    int i, j, k, kmax;
    _T ajnorm, sum, temp;

    _T LM_MACHEP;
    if (std::is_same<_T, float>::value)
    {
        LM_MACHEP = FP_LM_MACHEP;
    }
    else if (std::is_same<_T, double>::value)
    {
        LM_MACHEP = DP_LM_MACHEP;
    }

#ifdef LMFIT_DEBUG_MESSAGES
    printf("debug qrfac\n");
#endif

    /** Compute initial column norms;
        initialize Pivot with identity permutation. ***/
    for (j = 0; j < n; j++) {
        W[j] = Rdiag[j] = Acnorm[j] = lm_enorm(m, &A[j*m]);
        Pivot[j] = j;
    }

    /** Loop over columns of A. **/
    assert(n <= m);
    for (j = 0; j < n; j++) {

        /** Bring the column of largest norm into the pivot position. **/
        kmax = j;
        for (k = j+1; k < n; k++)
            if (Rdiag[k] > Rdiag[kmax])
                kmax = k;

        if (kmax != j) {
            /* Swap columns j and kmax. */
            k = Pivot[j];
            Pivot[j] = Pivot[kmax];
            Pivot[kmax] = k;
            for (i = 0; i < m; i++) {
                temp = A[j*m+i];
                A[j*m+i] = A[kmax*m+i];
                A[kmax*m+i] = temp;
            }
            /* Half-swap: Rdiag[j], W[j] won't be needed any further. */
            Rdiag[kmax] = Rdiag[j];
            W[kmax] = W[j];
        }

        /** Compute the Householder reflection vector w_j to reduce the
            j-th column of A to a multiple of the j-th unit vector. **/
        ajnorm = lm_enorm(m-j, &A[j*m+j]);
        if (ajnorm == 0) {
            Rdiag[j] = 0;
            continue;
        }

        /* Let the partial column vector A[j][j:] contain w_j := e_j+-a_j/|a_j|,
           where the sign +- is chosen to avoid cancellation in w_jj. */
        if (A[j*m+j] < 0)
            ajnorm = -ajnorm;
        for (i = j; i < m; i++)
            A[j*m+i] /= ajnorm;
        A[j*m+j] += 1;

        /** Apply the Householder transformation U_w := 1 - 2*w_j.w_j/|w_j|^2
            to the remaining columns, and update the norms. **/
        for (k = j+1; k < n; k++) {
            /* Compute scalar product w_j * a_j. */
            sum = 0;
            for (i = j; i < m; i++)
                sum += A[j*m+i] * A[k*m+i];

            /* Normalization is simplified by the coincidence |w_j|^2=2w_jj. */
            temp = sum / A[j*m+j];

            /* Carry out transform U_w_j * a_k. */
            for (i = j; i < m; i++)
                A[k*m+i] -= temp * A[j*m+i];

            /* No idea what happens here. */
            if (Rdiag[k] != 0) {
                temp = A[m*k+j] / Rdiag[k];
                if (std::fabs(temp) < 1) {
                    Rdiag[k] *= sqrt(1 - SQR(temp));
                    temp = Rdiag[k] / W[k];
                } else
                    temp = 0;
                if (temp == 0 || 0.05 * SQR(temp) <= LM_MACHEP) {
                    Rdiag[k] = lm_enorm(m-j-1, &A[m*k+j+1]);
                    W[k] = Rdiag[k];
                }
            }
        }

        Rdiag[j] = -ajnorm;
    }
} /*** lm_qrfac. ***/





/* Predefined control parameter sets (msgfile=NULL means stdout).
const lm_control_struct lm_control_double = {
    LM_USERTOL, LM_USERTOL, LM_USERTOL, LM_USERTOL,
    100., 100, 1, NULL, 0, -1, -1};
const lm_control_struct lm_control_float = {
    1.e-7, 1.e-7, 1.e-7, 1.e-7,
    100., 100, 1, NULL, 0, -1, -1};
*/
/******************************************************************************/
/*  Message texts (indexed by status.info)                                    */
/******************************************************************************/

static const char* lm_infmsg[] = {
    "found zero (sum of squares below underflow limit)",
    "converged  (the relative error in the sum of squares is at most tol)",
    "converged  (the relative error of the parameter vector is at most tol)",
    "converged  (both errors are at most tol)",
    "trapped    (by degeneracy; increasing epsilon might help)",
    "exhausted  (number of function calls exceeding preset patience)",
    "failed     (ftol < tol: cannot reduce sum of squares any further)",
    "failed     (xtol < tol: cannot improve approximate solution any further)",
    "failed     (gtol < tol: cannot improve approximate solution any further)",
    "crashed    (not enough memory)",
    "exploded   (fatal coding error: improper input parameters)",
    "stopped    (break requested within function evaluation)",
    "found nan  (function value is not-a-number or infinite)"};

static const char* lm_shortmsg[] = {
    "found zero",
    "converged (f)",
    "converged (p)",
    "converged (2)",
    "degenerate",
    "call limit",
    "failed (f)",
    "failed (p)",
    "failed (o)",
    "no memory",
    "invalid input",
    "user break",
    "found nan"};



/******************************************************************************/
/*  lmmin (main minimization routine)                                         */
/******************************************************************************/
/* Levenberg-Marquardt minimization. */
template <typename _T>
void lmmin(const int n, _T* x, const int m, const void* data,
           void (*evaluate)(const _T* par, const int m_dat,
                            const void* data, _T* fvec, int* userbreak),
           const lm_control_struct<_T>* C, lm_status_struct<_T>* S)
/*
 *   This routine contains the core algorithm of our library.
 *
 *   It minimizes the sum of the squares of m nonlinear functions
 *   in n variables by a modified Levenberg-Marquardt algorithm.
 *   The function evaluation is done by the user-provided routine 'evaluate'.
 *   The Jacobian is then calculated by a forward-difference approximation.
 *
 *   Parameters:
 *
 *      n is the number of variables (INPUT, positive integer).
 *
 *      x is the solution vector (INPUT/OUTPUT, array of length n).
 *        On input it must be set to an estimated solution.
 *        On output it yields the final estimate of the solution.
 *
 *      m is the number of functions to be minimized (INPUT, positive integer).
 *        It must fulfill m>=n.
 *
 *      data is a pointer that is ignored by lmmin; it is however forwarded
 *        to the user-supplied functions evaluate and printout.
 *        In a typical application, it contains experimental data to be fitted.
 *
 *      evaluate is a user-supplied function that calculates the m functions.
 *        Parameters:
 *          n, x, m, data as above.
 *          fvec is an array of length m; on OUTPUT, it must contain the
 *            m function values for the parameter vector x.
 *          userbreak is an integer pointer. When *userbreak is set to a
 *            nonzero value, lmmin will terminate.
 *
 *      control contains INPUT variables that control the fit algorithm,
 *        as declared and explained in lmstruct.h
 *
 *      status contains OUTPUT variables that inform about the fit result,
 *        as declared and explained in lmstruct.h
 */
{
    int j, i;
    _T actred, dirder, fnorm, fnorm1, gnorm, pnorm, prered, ratio, step,
        sum, temp, temp1, temp2, temp3;

    /***  Initialize internal variables.  ***/

    _T LM_DWARF;
    _T LM_MACHEP;
    if (std::is_same<_T, float>::value)
    {
        LM_DWARF = (_T)FP_LM_DWARF;
        LM_MACHEP = (_T)FP_LM_MACHEP;
    }
    else if (std::is_same<_T, double>::value)
    {
        LM_DWARF = (_T)DP_LM_DWARF;
        LM_MACHEP = (_T)DP_LM_MACHEP;
    }

    int maxfev = C->patience * (n+1);

    int inner_success; /* flag for loop control */
    _T lmpar = (_T)0.;  /* Levenberg-Marquardt parameter */
    _T delta = (_T)0.;
    _T xnorm = (_T)0.;
    _T eps = sqrt(MAX(C->epsilon, LM_MACHEP)); /* for forward differences */

    int nout = C->n_maxpri == -1 ? n : MIN(C->n_maxpri, n);

    /* Reinterpret C->msgfile=NULL as stdout (which is unavailable for
       compile-time initialization of lm_control and similar). */
    FILE* msgfile = C->msgfile ? C->msgfile : stdout;

    /***  Default status info; must be set before first return statement.  ***/

    S->outcome = 0; /* status code */
    S->userbreak = 0;
    S->nfev = 0; /* function evaluation counter */

    /***  Check input parameters for errors.  ***/

    if (n <= 0) {
        fprintf(stderr, "lmmin: invalid number of parameters %i\n", n);
        S->outcome = 10;
        return;
    }
    if (m < n) {
        fprintf(stderr, "lmmin: number of data points (%i) "
                        "smaller than number of parameters (%i)\n",
                m, n);
        S->outcome = 10;
        return;
    }
    if (C->ftol < 0 || C->xtol < 0 || C->gtol < 0) {
        fprintf(stderr,
                "lmmin: negative tolerance (at least one of %g %g %g)\n",
                C->ftol, C->xtol, C->gtol);
        S->outcome = 10;
        return;
    }
    if (maxfev <= 0) {
        fprintf(stderr, "lmmin: nonpositive function evaluations limit %i\n",
                maxfev);
        S->outcome = 10;
        return;
    }
    if (C->stepbound <= 0) {
        fprintf(stderr, "lmmin: nonpositive stepbound %g\n", C->stepbound);
        S->outcome = 10;
        return;
    }
    if (C->scale_diag != 0 && C->scale_diag != 1) {
        fprintf(stderr, "lmmin: logical variable scale_diag=%i, "
                        "should be 0 or 1\n",
                C->scale_diag);
        S->outcome = 10;
        return;
    }

    /***  Allocate work space.  ***/

    /* Allocate total workspace with just one system call */
    char* ws;
    if ((ws = (char*)malloc((2*m + 5*n + m*n) * sizeof(_T) + n * sizeof(int))) == NULL)
    {
        S->outcome = 9;
        return;
    }

    /* Assign workspace segments. */
    char* pws = ws;
    _T* fvec = (_T*)pws;
    pws += m * sizeof(_T) / sizeof(char);
    _T* diag = (_T*)pws;
    pws += n * sizeof(_T) / sizeof(char);
    _T* qtf = (_T*)pws;
    pws += n * sizeof(_T) / sizeof(char);
    _T* fjac = (_T*)pws;
    pws += n * m * sizeof(_T) / sizeof(char);
    _T* wa1 = (_T*)pws;
    pws += n * sizeof(_T) / sizeof(char);
    _T* wa2 = (_T*)pws;
    pws += n * sizeof(_T) / sizeof(char);
    _T* wa3 = (_T*)pws;
    pws += n * sizeof(_T) / sizeof(char);
    _T* wf = (_T*)pws;
    pws += m * sizeof(_T) / sizeof(char);
    int* Pivot = (int*)pws;
    pws += n * sizeof(int) / sizeof(char);

    /* Initialize diag. */
    if (!C->scale_diag)
        for (j = 0; j < n; j++)
            diag[j] = 1;

    /***  Evaluate function at starting point and calculate norm.  ***/

    if (C->verbosity) {
        fprintf(msgfile, "lmmin start ");
        lm_print_pars(nout, x, msgfile);
    }
    (*evaluate)(x, m, data, fvec, &(S->userbreak));
    if (C->verbosity > 4)
        for (i = 0; i < m; ++i)
            fprintf(msgfile, "    fvec[%4i] = %18.8g\n", i, fvec[i]);
    S->nfev = 1;
    if (S->userbreak)
        goto terminate;
    fnorm = lm_enorm(m, fvec);
    if (C->verbosity)
        fprintf(msgfile, "  fnorm = %18.8g\n", fnorm);

    if (!std::isfinite(fnorm)) {
        S->outcome = 12; /* nan */
        goto terminate;
    } else if (fnorm <= LM_DWARF) {
        S->outcome = 0; /* sum of squares almost zero, nothing to do */
        goto terminate;
    }

    /***  The outer loop: compute gradient, then descend.  ***/

    for (int outer = 0;; ++outer) {

        /** Calculate the Jacobian. **/
        for (j = 0; j < n; j++) {
            temp = x[j];
            step = MAX(eps * eps, eps * std::fabs(temp));
            x[j] += step; /* replace temporarily */
            (*evaluate)(x, m, data, wf, &(S->userbreak));
            ++(S->nfev);
            if (S->userbreak)
                goto terminate;
            for (i = 0; i < m; i++)
                fjac[j*m+i] = (wf[i] - fvec[i]) / step;
            x[j] = temp; /* restore */
        }
        if (C->verbosity >= 10) {
            /* print the entire matrix */
            printf("\nlmmin Jacobian\n");
            for (i = 0; i < m; i++) {
                printf("  ");
                for (j = 0; j < n; j++)
                    printf("%.5e ", fjac[j*m+i]);
                printf("\n");
            }
        }

        /** Compute the QR factorization of the Jacobian. **/

        /* fjac is an m by n array. The upper n by n submatrix of fjac is made
         *   to contain an upper triangular matrix R with diagonal elements of
         *   nonincreasing magnitude such that
         *
         *         P^T*(J^T*J)*P = R^T*R
         *
         *         (NOTE: ^T stands for matrix transposition),
         *
         *   where P is a permutation matrix and J is the final calculated
         *   Jacobian. Column j of P is column Pivot(j) of the identity matrix.
         *   The lower trapezoidal part of fjac contains information generated
         *   during the computation of R.
         *
         * Pivot is an integer array of length n. It defines a permutation
         *   matrix P such that jac*P = Q*R, where jac is the final calculated
         *   Jacobian, Q is orthogonal (not stored), and R is upper triangular
         *   with diagonal elements of nonincreasing magnitude. Column j of P
         *   is column Pivot(j) of the identity matrix.
         */

        lm_qrfac(m, n, fjac, Pivot, wa1, wa2, wa3);
        /* return values are Pivot, wa1=rdiag, wa2=acnorm */

        /** Form Q^T * fvec, and store first n components in qtf. **/
        for (i = 0; i < m; i++)
            wf[i] = fvec[i];

        for (j = 0; j < n; j++) {
            temp3 = fjac[j*m+j];
            if (temp3 != 0) {
                sum = 0;
                for (i = j; i < m; i++)
                    sum += fjac[j*m+i] * wf[i];
                temp = -sum / temp3;
                for (i = j; i < m; i++)
                    wf[i] += fjac[j*m+i] * temp;
            }
            fjac[j*m+j] = wa1[j];
            qtf[j] = wf[j];
        }

        /**  Compute norm of scaled gradient and detect degeneracy. **/
        gnorm = 0;
        for (j = 0; j < n; j++) {
            if (wa2[Pivot[j]] == 0)
                continue;
            sum = 0;
            for (i = 0; i <= j; i++)
                sum += fjac[j*m+i] * qtf[i];
            gnorm = MAX(gnorm, std::fabs(sum / wa2[Pivot[j]] / fnorm));
        }

        if (gnorm <= C->gtol) {
            S->outcome = 4;
            goto terminate;
        }

        /** Initialize or update diag and delta. **/
        if (!outer) { /* first iteration only */
            if (C->scale_diag) {
                /* diag := norms of the columns of the initial Jacobian */
                for (j = 0; j < n; j++)
                    diag[j] = wa2[j] ? wa2[j] : 1;
                /* xnorm := || D x || */
                for (j = 0; j < n; j++)
                    wa3[j] = diag[j] * x[j];
                xnorm = lm_enorm(n, wa3);
                if (C->verbosity >= 2) {
                    fprintf(msgfile, "lmmin diag  ");
                    lm_print_pars(nout, x, msgfile); // xnorm
                    fprintf(msgfile, "  xnorm = %18.8g\n", xnorm);
                }
                /* Only now print the header for the loop table. */
                if (C->verbosity >= 3) {
                    fprintf(msgfile, "  o  i     lmpar    prered"
                                     "          ratio    dirder      delta"
                                     "      pnorm                 fnorm");
                    for (i = 0; i < nout; ++i)
                        fprintf(msgfile, "               p%i", i);
                    fprintf(msgfile, "\n");
                }
            } else {
                xnorm = lm_enorm(n, x);
            }
            if (!std::isfinite(xnorm)) {
                S->outcome = 12; /* nan */
                goto terminate;
            }
            /* Initialize the step bound delta. */
            if (xnorm)
                delta = C->stepbound * xnorm;
            else
                delta = C->stepbound;
        } else {
            if (C->scale_diag) {
                for (j = 0; j < n; j++)
                    diag[j] = MAX(diag[j], wa2[j]);
            }
        }

        /** The inner loop. **/
        int inner = 0;
        do {

            /** Determine the Levenberg-Marquardt parameter. **/
            lm_lmpar(n, fjac, m, Pivot, diag, qtf, delta, &lmpar,
                     wa1, wa2, wf, wa3);
            /* used return values are fjac (partly), lmpar, wa1=x, wa3=diag*x */

            /* Predict scaled reduction. */
            pnorm = lm_enorm(n, wa3);
            if (!std::isfinite(pnorm)) {
                S->outcome = 12; /* nan */
                goto terminate;
            }
            temp2 = lmpar * SQR(pnorm / fnorm);
            for (j = 0; j < n; j++) {
                wa3[j] = 0;
                for (i = 0; i <= j; i++)
                    wa3[i] -= fjac[j*m+i] * wa1[Pivot[j]];
            }
            temp1 = SQR(lm_enorm(n, wa3) / fnorm);
            if (!std::isfinite(temp1)) {
                S->outcome = 12; /* nan */
                goto terminate;
            }
            prered = temp1 + 2*temp2;
            dirder = -temp1 + temp2; /* scaled directional derivative */

            /* At first call, adjust the initial step bound. */
            if (!outer && pnorm < delta)
                delta = pnorm;

            /** Evaluate the function at x + p. **/
            for (j = 0; j < n; j++)
                wa2[j] = x[j] - wa1[j];
            (*evaluate)(wa2, m, data, wf, &(S->userbreak));
            ++(S->nfev);
            if (S->userbreak)
                goto terminate;
            fnorm1 = lm_enorm(m, wf);
            if (!std::isfinite(fnorm1)) {
                S->outcome = 12; /* nan */
                goto terminate;
            }

            /** Evaluate the scaled reduction. **/

            /* Actual scaled reduction. */
            actred = 1 - SQR(fnorm1 / fnorm);

            /* Ratio of actual to predicted reduction. */
            ratio = prered ? actred / prered : 0;

            if (C->verbosity == 2) {
                fprintf(msgfile, "lmmin (%i:%i) ", outer, inner);
                lm_print_pars(nout, wa2, msgfile); // fnorm1,
            } else if (C->verbosity >= 3) {
                printf("%3i %2i %9.2g %9.2g %14.6g"
                       " %9.2g %10.3e %10.3e %21.15e",
                       outer, inner, lmpar, prered, ratio,
                       dirder, delta, pnorm, fnorm1);
                for (i = 0; i < nout; ++i)
                    fprintf(msgfile, " %16.9g", wa2[i]);
                fprintf(msgfile, "\n");
            }

            /* Update the step bound. */
            if (ratio <= 0.25) {
                if (actred >= 0)
                    temp = 0.5;
                else if (actred > -99) /* -99 = 1-1/0.1^2 */
                    temp = MAX(dirder / ((_T)2*dirder + actred), (_T)0.1);
                else
                    temp = (_T)0.1;
                delta = temp * MIN(delta, pnorm / (_T)0.1);
                lmpar /= temp;
            } else if (ratio >= 0.75) {
                delta = 2 * pnorm;
                lmpar *= 0.5;
            } else if (!lmpar) {
                delta = 2 * pnorm;
            }

            /**  On success, update solution, and test for convergence. **/

            inner_success = ratio >= 1e-4;
            if (inner_success) {

                /* Update x, fvec, and their norms. */
                if (C->scale_diag) {
                    for (j = 0; j < n; j++) {
                        x[j] = wa2[j];
                        wa2[j] = diag[j] * x[j];
                    }
                } else {
                    for (j = 0; j < n; j++)
                        x[j] = wa2[j];
                }
                for (i = 0; i < m; i++)
                    fvec[i] = wf[i];
                xnorm = lm_enorm(n, wa2);
                if (!std::isfinite(xnorm)) {
                    S->outcome = 12; /* nan */
                    goto terminate;
                }
                fnorm = fnorm1;
            }

            /* Convergence tests. */
            S->outcome = 0;
            if (fnorm <= LM_DWARF)
                goto terminate; /* success: sum of squares almost zero */
            /* Test two criteria (both may be fulfilled). */
            if (std::fabs(actred) <= C->ftol && prered <= C->ftol && ratio <= 2)
                S->outcome = 1; /* success: x almost stable */
            if (delta <= C->xtol * xnorm)
                S->outcome += 2; /* success: sum of squares almost stable */
            if (S->outcome != 0) {
                goto terminate;
            }

            /** Tests for termination and stringent tolerances. **/
            if (S->nfev >= maxfev) {
                S->outcome = 5;
                goto terminate;
            }
            if (std::fabs(actred) <= LM_MACHEP && prered <= LM_MACHEP &&
                ratio <= 2) {
                S->outcome = 6;
                goto terminate;
            }
            if (delta <= LM_MACHEP * xnorm) {
                S->outcome = 7;
                goto terminate;
            }
            if (gnorm <= LM_MACHEP) {
                S->outcome = 8;
                goto terminate;
            }

            /** End of the inner loop. Repeat if iteration unsuccessful. **/
            ++inner;
        } while (!inner_success);

    }; /***  End of the outer loop.  ***/

terminate:
    S->fnorm = lm_enorm(m, fvec);
    if (C->verbosity >= 2)
        printf("lmmin outcome (%i) xnorm %g ftol %g xtol %g\n", S->outcome,
               xnorm, C->ftol, C->xtol);
    if (C->verbosity & 1) {
        fprintf(msgfile, "lmmin final ");
        lm_print_pars(nout, x, msgfile); // S->fnorm,
        fprintf(msgfile, "  fnorm = %18.8g\n", S->fnorm);
    }
    if (S->userbreak) /* user-requested break */
        S->outcome = 11;

    /***  Deallocate the workspace.  ***/
    free(ws);

} /*** lmmin. ***/


#endif /* LMMIN_H */
