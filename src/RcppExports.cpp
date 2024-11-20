// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "aghqmm_types.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// optimizeaghq
List optimizeaghq(Vec theta, std::vector<Vec> y, std::vector<Mat> X, std::vector<Mat> Z, Mat nn, Vec ww, List control);
RcppExport SEXP _aghqmm_optimizeaghq(SEXP thetaSEXP, SEXP ySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP nnSEXP, SEXP wwSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<Vec> >::type y(ySEXP);
    Rcpp::traits::input_parameter< std::vector<Mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::vector<Mat> >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< Mat >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< Vec >::type ww(wwSEXP);
    Rcpp::traits::input_parameter< List >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(optimizeaghq(theta, y, X, Z, nn, ww, control));
    return rcpp_result_gen;
END_RCPP
}
// optimizeaghqscalar
List optimizeaghqscalar(Vec theta, std::vector<Vec> y, std::vector<Mat> X, Vec nn, Vec ww, List control);
RcppExport SEXP _aghqmm_optimizeaghqscalar(SEXP thetaSEXP, SEXP ySEXP, SEXP XSEXP, SEXP nnSEXP, SEXP wwSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<Vec> >::type y(ySEXP);
    Rcpp::traits::input_parameter< std::vector<Mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< Vec >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< Vec >::type ww(wwSEXP);
    Rcpp::traits::input_parameter< List >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(optimizeaghqscalar(theta, y, X, nn, ww, control));
    return rcpp_result_gen;
END_RCPP
}
// optimizegammscalar
List optimizegammscalar(Vec theta, std::vector<Vec> y, std::vector<Mat> X, Mat S, Vec nn, Vec ww, List control);
RcppExport SEXP _aghqmm_optimizegammscalar(SEXP thetaSEXP, SEXP ySEXP, SEXP XSEXP, SEXP SSEXP, SEXP nnSEXP, SEXP wwSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<Vec> >::type y(ySEXP);
    Rcpp::traits::input_parameter< std::vector<Mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< Mat >::type S(SSEXP);
    Rcpp::traits::input_parameter< Vec >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< Vec >::type ww(wwSEXP);
    Rcpp::traits::input_parameter< List >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(optimizegammscalar(theta, y, X, S, nn, ww, control));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_aghqmm_optimizeaghq", (DL_FUNC) &_aghqmm_optimizeaghq, 7},
    {"_aghqmm_optimizeaghqscalar", (DL_FUNC) &_aghqmm_optimizeaghqscalar, 6},
    {"_aghqmm_optimizegammscalar", (DL_FUNC) &_aghqmm_optimizegammscalar, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_aghqmm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
