// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// optimizeaghq
List optimizeaghq(Eigen::VectorXd theta, Eigen::VectorXd u, std::vector<Eigen::VectorXd> y, std::vector<Eigen::MatrixXd> X, std::vector<Eigen::MatrixXd> Z, Eigen::MatrixXd nn, Eigen::VectorXd ww, List control);
RcppExport SEXP _aghqmm_optimizeaghq(SEXP thetaSEXP, SEXP uSEXP, SEXP ySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP nnSEXP, SEXP wwSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::vector<Eigen::VectorXd> >::type y(ySEXP);
    Rcpp::traits::input_parameter< std::vector<Eigen::MatrixXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::vector<Eigen::MatrixXd> >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type ww(wwSEXP);
    Rcpp::traits::input_parameter< List >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(optimizeaghq(theta, u, y, X, Z, nn, ww, control));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_aghqmm_optimizeaghq", (DL_FUNC) &_aghqmm_optimizeaghq, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_aghqmm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}