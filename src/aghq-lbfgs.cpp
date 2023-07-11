// AGHQ marginal likelihood and gradient

/** Include */
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
#include <Eigen/Core>
#include <LBFGS.h>
using namespace LBFGSpp;
#include "include/helpers.h"

/** GLOBAL VARIABLES **/

// Maximum step to take in inner Newton algorithm; step is halved if above this.
int GLOBAL_MAXSTEP = 3;

/** END GLOBAL VARIABLES **/

/**
 * Object to store the model
 * This reads in the data and provides all model-specific calculations:
 *  joint log likelihood and its derivatives
 * This includes the parameterization of the multivariate Normal.
 * This is for multivariate random effects; scalar random effects are
 * handled by a separate class, because they can use scalar data types.
 */

class model {
private:
  // Data 
  const std::vector<Eigen::VectorXd>& y;  // Response, length m, each vector length m_i
  const std::vector<Eigen::MatrixXd>& X;  // Group fixed effect design matrices, length m, dimensions m_i x p
  const std::vector<Eigen::MatrixXd>& Z;  // Group random effect design matrices, length m, dimensions m_i x d
  // Temporary
  double tmp = 0.;
  int itr = 0;
  Eigen::Vector2i idx;
  // Control params
  const List control;
  
public:
  // Parameters
  Eigen::VectorXd u; 								// Full mode, length N*d (N=num groups, d=random effect dimension)
  Eigen::VectorXd beta;							// Regression coefficients
  Eigen::VectorXd delta;            // Variance parameters
  Eigen::VectorXd phi;							// Covariance parameters
  
  // Dimensions
  int n, d, sb, sd, sp, p, st;
  
  // Intermediate quantities
  std::vector<Eigen::VectorXd> eta, etagrad;
  Eigen::VectorXi ni;
  Eigen::MatrixXd D, Su, Ln, hess, hessuu;
  Eigen::VectorXd Ltu, grad, step, gradu;
  std::vector<Eigen::MatrixXd> SuD, etahess, dHuu;
  double nll;
  
  // Control parameters- make public
  int maxitr;
  double tol, h;
  bool verbose;
  
  // Set parameters
  // Take in theta and set beta, delta, phi
  void set_params(const Eigen::VectorXd& theta) {
    beta = theta.segment(0,sb);
    delta = theta.segment(sb,d);
    phi = theta.segment(sb+d,sp);
  }
  void set_L() {
    // Set L from phi
    int k=0;
    for (int j=0;j<d;j++)
      for (int i=(j+1);i<d;i++) {
        Ln(i,j) = phi(k);
        // }
        k++;
      }
  }
  void set_D() {
    for (int j=0;j<d;j++)
      D.diagonal()[j] = exp(delta(j)); // NOT Square root 
  }
  void set_D_sqrt() {
    for (int j=0;j<d;j++)
      D.diagonal()[j] = exp(delta(j)/2.); // Square root 
  }
  void set_ij(int l) {
    int i=1,j=0;
    while(j<d) {
      if (d*(j) - (j+1)*(j)/2 + (i-j-1) == l) {
        idx << i,j;
      }
      i++;
      if (i > d) {
        j++;
        i=j+1;
      }
    }
  }
  // Constructor
  model(const std::vector<Eigen::VectorXd>& y_,
        const std::vector<Eigen::MatrixXd>& X_,
        const std::vector<Eigen::MatrixXd>& Z_,
        const List control_,
        Eigen::VectorXd u_,
        Eigen::VectorXd beta_,
        Eigen::VectorXd delta_,
        Eigen::VectorXd phi_) : 
    y(y_), X(X_), Z(Z_), control(control_), u(u_), beta(beta_), delta(delta_), phi(phi_) {
    // Initialize dimensions
    n = y.size(); // Number of groups
    d = Z[0].cols(); // Number of random effects parameters
    sb = beta.size();
    sd = delta.size();
    sp = phi.size();
    st = sb + d + sp; // Dimension of theta
    p = d + st; // Dimension of (u,theta);
    
    D.setZero(d,d);
    Su.setZero(d,sp);
    Ltu.setZero(d);
    grad.setZero(p);
    step.setZero(d); // Newton step
    hess.setZero(p,p);
    gradu.setZero(d);
    hessuu.setZero(d,d);
    Ln = Eigen::MatrixXd::Identity(d,d);
    
    eta.resize(n);
    etagrad.resize(n);
    etahess.resize(n);
    dHuu.resize(p);
    ni.resize(n);
    for (int i=0;i<n;i++) {
      ni(i) = y[i].size();
      eta[i].setZero(ni(i));
      etagrad[i].setZero(ni(i));
      etahess[i].setZero(ni(i),ni(i));
    }
    for (int i=0;i<p;i++)
      dHuu[i].setZero(d,d);
    
    SuD.resize(d);
    for (int i=0;i<d;i++) {
      SuD[i].setZero(d,sp);
      set_SuD(i);
    }
    
    // Extract control parameters
    tol = control["tol"];
    maxitr = control["maxitr"];
    h = control["h"];
    verbose = control["verbose"];
  }
  
  // Normal log density
  void update_Su(int l,Eigen::VectorXd u) {
    Su.setZero();
    int i=0,j=1;
    for (int k=0;k<sp;k++) {
      if ((k+1) % (d-i) == 0) {
        i++;
        j=i+1;
      }
      //Su(i,k) = u.segment(d*l,d)(j);
      Su(i,k) = u(j);
      j++;
    }
  }
  void update_Su(int l) {
    update_Su(l,u.segment(d*l,d));
  }
  void set_SuD(int l) {
    int i=0,j=1;
    for (int k=0;k<p;k++) {
      if ((k+1) % (d-i) == 0) {
        i++;
        j=i+1;
      }
      if (j == l)
        SuD[l](i,k) = 1.;
        j++;
    }
  }
  double normal_logdens(int i,Eigen::VectorXd u) {
    update_Su(i,u);
    set_D_sqrt();
    
    
    Ltu = D * (u + Su*phi);
    double out = -0.5*d*log(2*3.14159236) + delta.sum()/2.;
    out -= 0.5 * (Ltu.squaredNorm());
    return out;
  }
  double normal_logdens(int i) {
    return normal_logdens(i,u.segment(i*d,d));
  }
  void normal_grad(int i,Eigen::VectorXd u,bool only_u) {
    // Gradient of NEGATIVE log likelihood
    // Set L from current phi
    set_L();
    set_D();
    update_Su(i,u);
    Ltu = u + Su*phi;
    
    // increment u segment
    grad.segment(0,d) += Ln * D * Ltu; 
    if (!only_u) {
      // no beta segment
      // increment delta segment
      for (int j=0;j<d;j++) {
        grad.segment(d+sb,d)(j) -= 0.5 * (1 - D(j,j)*pow(Ltu(j),2.));
      }
      // increment phi segment
      grad.segment(d+sb+d,sp) += (Su.transpose()) * D * Ltu;
    }
    
  }
  void normal_grad(int i,bool only_u) {
    normal_grad(i,u.segment(i*d,d),only_u);
  }
  void normal_grad(int i) {
    normal_grad(i,u.segment(i*d,d),false);
  }
  
  void normal_hess(int i,bool only_u) {
    set_L();
    set_D();
    update_Su(i);
    Ltu = u.segment(d*i,d) + Su*phi;
    
    // (u,u) block
    hess.block(0,0,d,d) = Ln * D * (Ln.transpose());
    for (int j=0;j<d;j++) {
      // (delta,delta) block
      hess(d+sb+j,d+sb+j) = 0.5 * exp(delta(j)) * Ltu(j) * Ltu(j);
      // (u,delta) cross
      hess.block(0,d+sb+j,d,1) = exp(delta(j)) * Ltu(j) * Ln.col(j);
      // (u,phi) cross
      //	hess.block(j,d+sb+d,sp,1) = (SuD[j].transpose())*D*u.segment(i*d,d) + ((Su.transpose())*D).col(j) + (SuD[j].transpose())*D*Su*phi + (Su.transpose())*D*(SuD[j])*phi; 
      hess.block(j,d+sb+d,sp,1) = (SuD[j].transpose())*D*u.segment(i*d,d) + (SuD[j].transpose())*D*Su*phi + (Su.transpose())*D*(SuD[j])*phi; 
    }
    hess.block(0,d+sb+d,d,sp) += D * Su;
    hess.block(d+sb,0,d,d) = hess.block(0,d+sb,d,d).transpose();
    hess.block(d+sb+d,0,sp,d) = hess.block(0,d+sb+d,d,sp).transpose();
    // (phi,phi) block
    hess.block(d + sb + d,d + sb + d,sp,sp) = (Su.transpose()) * D * Su;
    for (int j=0;j<sp;j++) {
      // (phi,delta) cross
      hess.block(d+sb,d+sb+d+j,d,1) = exp(delta(j)) * Ltu(j) * Su.col(j);
    }
    hess.block(d+sb+d,d+sb,sp,d) = hess.block(d+sb,d+sb+d,d,sp).transpose();
  }
  // Delta method for Wald intervals
  bool check_ik(int i,int k, int l) {
    k++; // 0-based vs 1-based
    i++;
    l++;
    if ((l == d*(k-1) - k*(k-1)/2 + (i-k)) & (i>k)) {
      return true;
    } else {
      return false;
    }
  }
  Eigen::MatrixXd compute_varcomp_sd(const Eigen::MatrixXd& covmat) {
    // get covariance matrix elements and their standard deviations 
    // using the Delta method
    // covmat is covmat of (delta,phi), already subsetted from the full one.
    set_D();
    set_L();
    Eigen::VectorXd Dij(d+sp);
    Dij.setZero();
    
    Eigen::MatrixXd H(d,d), S(d,d), out(d+sp,2);
    out.setZero();
    H = Ln * D * (Ln.transpose());
    S = H.inverse();
    
    for (int j=0;j<d;j++)
      //out(j,0) = S(j,j); // variances
      out(j,0) = log(S(j,j)); // log variances
    
    int k=d;
    for (int j=0;j<d;j++) {
      for (int i=(j+1);i<d;i++) {
        out(k,0) = S(i,j); // covariances
        k++;
      }
    }
    
    /** derivatives **/
    // H derivative
    std::vector<Eigen::MatrixXd> dHdt(d+sp), dSdt(d+sp);
    for (int s=0;s<d+sp;s++) {
      dHdt[s].setZero(d,d);
      dSdt[s].setZero(d,d);
    }
    for (int j=0;j<d;j++) {
      for (int i=j;i<d;i++) {
        // delta
        for (int k=0;k<d;k++) {
          dHdt[k](i,j) = exp(delta(k)) * Ln(i,k) * Ln(j,k);
          dHdt[k](j,i) = dHdt[k](i,j);
        }
        // phi
        for (int l=0;l<sp;l++) {
          for (int k=0;k<d;k++) {
            if (check_ik(i,k,l)) {
              dHdt[d+l](i,j) += exp(delta(k)) * Ln(j,k);
            }
            if (check_ik(j,k,l)) {
              dHdt[d+l](i,j) += exp(delta(k)) * Ln(i,k);
            }
            dHdt[d+l](j,i) = dHdt[d+l](i,j);
          }
        }
      }
    }
    // S derivative
    for (int k=0;k<d+sp;k++) {
      dSdt[k] = -S * dHdt[k] * S;
    }
    // sigma element derivative
    // log standard deviations:
    for (int j=0;j<d;j++) {
      // fill Djj
      for (int k=0;k<d+sp;k++) {
        Dij(k) = dSdt[k](j,j);
      }
      // compute the standard deviation of the log variance
      out(j,1) = sqrt(Dij.dot(covmat * Dij)) * exp(-out(j,0));
    }
    // covariances
    int idx=d;
    for (int j=0;j<d;j++) {
      for (int i=(j+1);i<d;i++) {
        // fill Dij
        for (int k=0;k<d+sp;k++) {
          Dij(k) = dSdt[k](i,j);
        }
        // standard deviation of the covariance
        out(idx,1) = sqrt(Dij.dot(covmat * Dij));
        idx++;
      }
    }
    
    
    
    return out;
  } 
  // Log likelihood
  double loglik_i(int i,Eigen::VectorXd u) {
    // Compute the log-likelihood for group i
    //eta[i] = X[i]*beta + Z[i]*u.segment(d*i,d);
    eta[i] = X[i]*beta + Z[i]*u;
    double out=0;
    for (int j=0;j<ni(i);j++) {
      out += y[i](j)*eta[i](j) - log1p(exp(eta[i](j)));
    }
    
    // TODO: this only works for u = current mode!!!
    out += normal_logdens(i,u);
    
    return -out; // Negative log-likelihood
  }
  double loglik_i(int i) {
    return loglik_i(i,u.segment(i*d,d));
  }
  // Gradient
  void grad_i(int i,Eigen::VectorXd u,bool only_u) {
    grad.setZero();
    // TODO: this only works for u = current mode!!!
    normal_grad(i,u,only_u);
    
    //eta[i] = X[i]*beta + Z[i]*u.segment(d*i,d);
    eta[i] = X[i]*beta + Z[i]*u;
    for (int j=0;j<ni(i);j++) {
      etagrad[i](j) = -y[i](j) + 1. / (1. + exp(-eta[i](j)));
    }	
    // Increment u
    grad.segment(0,d) += (Z[i].transpose()) * etagrad[i];
    if (!only_u) {
      // Increment beta
      grad.segment(d,sb) += (X[i].transpose()) * etagrad[i];
    }
  }
  void grad_i(int i,bool only_u) {
    return grad_i(i,u.segment(d*i,d),only_u);
  }
  // Log likelihood and gradient together
  void loglik_grad_i(int i,Eigen::VectorXd u) {
    //nll = loglik_i(i,u);
    //grad_i(i,u,false);
    grad.setZero();
    eta[i] = X[i]*beta + Z[i]*u;
    nll = 0;
    for (int j=0;j<ni(i);j++) {
      nll -= y[i](j)*eta[i](j) - log1p(exp(eta[i](j)));
      etagrad[i](j) = -y[i](j) + 1. / (1. + exp(-eta[i](j)));
    }
    grad.segment(0,d) += (Z[i].transpose()) * etagrad[i];
    grad.segment(d,sb) += (X[i].transpose()) * etagrad[i];
    
    // Normal
    set_L();
    set_D();
    update_Su(i,u);
    Ltu = u + Su*phi;
    // value
    nll += 0.5*(Ltu.transpose()) * D * Ltu;
    nll -= -0.5*d*log(2*3.14159236) + delta.sum()/2.;
    // u
    grad.segment(0,d) += Ln * D * Ltu; 
    // delta
    for (int j=0;j<d;j++) {
      grad(d+sb+j) -= 0.5 * (1. - D(j,j)*pow(Ltu(j),2.));
    }
    // phi 
    grad.segment(d+sb+d,sp) += (Su.transpose()) * D * Ltu;
  }
  // Hessian
  void hess_i(int i,bool only_u) {
    hess.setZero();
    normal_hess(i,only_u);
    eta[i] = X[i]*beta + Z[i]*u.segment(d*i,d);
    for (int j=0;j<ni(i);j++) {
      tmp = exp(eta[i](j)) / (1. + exp(eta[i](j)));
      etahess[i](j,j) = tmp * (1.- tmp);
    }
    // (u,u) block
    hess.block(0,0,d,d) += (Z[i].transpose()) * (etahess[i] * Z[i]);
    // (beta,beta) block
    hess.block(d,d,sb,sb) += (X[i].transpose()) * (etahess[i] * X[i]);
    // (u,beta) cross blocks
    hess.block(0,d,d,sb) += (Z[i].transpose()) * (etahess[i] * X[i]);
    hess.block(d,0,sb,d) = hess.block(0,d,d,sb).transpose();
  }
  // Inner optimization
  void newton_step(int i) {
    // directly compute a single newton step, efficiently
    // requires g and H; these share some quantities though
    // also, only requires the u segment/(u,u) block
    gradu.setZero();
    hessuu.setZero();
    
    eta[i] = X[i]*beta + Z[i]*u.segment(d*i,d);
    for (int j=0;j<ni(i);j++) {
      etagrad[i](j) = -y[i](j) + 1. / (1. + exp(-eta[i](j)));
      tmp = exp(eta[i](j)) / (1. + exp(eta[i](j)));
      etahess[i](j,j) = tmp * (1.- tmp);
    }	
    
    gradu += (Z[i].transpose()) * etagrad[i];
    hessuu += (Z[i].transpose()) * (etahess[i] * Z[i]);
    
    // normal
    set_L();
    set_D();
    update_Su(i);
    Ltu = u.segment(i*d,d) + Su*phi;
    
    gradu += Ln * D * Ltu; 
    hessuu += Ln * D * (Ln.transpose());
    
    step = hessuu.ldlt().solve(-gradu);
    
  }
  void inner_optimize_i(int i,bool verbose) {
    // compute the newton steps directly
    itr = 0;
    while(true) {
      if (verbose & (itr < 3)) {
        std::cout << "NEW newton iteration " << itr << ", step = " << step.transpose() << ", new u = " << u.segment(i*d,d) << ", gradnorm = " << gradu.norm() << std::endl;
        std::cout << "gradu: " << gradu.transpose() << std::endl;
        std::cout << "hessuu: " << hessuu.transpose() << std::endl;
      }
      // compute the step
      newton_step(i);
      // step halving
      while(step.norm()>GLOBAL_MAXSTEP) step /= 2.; 
      u.segment(i*d,d) += step;
      if (gradu.norm() < tol | itr > maxitr) break;
      itr++;
    }
  }
  
  // Third derivatives...
  void dHuu_i(int i) {
    for (int j=0;j<p;j++)
      dHuu[j].setZero();
    set_L();
    set_D();
    eta[i] = X[i]*beta + Z[i]*u.segment(d*i,d);
    for (int j=0;j<ni(i);j++) {
      tmp = exp(eta[i](j)) / (1. + exp(eta[i](j)));
      etagrad[i](j) = tmp * (1.- tmp) * (1. - 2.*tmp);
    }
    // u
    for (int j=0;j<d;j++) {
      etahess[i].diagonal() = etagrad[i].array() * Z[i].col(j).array();
      dHuu[j] = (Z[i].transpose()) * (etahess[i] * Z[i]);
    }
    // beta
    for (int j=0;j<sb;j++) {
      etahess[i].diagonal() = etagrad[i].array() * X[i].col(j).array();
      dHuu[d+j] = (Z[i].transpose()) * (etahess[i] * Z[i]);
    }
    // delta
    for (int j=0;j<d;j++) {
      dHuu[d+sb+j] = exp(delta(j)) * Ln.col(j) * Ln.col(j).transpose();
    }
    // phi
    for (int l=0;l<sp;l++) {
      // get the index of l	
      set_ij(l);
      dHuu[d+sb+d+l].row(idx(0)) = (Ln*D).col(idx(1));
      dHuu[d+sb+d+l].col(idx(0)) += dHuu[d+sb+d+l].row(idx(0));
    }
  }
};

/** Log-likelihood and gradient
 * Evaluate the log likelihood and its gradient,
 * for input to the BFGS methods
 * 
**/

class margloglik {
  
private:
  model& modelobj;      // Model object, above
  const Eigen::MatrixXd& nn;  // Quadrature nodes
  const Eigen::VectorXd& ww;  // Quadrature weights
  int evalcount=0, idx=0;
  
  Eigen::MatrixXd Li;
  Eigen::LLT<Eigen::MatrixXd> LLt;
  
  double aghqnll;
  Eigen::VectorXd aghqgrad, aghqgradperturb, aghqthetaperturb, aghqstep;
  Eigen::MatrixXd aghqhess;
  
  Eigen::VectorXd grad_h_L;          // Gradient of Cholesky term with respect to theta
  Eigen::VectorXd grad_h_ut_noL;     // Gradient ignoring L
  Eigen::VectorXd grad_h_ut;         // Gradient wrt (u,theta) ignoring mode
  Eigen::VectorXd grad_h_t;          // Gradient taking mode into account
  Eigen::VectorXd val_vec;           // Vector of weighted log-likelihood evaluations for a single group
  Eigen::MatrixXd Fi;        // Derivative matrix of g with respect to L; F_ij = dg/dL_ij
  Eigen::MatrixXd dLinvzmat; // Intermediate matrix of derivatives of the forward solve
  Eigen::MatrixXd W_L;       // Intermediate weight matrix to be summed to give F
  Eigen::MatrixXd W_u;       // Intermediate weight matrix to be summed to give dg/du; also storage for gradients
  Eigen::MatrixXd zA;        // Adapted quadrature points
  Eigen::VectorXd tmpZ;      // Placeholder for quadrature nodes within forward sub algorithm 
  int kd;
  
  
public:
  Eigen::VectorXd raw_sd; 
  Eigen::MatrixXd vcomp_sd;
  margloglik(model& modelobj_, const Eigen::MatrixXd& nn_, const Eigen::VectorXd& ww_) : modelobj(modelobj_), nn(nn_), ww(ww_) {
    Li.setZero(modelobj.d,modelobj.d);
    kd = ww.size();
    
    aghqgrad.setZero(modelobj.st);
    aghqgradperturb.setZero(modelobj.st);
    aghqthetaperturb.setZero(modelobj.st);
    aghqstep.setZero(modelobj.st);
    aghqhess.setZero(modelobj.st,modelobj.st);
    
    zA.setZero(kd,modelobj.d);
    grad_h_L.setZero(modelobj.st);
    grad_h_ut_noL.setZero(modelobj.p);
    grad_h_ut.setZero(modelobj.p);
    grad_h_t.setZero(modelobj.st);
    val_vec.setZero(kd); 
    Fi.setZero(modelobj.d,modelobj.d); 
    dLinvzmat.setZero(modelobj.d,modelobj.d*(modelobj.d+1)/2);
    W_L.setZero(kd,modelobj.d*(modelobj.d+1)/2);
    W_u.setZero(kd,modelobj.p);
    tmpZ.setZero(modelobj.d);
    
    raw_sd.setZero(modelobj.st);
    vcomp_sd.setZero(modelobj.d + modelobj.sp,2);
  }
  // getters
  Eigen::MatrixXd get_hessian() {
    return aghqhess;
  }
  // wald intervals
  
  void compute_all_sd() {
    // compute the sd for wald intervals for raw and transformed params
    Eigen::MatrixXd covmatfull = -aghqhess.inverse();
    raw_sd = covmatfull.diagonal().cwiseSqrt();
    vcomp_sd = modelobj.compute_varcomp_sd(covmatfull.bottomRightCorner(modelobj.d+modelobj.sp,modelobj.d+modelobj.sp));
  }
  
  // Forward sub D without copies
  // Directly overwrites dLinvzmat 
  void forwardsubD(int j) {
    tmpZ = nn.row(j); // Copy, to be overwritten inside algorithm
    int ip=0,jp=0;
    dLinvzmat.setZero();
    for (int i=0;i<modelobj.d;i++) {
      for (int j=0;j<i;j++) {
        tmpZ(i) -= Li(i,j)*tmpZ(j);
        ip=0;
        jp=0;
        for (int id=0;id<dLinvzmat.cols();id++) {
          dLinvzmat(i,id) -= Li(i,j)*dLinvzmat(j,id);
          if (ip==i && jp==j)
            dLinvzmat(i,id) -= tmpZ(j);
          jp++;
          if(jp>ip) {
            jp=0;
            ip++;
          }
        }
      }
      tmpZ(i) /= Li(i,i);
      ip=0;
      jp=0;
      for (int id=0;id<dLinvzmat.cols();id++) {
        dLinvzmat(i,id) /= Li(i,i);
        if (ip==i && jp==i)
          dLinvzmat(i,id) -= tmpZ(i)/Li(i,i);
        jp++;
        if(jp>ip) {
          jp=0;
          ip++;
        }
      }
    }
  }
  // Stack a vector into a lower triangular matrix
  void get_L_from_vector(Eigen::MatrixXd& F, Eigen::VectorXd Lvec) {
    // put vector Lvec into matrix F
    F.setZero();
    int tmpidx=0;
    for (int j=0;j<modelobj.d;j++)
      for (int i=j;i<modelobj.d;i++) {
        F(i,j) = Lvec(tmpidx);
        tmpidx++;
      }
  }
  double operator()(const Eigen::VectorXd& theta,Eigen::VectorXd& grad) {
    // Overwrite grad with gradient at theta
    // Return negative log likelihood
    evalcount++;
    double loglik = 0., loglik_tmp = 0., logdet = 0.;
    grad.setZero();
    /** Set modelobj parameters **/
    modelobj.set_params(theta);
    
    
    /** Loop over i **/
    bool verbose = false;
    for (int i=0;i<modelobj.n;i++) {
      loglik_tmp = 0.;
      //// Inner Optimization
      //if (std::find(std::begin(ITOPRINT),std::end(ITOPRINT),i) != std::end(ITOPRINT)) {
      //	verbose = true;
      //} else {
      //	verbose = false;
      //}
      modelobj.inner_optimize_i(i,verbose);
      // Hessian
      // TODO: set this within the newton iteration
      modelobj.hess_i(i,false);
      LLt.compute(modelobj.hess.block(0,0,modelobj.d,modelobj.d));
      Li = LLt.matrixL();
      // Third derivatives
      modelobj.dHuu_i(i);
      logdet = 0;
      loglik_tmp = 0.;
      for (int j=0;j<modelobj.d;j++)
        logdet += log(Li(j,j));
      //loglik_tmp -= log(Li(j,j));
      
      // Adaptation
      zA = Li.triangularView<Eigen::Lower>().solve(nn.transpose());
      zA.colwise() += modelobj.u.segment(i*modelobj.d,modelobj.d);
      
      // Loop over quadrature points
      val_vec.setZero();
      W_L.setZero();
      for (int j=0;j<kd;j++) {
        // Joint log-likelihood and gradient evaluation
        modelobj.loglik_grad_i(i,zA.col(j));
        val_vec(j) = -modelobj.nll + log(ww(j)) - logdet;
        //val_vec(j) = -modelobj.nll + log(ww(j));
        W_u.row(j) = -modelobj.grad;
        // Cholesky gradient
        forwardsubD(j); // Overwrites dLinvzmat
        W_L.row(j) += (dLinvzmat.transpose()) * (((W_u.row(j)).segment(0,modelobj.d)).transpose());
        idx=0;
        for (int l=0;l<modelobj.d;l++)
          for (int k=l;k<modelobj.d;k++) {
            if (l==k)
              W_L(j,idx) -= 1/Li(k,k);
            idx++;
          }
      }
      // Log-likelihood, for scaling
      loglik_tmp += logsumexp(val_vec);
      loglik -= loglik_tmp; // negative log-likelihood 
      /** Collate **/
      grad_h_ut_noL = vector_sumexp(W_u,val_vec);
      get_L_from_vector(Fi,vector_sumexp(W_L,val_vec));
      grad_h_L = dgdtheta(Li,Fi,modelobj.dHuu);
      grad_h_ut = grad_h_L + grad_h_ut_noL;
      /** Mode derivative **/
      grad_h_t = grad_h_ut.segment(modelobj.d,modelobj.st) + (LLt.solve(-modelobj.hess.block(0,modelobj.d,modelobj.d,modelobj.st)).transpose()) * (grad_h_ut.segment(0,modelobj.d));
      /** Scaling by marginal likelihood **/
      grad += -grad_h_t * exp(-loglik_tmp);
    }
    return loglik;
  }
  // Numeric hessian by forward difference
  void numerichessian(Eigen::VectorXd& theta) {
    // hessian of aghq log likelihood via finite difference
    
    // evaluate loglikelihood and gradient
    aghqnll = (*this)(theta,aghqgrad);
    // compute the hessian
    aghqgradperturb.setZero();
    aghqhess.setZero();
    for (int j=0;j<theta.size();j++) {
      // Perturb in the jth dimension
      aghqthetaperturb = theta;
      aghqthetaperturb(j) += modelobj.h;
      // Evaluate the gradient
      aghqnll = (*this)(aghqthetaperturb,aghqgradperturb);
      aghqhess.col(j) = (aghqgrad-aghqgradperturb)/modelobj.h;
    }
  }
  // Newton step
  void aghqnewtonstep(Eigen::VectorXd& aghqstep, Eigen::VectorXd& theta) {
    // overwrite aghqstep
    
    // evaluate the hessian and gradient
    numerichessian(theta);
    // overwrite the step
    aghqstep = aghqhess.ldlt().solve(aghqgrad); // NEGATIVE hessian --> NEGATIVE gradient!!
  }
  // Newton optimization
  void aghqnewton(Eigen::VectorXd& theta) {
    // overwrite theta with theta-hat
    int itr=0;	
    while(true) {
      if (modelobj.verbose & (itr < 3)) {
        std::cout << "aghq newton iteration " << itr << ", step = " << aghqstep.transpose() << ", new theta  = " << theta.transpose() << ", gradnorm = " << aghqgrad.norm() << std::endl;
        std::cout << "aghqgrad: " << aghqgrad.transpose() << std::endl;
        std::cout << "aghqhess: " << aghqhess.transpose() << std::endl;
      }
      // compute the step
      aghqnewtonstep(aghqstep,theta);
      // step halving
      while(aghqstep.norm()>GLOBAL_MAXSTEP) aghqstep /= 2.; 
      theta += aghqstep;
      if (aghqgrad.norm() < modelobj.tol | itr > modelobj.maxitr) break;
      itr++;
    }
  }
}; 

/**
 * Main function
 * Read in data from R, perform the mixed model computations,
 * and return results in a list.
 */
//' Fit an AGHQ model via L-BFGS optimization
//' 
//' This function is a C++ implementation of the exact gradient-based optimization
//' of the AGHQ approximate log-marginal likelihood in a binary mixed model.
//' 
//' @param theta Starting value for the outer parameter, containing regression
//' coefficients and variance components on the log-Cholesky scale.
//' @param u Starting value for the inner parameter, representing the mode of the
//' random effects vector.
//' @param y A \code{std::vector} of length \code{m} \code{Eigen} vectors of lengths \code{m_i},
//' containing the responses: one vector of within-group observations per group.
//' @param X A \code{std::vector} of length \code{m} \code{Eigen} matrices of dimensions \code{m_i x p},
//' representing the design matrices for the fixed effects variables.
//' @param Z A \code{std::vector} of length \code{m} \code{Eigen} matrices of dimensions \code{m_i x d},
//' representing the design matrices for the random effects.
//' @param nn An \code{Eigen} matrix of dimension \code{k^d x d} containing the product-rule quadrature nodes.
//' @param ww An \code{Eigen} vector of length \code{k^d} containing the product-rule quadrature weights.
//' @param control a \code{List} containing control arguments.
// [[Rcpp::export]]
List optimizeaghq(
    Eigen::VectorXd theta,          // Initial guess for theta
    Eigen::VectorXd u,              // Initial guess for (utheta)
    std::vector<Eigen::VectorXd> y, // Response, list, one vector per group
    std::vector<Eigen::MatrixXd> X, // Fixed effects covariates, list, one matrix per group
    std::vector<Eigen::MatrixXd> Z, // Random effects covariates, list, one matrix per group
    Eigen::MatrixXd nn,             // Matrix of base quadrature nodes
    Eigen::VectorXd ww,             // Matrix of base quadrature weights
    List control                    // List of control parameters   
) {
  // Extract control arguments
  /* Control arguments:
   * tol: tolerance for outer BFGS optimization
   * inner_tol: tolerance for inner newton optimization
   * maxitr: maximum iterations of outer BFGS optimization
   * inner_maxitr: maximum iterations of inner Newton optimization
   * bfgshist: number of iterations of gradient information to store for Hessian approximation
   * delta: convergence parameter for function change
   * past: number of iterations to compare againt for function change convergence
   * max_linesearch: maximum number of line search iterations
   */
  double tol       = control["tol"];
  double inner_tol = control["inner_tol"];                         // Tolerance for the inner optimization
  int inner_maxitr = control["inner_maxitr"];                // Maximum number of iterations for the inner optimization 
  String method    = control["method"];
  
  // Set up the model object
  int sb = X[0].cols(), d = Z[0].cols(), st = theta.size();
  int sp = st - (d+sb);
  model modelobj(y,X,Z,control,u,theta.segment(0,sb),theta.segment(sb,d),theta.segment(sb+d,sp));
  
  // Set up the optimization object
  margloglik nll(modelobj,nn,ww);
  Eigen::VectorXd grad(theta.size());
  grad.setZero();
  
  // Allow for just the gradient and negative log likelihood computation
  bool onlynllgrad = control["onlynllgrad"];
  if (onlynllgrad) {
    double nllval = nll(theta,grad);
    return List::create(Named("nll") = nllval,Named("grad") = grad);
  }
  
  /**
   * OPTIMIZATION
   * Newton or L-BFGS
   */
  double val=0.;
  // Newton
  if (method == "newton") {
    nll.aghqnewton(theta);
    val = nll(theta,grad);
  } else {
    LBFGSParam<double> param;
    param.m = control["bfgshist"];
    param.delta = control["bfgsdelta"];
    param.past = control["past"];
    param.epsilon = control["tol"];
    param.max_iterations = control["maxitr"];
    param.max_linesearch = control["max_linesearch"];
    LBFGSSolver<double,LineSearchNocedalWright> solver(param);
    
    // Run the minimization
    int niter = solver.minimize(nll, theta, val);
    val = nll(theta,grad);
    if (method == "both" & (grad.lpNorm<Eigen::Infinity>() > tol)) {
      // Now compute the newton steps
      nll.aghqnewton(theta);
      val = nll(theta,grad);
    } else {
      // compute the FD hessian
      nll.numerichessian(theta);
    }
  }
  /** END OPTIMIZATION **/
  
  // WALD Confidence intervals
  nll.compute_all_sd();
  Eigen::Matrix<double,Eigen::Dynamic,3> waldints(theta.size(),3);
  waldints.col(0) = theta - 2*nll.raw_sd;
  waldints.col(1) = theta;
  waldints.col(2) = theta + 2*nll.raw_sd;
  // variance components
  Eigen::Matrix<double,Eigen::Dynamic,3> waldintsvarcomp(modelobj.d+modelobj.sp,3);
  // convert back to sigma^2 scale
  // sigma^2_1
  waldintsvarcomp(0,1) = exp(nll.vcomp_sd(0,0));
  waldintsvarcomp(0,0) = exp((nll.vcomp_sd(0,0) - 2*nll.vcomp_sd(0,1)));
  waldintsvarcomp(0,2) = exp((nll.vcomp_sd(0,0) + 2*nll.vcomp_sd(0,1)));
  // sigma^2_2
  waldintsvarcomp(1,1) = exp(nll.vcomp_sd(1,0));
  waldintsvarcomp(1,0) = exp((nll.vcomp_sd(1,0) - 2*nll.vcomp_sd(1,1)));
  waldintsvarcomp(1,2) = exp((nll.vcomp_sd(1,0) + 2*nll.vcomp_sd(1,1)));
  // sigma_12
  waldintsvarcomp(2,1) = nll.vcomp_sd(2,0);
  waldintsvarcomp(2,0) = nll.vcomp_sd(2,0) - 2*nll.vcomp_sd(2,1);
  waldintsvarcomp(2,2) = nll.vcomp_sd(2,0) + 2*nll.vcomp_sd(2,1);
  
  return List::create(Named("method") = method,
                      Named("theta") = theta,
                      Named("H") = -nll.get_hessian(),
                      Named("betaints") = waldints.block(0,0,4,waldints.cols()),
                      Named("sigmaints") = waldintsvarcomp,
                      Named("nll") = val,
                      Named("grad") = grad
  );

}