// Helpers for aghqmm
// Reverse mode AD of Cholesky
Eigen::VectorXd dgdtheta(Eigen::MatrixXd L,Eigen::MatrixXd F, std::vector<Eigen::MatrixXd> M) {
  int d = L.cols(), p = M.size();
  // Output
  Eigen::VectorXd out(p);
  out.setZero();

  /** Backwards recursions **/
  for (int k=d-1;k>=0;k--) {
    /** (a) Row Operations **/
    for (int j=k+1;j<d;j++) {
      for (int i=j;i<d;i++) {
        F(i,k) -= F(i,j)*L(j,k);
        F(j,k) -= F(i,j)*L(i,k);
      }
    }

    /** (b) Lead Column **/
    for (int j=k+1;j<d;j++) {
      F(j,k) /= L(k,k);
      F(k,k) -= L(j,k)*F(j,k);
    }

    /** (c) Pivot **/
    F(k,k) *= 0.5;
    F(k,k) /= L(k,k);
  }

  /** Populate the gradient **/
  for (int l=0;l<p;l++) {
    for (int j=0;j<d;j++) {
      for (int i=j;i<d;i++) {
        out(l) += F(i,j) * (M[l])(i,j);
      }
    }
  }

  return out;
}

// Scalar logsumexp with weights
double logsumexp(Eigen::VectorXd l) {
  // compute log(exp(pp1)*ww1 + ... + exp(ppd)*wwd)
  int n = l.size();
  if (n == 1) return l(0);
  
  std::sort(l.data(),l.data()+l.size(),std::greater<double>());
  Eigen::VectorXd S;
  S.setZero(n);
  S(0) = l(0);
  
  for (int i=0;i<(n-1);i++) 
    S(i+1) = S(i) + log1p(exp(-std::abs(l(i+1)-S(i))));
  
  return S(n-1);
}

// Vector sumexp with weights
Eigen::VectorXd vector_sumexp(Eigen::MatrixXd vv,Eigen::VectorXd w) {
  // compute sum vv_j*exp(ww) for very large, negative ww, for each column j of vv
  // vv(j) can be negative or zero so can't be logged, but is not assumed to be
  // very large/small
  int n = w.size(),m = vv.cols();
  Eigen::VectorXd out(m);
  out.setZero();
  double themaxpos = -INFINITY,themaxneg = -INFINITY;
  double pos = 0.,neg = 0.;
  
  for (int j=0;j<m;j++) {
    for (int i=0;i<n;i++) {
      if (vv(i,j) == 0) {
        // Do nothing
        continue;
      } else if (vv(i,j) > 0) {
        if (w(i) <= themaxpos) {
          pos += vv(i,j)*exp(w(i) - themaxpos);
        } else {
          pos *= exp(themaxpos - w(i));
          pos += vv(i,j);
          themaxpos = w(i);
        }
      } else if (vv(i,j) < 0) {
        if (w(i) <= themaxneg) {
          neg -= vv(i,j)*exp(w(i) - themaxneg);
        } else {
          neg *= exp(themaxneg - w(i));
          neg -= vv(i,j);
          themaxneg = w(i);
        }
      }
    }
    out(j) = exp(themaxpos)*pos - exp(themaxneg)*neg;
    themaxpos = -INFINITY;
    themaxneg = -INFINITY;
    pos = 0.;
    neg = 0.;
  }
  
  return out;
}