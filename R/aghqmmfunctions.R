#' Control options for aghqmm
#' 
#' Control options for the \code{aghqmm} function, see that function.
#'
#' @param ... Used to override default arguments, see Details.
#' 
#' @details You can provide options for the following control arguments
#' to override their defaults:
#' 
#' TODO
#'
#' @family aghqmm
#' 
#' @export
aghqmm_control <- function(...) {
  # default control arguments
  out <- list(
    tol = 1e-08,
    maxitr = 100,
    inner_tol = 1e-08,
    inner_maxitr = 10,
    h = 1e-08,
    verbose = FALSE,
    bfgshist = 4,
    bfgsdelta = 1e-06,
    past = 3,
    max_linesearch = 100,
    method = "lbfgs",
    iter_EM = 0, # for GLMMadaptive only
    update_GH_every = 1, # for GLMMadaptive only
    onlynllgrad = FALSE,
    nllgradk = NULL # Compute the nll and grad approximations at an alternate value of k?
  )
  userargs <- list(...)
  for (arg in names(userargs)) out[arg] <- userargs[arg]
  out
}

#' Evaluate the approximate log-marginal likelihood and its gradient
#' 
#' Internal, not for user use.
#' 
#' @param theta Point at which to evaluate the approximate log-marginal likelihood and its gradient
#' @param formula An R mixed model formula pf the form \code{y ~ x*t + (t|id)},
#' compatible with \code{lme4::glmer}; see that function for details.
#' @param data A \code{data.frame} containing the variables found in \code{formula}.
#' @param k Order of the adaptive quadrature. In 1 dimension, \code{k} is the number of
#' points; in d-dimensions there are \code{k^d} total points.
logmarglik <- function(theta,formula,data,k=5) {
  # prepare data
  response <- all.vars(formula)[1]
  # formula = y ~ x*t + (t|id)
  # generate the data
  reterms <- lme4::glFormula(formula,data=data,family=stats::binomial) # TODO: update this for other families
  idvar <- names(reterms$reTrms$cnms)[1]
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  
  # check if response is 0/1 coded
  if(!all.equal(sort(unique(data[[response]])),c(0,1))) 
    stop(paste0("Response should be coded as numeric 0/1, yours is coded as",sort(unique(data[[response]])),". I don't know how to automatically convert it, sorry!"))
  
  
  modeldata <- with(reterms,list(
    X = X,
    Z = t(reTrms$Zt),
    y = data[[response]],
    group = data[[idvar]],
    id = as.numeric(table(data[[idvar]]))
  ))
  
  # Prepare the data for the likelihood functions
  yy <- with(modeldata,split(y,group))
  XX <- with(modeldata,lapply(split(X,group),matrix,ncol=ncol(X)))
  ZZ <- vector(mode = 'list',length = length(modeldata$id))
  d <- length(Reduce(c,reterms$reTrms$cnms)) # NOTE: not really tested
  for (i in 1:length(modeldata$id)) {
    row_id <- with(modeldata,sum(id[1:i]) + 1 - id[i]):with(modeldata,sum(id[1:i]))
    col_id <- (1 + (i-1)*d):(i*d)
    ZZ[[i]] <- as.matrix(modeldata$Z[row_id,col_id])
    colnames(ZZ[[i]]) <- NULL
  }
  
  # Quadrature
  gg <- mvQuad::createNIGrid(d,'GHe',k)
  nn <- mvQuad::getNodes(gg)
  ww <- mvQuad::getWeights(gg)
  # ustart <- rep(0,d*length(modeldata$id))
  
  # evaluate and return
  control <- aghqmm_control(onlynllgrad = TRUE)
  optimizeaghq(theta,yy,XX,ZZ,nn,ww,control)
}


#' Binary Mixed Model via AGHQ with Exact Gradients
#' 
#' Fit a binary mixed model using AGHQ with exact gradients of the
#' approximate log-marginal likelihood.
#' 
#' @param formula An R mixed model formula pf the form \code{y ~ x*t + (t|id)},
#' compatible with \code{lme4::glmer}; see that function for details.
#' @param data A \code{data.frame} containing the variables found in \code{formula}.
#' @param k Order of the adaptive quadrature. In 1 dimension, \code{k} is the number of
#' points; in d-dimensions there are \code{k^d} total points.
#' @param family \code{R glm} family; currently only \code{binomial()} is supported
#' and this assumes a binary response.
#' @param method Method to use to fit the model. \code{lbfgs} implements the novel
#' exact gradient-based quasi-Newton method from the paper; \code{newton} uses the
#' exact gradient to build a finite-differenced Hessian for use in Newton optimization.
#' The other options call others' software to facilitate comparison.
#' @param control Output of \code{aghqmm_control()}, see that function.
#' 
#' @details TODO
#' 
#' @return A list containing elements: TODO
#'
#' @examples 
#' library(mlmRev)
#' data("Contraception")
#' Contraception$response <- as.numeric(Contraception$use)-1
#' aghqmm::aghqmm(response ~ age + (age|district),data=Contraception)
#' 
#' \dontrun{
#' ## Not run
#' library(lme4)
#' glmer(use ~ age + urban + (age|district),data=Contraception,family = binomial)
#' library(GLMMadaptive)
#' mixed_model(use ~ age + urban,random = ~age|district,data=Contraception,family=binomial)
#' library(glmmEP)
#' reterms <- lme4::glFormula(formula,data=data,family=stats::binomial)
#' X <- reterms$X
#' Z <- X[ ,c(1,2)]
#' idvec <- reterms$reTrms$flist$district
#' glmmEP(as.numeric(Contraception$use)-1,X,Z,idvec) # note: singular
#' }
#' 
#' 
#' @family aghqmm
#' 
#' @export
#'
aghqmm <- function(
  formula,
  data,
  k=5,
  family=stats::binomial(),
  method = c("lbfgs","newton","both","GLMMadaptive","lme4","glmmEP"),
  control = aghqmm_control()) {
  
  tm <- Sys.time()
  
  method <- method[1]
  response <- all.vars(formula)[1]
  # formula = y ~ x*t + (t|id)
  # generate the data
  reterms <- lme4::glFormula(formula,data=data,family=stats::binomial) # TODO: update this for other families
  idvar <- names(reterms$reTrms$cnms)[1]
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  
  # check if response is 0/1 coded
  if(!all.equal(sort(unique(data[[response]])),c(0,1))) 
    stop(paste0("Response should be coded as numeric 0/1, yours is coded as",sort(unique(data[[response]])),". I don't know how to automatically convert it, sorry!"))
  
  
  modeldata <- with(reterms,list(
    X = X,
    Z = t(reTrms$Zt),
    y = data[[response]],
    group = data[[idvar]],
    id = as.numeric(table(data[[idvar]]))
  ))
  
  # Prepare the data for the likelihood functions
  yy <- with(modeldata,split(y,group))
  XX <- with(modeldata,lapply(split(X,group),matrix,ncol=ncol(X)))
  ZZ <- vector(mode = 'list',length = length(modeldata$id))
  d <- length(Reduce(c,reterms$reTrms$cnms)) # NOTE: not really tested
  for (i in 1:length(modeldata$id)) {
    row_id <- with(modeldata,sum(id[1:i]) + 1 - id[i]):with(modeldata,sum(id[1:i]))
    col_id <- (1 + (i-1)*d):(i*d)
    ZZ[[i]] <- as.matrix(modeldata$Z[row_id,col_id])
    colnames(ZZ[[i]]) <- NULL
  }
  
  # Quadrature
  gg <- mvQuad::createNIGrid(d,'GHe',k)
  nn <- mvQuad::getNodes(gg)
  ww <- mvQuad::getWeights(gg)
  
  # Optimize
  pardim <- ncol(modeldata$X) + d*(d+1)/2
  linmod <- stats::glm(lme4::nobars(formula),data,family=family)
  betastart <- stats::coef(linmod)
  betadim <- length(betastart)
  thetastart <- c(betastart,rep(0,d*(d+1)/2))
  # ustart <- rep(0,d*length(modeldata$id))
  preptime <- as.numeric(difftime(Sys.time(),tm,units='secs'))
  if (method %in% c("lbfgs","newton","both")) {
    tm <- Sys.time()
    control$method <- method
    # call appropriate function based on the number of random effects
    if (d==1) {
      opt <- optimizeaghqscalar(thetastart,yy,XX,as.numeric(nn),as.numeric(ww),control) 
    } else {
      opt <- optimizeaghq(thetastart,yy,XX,ZZ,nn,ww,control)
    }
    opttime <- as.numeric(difftime(Sys.time(),tm,units='secs'))
  } else if (method == "GLMMadaptive") {
    tm <- Sys.time()
    mod <- GLMMadaptive::mixed_model(
      fixed = lme4::nobars(formula),
      random = stats::as.formula(paste0("~",lme4::findbars(formula))),
      data = data,
      family = stats::binomial(),
      initial_values = list(betas = betastart,D = diag(d)),
      control = list(nAGQ = k,update_GH_every=control$update_GH_every,iter_EM = control$iter_EM)
    )
    betaest <- summary(mod)$coef_table[ ,'Estimate']
    covmatest <- summary(mod)$D
    sigmasqest <- diag(covmatest)
    if (d>1)
      covest <- covmatest[2,1]
    glmmaldl <- fastmatrix::ldl(solve(covmatest))
    glmmadelta <- log(glmmaldl$d)
    glmmaphi <- NULL
    if (d>1)
      glmmaphi <- glmmaldl$lower[2,1]
    if (d==1) {
      confintssigma <- stats::confint(mod,parm='var-cov')[ , ,drop=FALSE]
    } else {
      confintssigma <- stats::confint(mod,parm='var-cov')[c(1,3,2), ]
    }
    vc <- stats::vcov(mod)
    vcsd <- sqrt(diag(vc))
    betaints <- cbind(betaest - 2*vcsd[1:length(betaest)],betaest,betaest + 2*vcsd[1:length(betaest)])
    opt <- list(
      method = "GLMMadaptive",
      theta = c(betaest,glmmadelta,glmmaphi),
      H = solve(vc),
      betaints = betaints,
      sigmaints = confintssigma
    )
    opttime <- as.numeric(difftime(Sys.time(),tm,units='secs'))
  } else if (method == "lme4") {
    tm <- Sys.time()
    mod <- lme4::glmer(formula,data,family,nAGQ = k)
    ms <- summary(mod)
    betaest <- ms$coefficients[ ,1]
    betasd <- ms$coefficients[ ,2]
    betaints <- matrix(0,length(betaest),3)
    betaints[ ,1] <- betaest - 2*betasd
    betaints[ ,2] <- betaest
    betaints[ ,3] <- betaest + 2*betasd
    
    covmatest <- ms$varcor[[idvar]]
    lme4ldl <- fastmatrix::ldl(solve(covmatest))
    lme4delta <- log(lme4ldl$d)
    lme4phi <- NULL
    if (d>1)
      lme4phi <- lme4ldl$lower[2,1]
    opt <- list(
      method = "lme4",
      theta = c(betaest,lme4delta,lme4phi),
      betaints = betaints
    )
    opttime <- as.numeric(difftime(Sys.time(),tm,units='secs'))
  } else if (method == "glmmEP") {
    tm <- Sys.time()
    # construct the inputs
    Xf <- modeldata$X
    Xr <- as.matrix(flatten_bdmat(modeldata$Z,modeldata$id,d))
    mod <- glmmEP::glmmEP(modeldata$y,Xf,Xr,data[[idvar]])
    betaest <- mod$parameters[1:ncol(Xf),2]
    betaints <- as.matrix(mod$parameters[1:ncol(Xf), ])
    # sigma1^2, sigma2^2, sigma_12
    sigmaints <- mod$parameters[(ncol(Xf)+1):nrow(mod$parameters), ]
    if (d==1) {
      sigmaints <- sigmaints^2
    } else {
      sigmaints[3, ] <- sigmaints[3, ] * prod(sigmaints[1:2,2]) # corr --> cov
      sigmaints[1:2, ] <- sigmaints[1:2, ]^2 # variance
    }
    rownames(sigmaints) <- NULL
    
    # get delta and phi
    if (d==1) {
      S <- sigmaints[2]
      glmmepphi <- NULL
    } else {
      S <- diag(sigmaints[1:2,2])
      S[2,1] <- sigmaints[3,2]
    }
    glmmepldl <- fastmatrix::ldl(solve(S))
    glmmepdelta <- log(glmmepldl$d)
    if (d>1)
      glmmepphi <- glmmepldl$lower[2,1]
    
    opt <- list(
      method = "glmmEP",
      theta = c(betaest,glmmepdelta,glmmepphi),
      betaints = betaints,
      sigmaints = sigmaints
    )
    opttime <- as.numeric(difftime(Sys.time(),tm,units='secs'))
  }
  
  
  tm <- Sys.time()
  # add on nll and grad at a potentially more accurate value of k
  control$onlynllgrad <- TRUE
  # re-calculate the grid at a potentially more accurate value of k
  if (!is.null(control$nllgradk)) {
    gg <- mvQuad::createNIGrid(d,'GHe',control$nllgradk)
    nn <- mvQuad::getNodes(gg)
    ww <- mvQuad::getWeights(gg)
  }
  if (d==1) {
    nllandgrad <- optimizeaghqscalar(opt$theta,yy,XX,nn,ww,control)
  } else {
    nllandgrad <- optimizeaghq(opt$theta,yy,XX,ZZ,nn,ww,control)
  }
  opt$nll <- nllandgrad$nll
  opt$grad <- nllandgrad$grad
  posttime <- as.numeric(difftime(Sys.time(),tm,units='secs')) # not counted
  opt$normgrad_infinity <- max(abs(opt$grad)) # infinity norm
  opt$normgrad_2 <- sqrt(sum((opt$grad)^2)) # 2 norm
  
  # compute comp times according to method
  if (method %in% c("lbfgs","newton","both","glmmEP")) {
    comptime <- preptime + opttime
  } else if (method %in% c("GLMMadaptive","lme4")) {
    comptime <- opttime
  }
  opt$comptime <- comptime
  
  opt
}