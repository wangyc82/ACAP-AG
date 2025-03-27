crossvalDNN<-function(X,Y,fold)
{
  # this procedure is to implement Random Forest in cross-validation way;
  # X is inputs
  # Y is outputs
  # fold is number of K for K-fold cross-validation
  # C is the cost of constraints violation in Standard SVM model
  # G is rbf kernel parameter 
  prdY<-vector()
  
  library(pracma)
  library(h2o)
  h2o.init()
  len<-nrow(X) # the number of input
  
  temp<-randperm(len)
  Xr<-X[temp,];Yr<-Y[temp];
  data<-data.frame(Xr,Yr)
  #data<-data.frame(X,Y)
  t<-floor(len/fold);
  for (k in 1:fold) {
    if (k==fold) {tk<-t*(k-1)+1;data_tst<-data[tk:len,];data_trn<-data[-(tk:len),]} else {tk1<-t*(k-1)+1;tk2<-t*k;data_tst<-data[tk1:tk2,];data_trn<-data[-(tk1:tk2),]}
    cat("convert training data \n")
    train.hex <- as.h2o(data_trn)
    cat("convert testing data \n")
    test.hex <- as.h2o(data_tst)
    cat("bulid the training model \n")
    model=h2o.deeplearning(x = 1:ncol(X), y = ncol(X)+1, training_frame = train.hex, hidden=c(200,200), epochs=10, activation="Tanh",seed = 10,reproducible = TRUE)
    cat("performing the prediction \n")
    model_prediction<-h2o.predict(model, test.hex)
    prob_test<-as.data.frame(model_prediction)[,1]
    prdY<-c(prdY,prob_test)
    cat(k,"\n")
  }
  h2o.shutdown()
  return(list(target=Yr, predY=prdY))
}
