# this procedure is to get predictions for newly given compounds based on the ACAP-AG framework

# suppose you already get the emdedding representation of compounds both in training set and testing set

# here taken example of NPASS data as testing data, and GDSCandCTRP combination data as training data

# AUCmat_comb_S2 is the drug-cell relationships obtained from GDSCandCTRP combination data
# responseNP_mat2 is anticancer activities (IC50) for natural products obtained from NPASS data
# sim.cell is kernel-based similarity matrix for cancer cell lines in GDSCandCTRP combination data

# load training data
load("~/Documents/ACAP-AG/train-example.RData")
# import the kernel-based cell similarity matrix
sim_cell <- read_csv("Documents/ACAP-AG/sim.cell.csv")
sim.cell<-data.matrix(sim_cell)
rownames(sim.cell)<-colnames(sim_cell)

library(readr)
GDSCandCTRP_NPASS_drugsGATrepresentation_cophyDDI <- read_csv("Documents/ACAP-AG/GDSCandCTRP-NPASS-drugsGATrepresentation-cophyDDI.csv")

#construct kernel-based similarity matrices
sim.drug.phy.GAT<-exp(-as.matrix(dist(GDSCandCTRP_NPASS_drugsGATrepresentation_cophyDDI[,-1])))
rownames(sim.drug.phy.GAT)<-GDSCandCTRP_NPASS_drugsGATrepresentation_cophyDDI$`Unnamed: 0`
colnames(sim.drug.phy.GAT)<-GDSCandCTRP_NPASS_drugsGATrepresentation_cophyDDI$`Unnamed: 0`

A<-which(AUCmat_comb_S2==1,arr.ind = TRUE)
B<-which(AUCmat_comb_S2==-1,arr.ind = TRUE)
Ast<-which(responseNP_mat2<=1 & responseNP_mat1!=0,arr.ind = TRUE)
Bst<-which(responseNP_mat2>100,arr.ind = TRUE)

sim.drug.trn<-sim.drug.phy.GAT[colnames(AUCmat_comb_S2),colnames(AUCmat_comb_S2)]
sim.drug.tst<-sim.drug.phy.GAT[colnames(responseNP_mat2),colnames(AUCmat_comb_S2)]

Xp<-cbind(sim.cell[A[,1],],sim.drug.trn[A[,2],])
Xn<-cbind(sim.cell[B[,1],],sim.drug.trn[B[,2],])
Xrn<-rbind(Xp,Xn)
Yrn<-rep(c(1,0),c(nrow(A),nrow(B)))
Xp.st<-cbind(sim.cell[Ast[,1],],sim.drug.tst[Ast[,2],])
Xn.st<-cbind(sim.cell[Bst[,1],],sim.drug.tst[Bst[,2],])
Xst<-rbind(Xp.st,Xn.st)
Yst<-rep(c(1,0),c(nrow(Ast),nrow(Bst)))
data_trn=cbind(Xrn,Yrn)
data_tst=cbind(Xst,Yst)
colnames(data_tst)<-colnames(data_tst)

library(h2o)
h2o.init()
train.hex <- as.h2o(data_trn)
test.hex <- as.h2o(data_tst)

model.DNN=h2o.deeplearning(x = 1:ncol(Xrn), y = ncol(Xrn)+1, training_frame = train.hex, hidden=c(200,200), epochs=10, activation="Tanh",seed = 10,reproducible = TRUE)
model_prediction<-h2o.predict(model.DNN, test.hex)
prob_test<-as.data.frame(model_prediction)[,1]

h2o.shutdown()



