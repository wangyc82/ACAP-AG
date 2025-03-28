# ACAP-AG

The ACAP-AG model is to predict the anticancer activity of given compound from integrating multiple sources of data using attentive graph neural network.

ACAP-AG is capable to applied on various genetic scenarios, including cancer cell lines, patient-derived cancer cells, and cancer patients.

The independences:

pytorch framework: pandas,torch_geometric.data,torch.nn.functional,torch_geometric.nn,numpy

R: h2o,readr

The ACAP-AG model includes two steps:

1. Representation of compounds and cancer cells or cancer patients.

   The embedding representation was achieved by a modified attentive graph neural network with integration of chemical structure, physicochemical properties, and target proteins of given compound.

   This substep was performed via pytorch framework refered by getnoteAttrfromGAT.py. You have to prepare the MACCS keys for compounds as the note attributes for graph (GDSCandCTRP-drug-MACCkeys.csv, for instance), adjacent matrix for compound-compound interaction newtwork which was established by physicochemical properties (GDSCandCTRP-DDI-adj.csv, for example), and target proteins of given compounds. If you would like to do the pretraining, you have to provide the note properties as the output of AGNN (GDSCandCTRP-drug_polorea.csv, for example).  

   The cancer cells or patients were represented by their transcriptomic profiles.

3. Prediction of active compounds for anticancer therapy.

   The representations of compounds and cells/patients were used to construct kernel-based similarity matrices. the fully connected deep neural network was introdcued to conduct the calssification task, that is, distinguish active compounds from those compounds with no anticancer activity. The detail of this procedure can be found in our previous repository (DeepDRK).

   The steps was implemented via h2o R package, with hyperbolic tangent function (Tanh) as activation function, cross-entropy function was chosen as the loss function, two hidden layers and each layer limited to a maximum of 200 nodes.

The example to perform ACAP-AG model can be seen in ACAP.R, once you get embedding representations of compounds via AGNN (getnoteAttrfromGAT.py). In pariticular, replaced GDSCandCTRP-drug-MACCkeys.csv with GDSCandCTRP-NPASS_MACCSkeys.csv, GDSCandCTRP-DDI-adj.csv with GDSCandCTRP-NPASS-DDIcophy.csv, and GDSCandCTRP-drug_polorea.csv with GDSCandCTRP-NPASS-clusMember.csv,and then follow the precedure in ACAP.R.

The example is that using GDSCandCTRP combination data for training ACAP-AG model, and predicting anticancer activities for natural products in NAPSS data. Due to the limited target data, the physcichemical properties were integrated.
