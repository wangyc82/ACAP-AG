# ACAP-AG

The ACAP-AG model is to predict the anticancer activity of given compound from integrating multiple sources of data using attentive graph neural network.

ACAP-AG is capable to applied on various genetic scenarios, including cancer cell lines, patient-derived cancer cells, and cancer patients.

The ACAP-AG model includes two steps:

1. Representation of compounds and cancer cells or cancer patients.

   The embedding representation was achieved by a modified attentive graph neural network with integration of chemical structure, physicochemical properties, and target proteins of given compound.

   The cancer cells or patients were represented by their transcriptomic profiles.

2. Prediction of active compounds for anticancer therapy.

   The representations of compounds and cells/patients were used to construct kernel-based similarity matrices, and the fully connected deep neural network was introdcued to conduct the calssification task, that is, distinguish active compounds from those compounds with no anticancer activity.
