# HDS-Copilot

 Quote Copilot, a retrieval-augmented system for CNC machin-
ing RFQs. The system parses drawings into structured job cards, retrieves similar
historical jobs, predicts a reference unit price with a learned regression model,
applies a quantity adjustment, and generates a schema-constrained quote package
with a large language model. It also supports a confidentiality-aware dual mode: in
AI mode, the PDF may be sent to a cloud model for extraction, while in manual
mode, only user-entered structured fields are shared. On a manually validated 20-
part holdout set, the final quantity-decoupled model achieved a mean absolute error
of $22.77 and mean absolute percentage error of 14.17%, with 85% of predictions
within 20% of ground truth. These results show that retrieval-grounded hybrid
pricing can produce both useful estimates and estimator-facing quote rationale.

Run Steps in Order
Change working directories accordingly 
