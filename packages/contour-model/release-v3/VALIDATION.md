# Validation summary

The selected checkpoint is step 4,000 of the MPS warm-start run.

| Evaluation | Designs | Base macro F1 | Base accuracy | Compound macro F1 |
|---|---:|---:|---:|---:|
| Held-out test | 923 | 0.9904 | 0.9957 | 0.9937 |
| Generator stress/OOD | 499 | 0.9874 | 0.9949 | 0.9920 |
| Independent cohort | 1,000 | 0.9913 | 0.9963 | 0.9935 |

On the held-out test, line/arc/circle F1 is 0.9963 / 0.9748 / 1.0000.
On the independent cohort it is 0.9967 / 0.9772 / 1.0000.

Raw-edge instance validation on 200 held-out designs gives base-instance F1
0.9366 and compound-instance F1 0.9993 at IoU 0.5. Arc instance precision is
lower than token classification precision because the inference boundary head
can subdivide a reference arc near transitions.

These are synthetic-corpus results. They do not replace validation on real
captured contours.
