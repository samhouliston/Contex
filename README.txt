This repository contains my paper and implementation of ConTex: a post-hoc interpretability method which contructs a corpus of examples and studies feasible counterfactual examples.

The goal of this project was to improve on the SimplEx developed at the Van der Schaar Lab. The code and experiments heavily rely on the lab's original public repository. 

Abstract: 
Post-hoc interpretability approaches have been proven to be powerful tools to
generate explanations for the predictions made by a trained blackbox model. In
addition, the needs of end-users must be carefully considered when providing
useful and contextually-relevant explanations. In this work, we address this by
extending the SimplEx algorithm by Crabbé et al. [2021] which provides local
post-hoc explanations of a black-box prediction by extracting similar examples
from a corpus and by revealing the relative importance of their features. Our
method ContEx improves on SimplEx by providing both a contextually-relevant
corpus selection scheme and the study of feasible counterfactual examples.

The corpus selection scheme remains faithful to the underlying data distribution and
respect conditions specified by the user. We generate counterfactual examples with
the same principles via the FACE algorithm from [Poyiadzi et al., 2020], and then
apply SimplEx’s projected jacobians analysis to explain a sequence of examples
from the test point to the counterfactual. Such a sequence enables users to draw a
quantitatively interpretable story of the shortest realizable path from one prediction
to another type. 
