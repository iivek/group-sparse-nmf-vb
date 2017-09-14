## Synopsis

A probabilistic formulation NMF with group sparsity prior. Learning is variational Bayesian.

## Example

To be added.

## About the Model

For more details about the model, please refer to [1].
A short note: even though the formulation allows for inferring labels of unlabeled data, even simultaneousy with learning other model parameters, in practice it can be observed that the inferred labels often get stuck in local minima early on in the training stage. For this reason, in [1] this algorithm has been used to learn label-driven data representations, while for classification *per se* k-NN has been chosen.

## References

If you have used this code, please cite the following paper,
[1] I. Ivek: "Interpretable Low-Rank Document Representations with Label-Dependent Sparsity Patterns", Proceedings of DMNLP Workshop at ECML/PKDD / Cellier P., Charnois T., Hotho A., Matwin S., Moens M.F., Toussaint Y.; 2014

## License

Published under GPL-3.0 License.