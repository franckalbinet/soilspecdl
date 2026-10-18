# Release notes

<!-- do not remove -->

## 0.1.3

### New Features

- Add `plot_cv` and a `stage` column to `cross_validate` ([#13](https://github.com/franckalbinet/soilspecdl/issues/13))


## 0.1.2

### New Features

- Add `cross_validate`, returning per-fold, per-epoch losses and metrics ([#12](https://github.com/franckalbinet/soilspecdl/issues/12))
- Add `get_kfold_splitters` for cross-validation ([#10](https://github.com/franckalbinet/soilspecdl/issues/10))

### Bugs Squashed

- `get_splitter` and `get_pre_splitter` reset the global NumPy random state ([#11](https://github.com/franckalbinet/soilspecdl/issues/11))


## 0.1.1

### New Features

- Add `spectral_learner`, with backbone and head parameter groups ([#9](https://github.com/franckalbinet/soilspecdl/issues/9))


## 0.1.0

### Breaking Changes

- Require soilspecdata 0.2.1, soilspectfm 0.1.1, fastai 2.8.12 and fastcore 2.2.22 ([#4](https://github.com/franckalbinet/soilspecdl/issues/4))
- Remove unused and dataset-specific API ([#3](https://github.com/franckalbinet/soilspecdl/issues/3))
- Store wavenumbers once in `dls.wns` instead of a `TensorSpectrum` class attribute ([#2](https://github.com/franckalbinet/soilspecdl/issues/2))
- Generalise the CNN: add `SpectralCNN`, make `MirzaiCNN` a configuration of it ([#1](https://github.com/franckalbinet/soilspecdl/issues/1))

### New Features

- Rewrite the docs front page and align the docs site with soilspectfm ([#7](https://github.com/franckalbinet/soilspecdl/issues/7))
- Add `soilspecdl.all` ([#6](https://github.com/franckalbinet/soilspecdl/issues/6))
- Add a bundled toy dataset, `load_toy_kssl` ([#5](https://github.com/franckalbinet/soilspecdl/issues/5))

### Bugs Squashed

- `get_splitter` never rejects `train_pct + valid_pct` above 1 ([#8](https://github.com/franckalbinet/soilspecdl/issues/8))
