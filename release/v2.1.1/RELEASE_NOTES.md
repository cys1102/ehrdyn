# EHRDyn 2.1.1

This patch repairs packaging and provenance behavior only. It does not change
benchmark tasks, methods, environments, training or evaluation budgets,
scientific results, figures, or paper claims.

Changes:

- public default fixtures, controlled manifests, paper-task identity, and
  demonstration entrants are installed with the package;
- documented base, test, credentialed, and fitted dependency groups are
  separated;
- recursive world-model outputs retain the submitted entrant identifier;
- the five submitted EHR tasks are the default reader-facing identity, while
  the seven historical configuration files remain compatibility assets;
- documentation distinguishes capability checks from full numerical-result
  reproduction and identifies the six OPE estimators as built-in
  implementations.

The public 21-method smokes are bounded capability checks. They do not
regenerate the paper's complete 21-method by 40-environment result tables.
MIMIC-IV data and derived private artifacts remain credentialed and are not
redistributed.
