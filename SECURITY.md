# Data and publication safety

This is an independent public portfolio repository.

- Every committed customer, transaction, value, outcome, and identifier is synthetic.
- Committed customer-level reports use HMAC-derived pseudonymous keys.
- No production query, table, schema, server, credential, internal threshold, or company result is
  included.
- Raw generated data, supplied inputs, model bundles, and run-specific output directories are
  excluded from version control.
- Supplied runs require `CHURN_ID_SALT`; the secret is read from the environment and never logged.
- Provenance contains a content checksum and bounded metadata, not a local source path or filename.
- CI scans tracked public files and tests source, supplied-history, operational-scoring, and
  non-editable wheel paths.

Do not open a public issue for an exposed secret or sensitive value. Contact the repository owner
privately and revoke the affected credential immediately.
