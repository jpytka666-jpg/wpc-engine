# AIONS OS Integration module

Purpose: assemble the finished modules into one predictable organism without
coupling their internal implementations.

## Integration order

1. capability discovery;
2. resident runtime;
3. memory/KV services;
4. agent/repair services;
5. Studio control surface;
6. graph views;
7. Ghost Gate network boundary;
8. kernel/user-space integration.

## Gate rules

- integration is contract-driven;
- every dependency is explicit;
- module failure remains isolated;
- health is reported per module and globally;
- no deployment to the local machine until the integration CI is green.

## Next gate

Add a machine-readable module manifest and a global health contract. Then build
an integration workflow that fans out module checks and runs one final gate.
