# The build helper verifies this installed base's ID and all source-tree labels.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG B12X_TREE
ARG B12X_HEAD
ARG DEPLOYMENT_COMMIT
COPY --from=b12x-source . /opt/spark3/candidate/b12x/
LABEL local.spark3.b12x.tree=${B12X_TREE} \
      local.spark3.b12x.patch-head=${B12X_HEAD} \
      org.opencontainers.image.revision=${DEPLOYMENT_COMMIT}
