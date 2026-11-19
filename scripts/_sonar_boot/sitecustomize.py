"""Activate spark-ds41f's post-pin Sonar extensions in child processes."""
import os

if os.environ.get("SPARK_DS41F_SONAR_EXT") == "1":
    try:
        from spark_ds41f import sonar as _sonar_client

        _sonar_client.apply_deployment_extensions(include_main=False)
        _sonar_client.install_main_hook()
    except Exception:
        pass
