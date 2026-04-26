"""
Optional Sentry integration.

Set ``SENTRY_DSN`` to enable error reporting. When unset, this module
is a no-op — production users opt in by setting the env var on the
deployment.

PII safety: ``send_default_pii=False`` keeps Sentry from collecting
request bodies, query strings, headers, or user IPs. Combined with
the regex-based redaction in ``api/pii.py`` (applied before MongoDB
storage), this means PII never leaves the box.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_sentry() -> None:
    """Initialise Sentry if ``SENTRY_DSN`` is set. Idempotent."""
    dsn = (os.getenv("SENTRY_DSN") or "").strip()
    if not dsn:
        logger.info("SENTRY_DSN not set; Sentry disabled")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
        environment        = os.getenv("SENTRY_ENVIRONMENT", "production")
        release            = os.getenv("SENTRY_RELEASE") or None

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=traces_sample_rate,
            environment=environment,
            release=release,
            send_default_pii=False,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
            ],
        )
        logger.info(
            "Sentry initialised",
            extra={"environment": environment, "traces_sample_rate": traces_sample_rate},
        )
    except ImportError:
        logger.warning("sentry-sdk not installed; SENTRY_DSN is set but ignored")
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Sentry init failed: %r", exc)
