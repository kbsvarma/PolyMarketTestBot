from __future__ import annotations

from src.config import AppConfig
from src.models import GeoblockStatus


class GeoblockChecker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def preflight_status(self) -> GeoblockStatus:
        simulated = self.config.env.simulate_geoblock_status.lower()
        if simulated == "blocked":
            return GeoblockStatus(eligible=False, status="BLOCKED", detail="Configured geoblock simulation is blocked.")
        if simulated == "uncertain":
            return GeoblockStatus(eligible=False, status="UNCERTAIN", detail="Eligibility could not be determined with confidence.")

        country_ok = self.config.env.country_code in self.config.env.allowed_country_codes
        if not country_ok:
            return GeoblockStatus(
                eligible=False,
                status="BLOCKED",
                detail=f"Country {self.config.env.country_code} not in allowed country list.",
            )
        return GeoblockStatus(eligible=True, status="ELIGIBLE", detail="Eligibility preflight passed.")

    def live_trading_allowed(self) -> GeoblockStatus:
        status = self.preflight_status()
        if self.config.mode.value == "LIVE" and not status.eligible:
            return status
        return status
