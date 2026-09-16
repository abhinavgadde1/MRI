"""Back-compat shim — prefer ``preprocessing.skull_strip``."""

from preprocessing.skull_strip import SkullStripResult, hd_bet_available, skull_strip

__all__ = ["SkullStripResult", "hd_bet_available", "skull_strip"]
