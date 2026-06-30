"""Optional Chronos-2 availability and initialization check.

Install with:
    pip install -e .[chronos]

The script intentionally does not download or fine-tune weights in smoke mode.
"""

from support_capacity_reliability.forecasting.chronos_adapter import Chronos2Adapter

if __name__ == "__main__":
    status = Chronos2Adapter.availability()
    print(status)
    if not status.available:
        raise SystemExit(1)
    adapter = Chronos2Adapter("amazon/chronos-2")
    adapter.fit(None, [], "")  # type: ignore[arg-type]
    print("Chronos-2 initialized")
